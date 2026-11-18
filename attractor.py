"""Why the attractor forms, and what it is made of.

The terrain the marbles settle into is not an external landscape — it is a
kernel-density field over the run's own projected states.  A "basin" is
therefore a *pile-up*: many (layer, token) states landing in the same
neighborhood.  This module measures the mechanism behind that pile-up so
the UI can explain it instead of merely drawing it:

- **deceleration** — the per-layer residual-stream step of a tracked token
  peaks early and shrinks late; once steps are small, consecutive layers
  deposit near-identical states and density rises there;
- **constitution** — which (layer, token) states actually sit above a
  density threshold (the basin's membership roster);
- **meaning** — whether the logit-lens readout has stabilized and entropy
  has collapsed inside the basin, and (when components were captured)
  whether the settled writes come from attention or the MLP.

`analyze` produces the numbers; `explain` turns one report into prose.
"Attractor" here is descriptive geometry — where this run's states
accumulate — not a dynamical-systems claim.  Everything is a pure numpy
function over (StateTrajectory, coords, Landscape); no transformer
internals, any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import metrics as metrics_mod
from density import Landscape
from projection import fidelity_basis, low_fidelity_cut
from trajectory import StateTrajectory
from tuned_lens import EARLY_LAYER_CAVEAT, readout_name


@dataclass
class BasinReport:
    """Measured account of one run's density basin (see `analyze`)."""

    token: int                        # tracked token index
    n_states: int                     # total states in the run (L * T)
    center: np.ndarray                # (2,) projected coords of the density peak
    members: np.ndarray               # (M, 2) int (layer, token) states in the basin
    member_share: float               # M / n_states
    member_threshold: float           # density cut as a fraction of the peak
    layer_range: tuple[int, int] | None  # first / last member layer; None if no members
    late_share: float                 # fraction of members from the last half of layers
    step: np.ndarray                  # (L-1,) tracked token's hidden-space step per layer
    peak_step_layer: int              # layer receiving the largest step
    settle_layer: int | None          # layer from which every step stays small
    entropy: np.ndarray | None        # (L,) tracked token's predictive entropy
    readout_stable_from: int | None   # layer from which logit-lens top-1 never changes
    top_token: str | None             # the final top-1 token
    late_attn_share: float | None     # attention share of the settled residual writes

    @property
    def n_members(self) -> int:
        return len(self.members)


def density_at(landscape: Landscape, points_xy: np.ndarray) -> np.ndarray:
    """Normalized density interpolated at arbitrary (x, y) positions."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))[:, :2]
    interp = RegularGridInterpolator(
        (landscape.grid_y.astype(np.float64), landscape.grid_x.astype(np.float64)),
        landscape.density.astype(np.float64),
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.column_stack([pts[:, 1], pts[:, 0]]))


def members_of(coords: np.ndarray, landscape: Landscape,
               member_threshold: float = 0.5) -> np.ndarray:
    """(M, 2) int (layer, token) states whose interpolated density reaches
    `member_threshold` of the field's peak. coords: (L, T, 2+)."""
    pts = np.asarray(coords, dtype=np.float64)[..., :2]
    dens = density_at(landscape, pts.reshape(-1, 2)).reshape(pts.shape[:2])
    return np.argwhere(dens >= member_threshold * max(dens.max(), 1e-12)).astype(np.int64)


def analyze(
    traj: StateTrajectory,
    coords: np.ndarray,
    landscape: Landscape,
    token: int = -1,
    member_threshold: float = 0.5,
    settle_frac: float = 0.25,
) -> BasinReport:
    """Measure the basin of one run drawn on `landscape`.

    coords: (L, T, 2) projected states of this run (the landscape may have
    been built from a larger union, e.g. a multi-prompt scene).  Basin
    members are the states whose interpolated density reaches
    `member_threshold` of the field's peak; the run "settles" at the first
    layer from which every remaining step of the tracked token stays below
    `settle_frac` of its largest step.
    """
    token = int(token) % traj.n_tokens
    L, T = traj.n_layers, traj.n_tokens

    members = members_of(coords, landscape, member_threshold)
    layers = members[:, 0]
    layer_range = (int(layers.min()), int(layers.max())) if len(layers) else None
    late_share = float((layers >= L / 2).mean()) if len(layers) else 0.0

    peak = np.unravel_index(np.argmax(landscape.density), landscape.density.shape)
    center = np.array([landscape.grid_x[peak[1]], landscape.grid_y[peak[0]]],
                      dtype=np.float32)

    step = metrics_mod.layer_displacement(traj.hidden[:, token : token + 1, :])[:, 0]
    peak_step_layer = int(np.argmax(step)) + 1 if len(step) else 0  # step l lands in layer l+1
    settle_layer = None
    if len(step):
        small = settle_frac * step.max()
        for l in range(len(step)):
            if step[l:].max() <= small:
                settle_layer = l
                break

    ent = None if traj.entropy is None else np.asarray(traj.entropy[:, token], dtype=np.float64)

    top1 = [s.topk[0][0] if s.topk else None
            for s in (traj.state(l, token) for l in range(L))]
    readout_stable_from = top_token = None
    if top1[-1] is not None:
        top_token = top1[-1]
        readout_stable_from = L - 1
        while readout_stable_from > 0 and top1[readout_stable_from - 1] == top_token:
            readout_stable_from -= 1

    late_attn_share = None
    if traj.components is not None and {"attn", "mlp"} <= set(traj.components):
        shares = metrics_mod.component_shares(traj, token=token)
        if len(shares):  # empty for single-layer trajectories (no block writes)
            start = min(settle_layer if settle_layer is not None else L // 2, len(shares) - 1)
            late_attn_share = float(shares[start:, 0].mean())

    return BasinReport(
        token=token,
        n_states=L * T,
        center=center,
        members=members,
        member_share=float(len(members)) / (L * T),
        member_threshold=float(member_threshold),
        layer_range=layer_range,
        late_share=late_share,
        step=step,
        peak_step_layer=peak_step_layer,
        settle_layer=settle_layer,
        entropy=ent,
        readout_stable_from=readout_stable_from,
        top_token=top_token,
        late_attn_share=late_attn_share,
    )


def _fidelity_clause(report: BasinReport, quality) -> str:
    """A trust sentence generated from the projection's own distortion.

    A basin is a pile-up *in the 2-D projection*; if the states that form it
    lost their hidden-space neighborhoods on the way down, its shape is an
    artifact of the flattening, not structure in the model. This turns
    `projection_quality` into prose so the explanation never oversells the
    picture — the exact failure mode the tool is most at risk of. It reads
    what the ✕ markers read (`projection.fidelity_basis`), so the prose and
    the scene cannot disagree about the same states.
    """
    basis = fidelity_basis(quality)
    pres = basis.values
    if report.n_members:
        m = report.members
        basin_pres = float(pres[m[:, 0], m[:, 1]].mean())
    else:
        basin_pres = float(pres[:, report.token].mean())

    ev = quality.explained_variance
    var_phrase = f"keeps **{ev:.0%}** of the variance globally and " if ev is not None else ""
    within = basis.metric == "within-layer"
    lead = (f"**How much to trust this** — the projection {var_phrase}the states "
            f"that make up this basin retain **{basin_pres:.0%}** of their "
            f"{'within-layer ' if within else ''}hidden-space nearest neighbors "
            f"(k={basis.k}) on average"
            + (f", where a random placement would keep {basis.chance:.0%}. " if within
               else ", pooled across layers. "))
    if basin_pres < low_fidelity_cut(basis.chance):
        tail = (("That is no better than a random placement: " if within else
                 "That is low: ") +
                "read the basin's *shape* as suggestive, not established — much "
                "of the local structure did not survive the flattening.")
    # "high" is 0.8 of the way from chance to every neighbor kept (the plain
    # 0.8 for the pooled score, whose chance is taken as 0)
    elif basin_pres < basis.chance + 0.8 * (1.0 - basis.chance):
        tail = ("That is moderate: the basin's broad location is meaningful, its "
                "fine structure less so.")
    else:
        tail = "That is high, so the geometry you are looking at is a faithful view."
    return lead + tail


def explain(report: BasinReport, traj: StateTrajectory, quality=None) -> str:
    """One report → markdown prose: the design language doing explanatory work.

    Every sentence is generated from measurements in the report — nothing is
    canned lore about transformers in general. Pass `quality` (a
    `projection.ProjectionQuality`) to append a trust sentence measured from
    the projection's distortion, so the prose states how much of the basin's
    shape survived the flattening.
    """
    tok = traj.tokens[report.token]
    parts = [
        f"The terrain is made of the states themselves: a density field over "
        f"all **{report.n_states}** projected states of this run "
        f"({traj.n_layers} layers × {traj.n_tokens} tokens). Height is not an "
        f"external landscape the trajectory rolls over — a peak literally "
        f"means *many states landed here*."
    ]

    step = report.step
    if len(step):
        drop = 1.0 - step[-1] / max(step.max(), 1e-12)
        if report.settle_layer is not None:
            why = (
                f"**Why the basin forms** — token `{tok}`'s per-layer step "
                f"decelerates: it peaks at **{step.max():.2f}** (into layer "
                f"{report.peak_step_layer}) and drops to **{step[-1]:.2f}** "
                f"({drop:.0%} smaller); from layer **{report.settle_layer}** "
                f"it barely moves. Small steps mean consecutive layers deposit "
                f"near-identical states in one neighborhood, and the density "
                f"estimator raises a peak there — for this token, the pile-up "
                f"is a trajectory that has stopped travelling."
            )
        else:
            why = (
                f"**This token isn't what settles here** — `{tok}`'s per-layer "
                f"step peaks at **{step.max():.2f}** (into layer "
                f"{report.peak_step_layer}) and is still **{step[-1]:.2f}** at "
                f"the last layer ({drop:.0%} off peak, never small enough to "
                f"call settled). Whatever basin exists below is built from "
                f"*other* states in this run piling up in that neighborhood — "
                f"this token's own trajectory is passing through it, not "
                f"resting in it."
            )
        parts.append(why)

    if report.n_members == 0:
        parts.append(
            f"**What it is made of** — no state in this run reaches "
            f"{report.member_threshold:.0%} of peak density: at this "
            f"threshold there is no basin to describe."
        )
    else:
        a, b = report.layer_range
        parts.append(
            f"**What it is made of** — **{report.n_members} of {report.n_states}** "
            f"states (across every token in the run, not just `{tok}`) sit above "
            f"{report.member_threshold:.0%} of peak density: layers {a}–{b}, "
            f"with {report.late_share:.0%} of members from the second half of "
            f"the stack."
        )

    meaning = []
    if report.top_token is not None and report.readout_stable_from is not None:
        lens = readout_name(traj.meta)
        meaning.append(
            f"the {lens} top-1 is `{report.top_token}` from layer "
            f"**{report.readout_stable_from}** onward"
            + (f" ({EARLY_LAYER_CAVEAT})" if lens == "logit lens" else "")
        )
    if report.entropy is not None and len(report.entropy):
        meaning.append(
            f"predictive entropy has moved "
            f"{report.entropy[0]:.2f} → {report.entropy[-1]:.2f} nats"
        )
    if report.late_attn_share is not None:
        meaning.append(
            f"the settled residual writes are "
            f"{1 - report.late_attn_share:.0%} MLP / "
            f"{report.late_attn_share:.0%} attention"
        )
    if meaning:
        parts.append("**What it means** — inside the basin the computation is "
                     "largely decided: " + "; ".join(meaning) + ".")

    if quality is not None:
        parts.append(_fidelity_clause(report, quality))

    parts.append(
        "*\"Attractor\" here is descriptive geometry — where this run's "
        "states accumulate — not a dynamical-systems claim. The pull lives "
        "in the trained weights; the density of visits is how it becomes "
        "visible.*"
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# The basin null
# --------------------------------------------------------------------------
# The criterion, fixed here before any run is looked at (docs/validity.md asks
# for one declared in advance). Every result reports both.
NULL_DRAWS = 200
NULL_ALPHA = 0.05
# A token whose median state norm is this many times every token's median
# makes the step-direction null unfair: its directions, handed to ordinary
# tokens, scatter them, and the relabelled reference sets inherit its
# distance, so any basin of the rest looks concentrated. On five GPT-2 small
# prompts every raw-space basin "survived" because of position 0; with it
# dropped, or each state scaled to unit RMS, none did. Measured, raw /
# `center` gauge: GPT-2 position 0 at 33-35x / 4.0-9.9x, distilgpt2 23-25x,
# pythia-70m 4.0-4.8x; with position 0 removed, no token in any of them
# above 1.40x (synthetic walks 1.09-1.24x). The cut sits between.
SCALE_OUTLIER = 2.0
# A basin holding more than this share of the run's states is the typical
# state, not a concentration of states, and there is nothing to test. The
# explorer's half-peak basin is usually this broad: 81-87% of states on five
# GPT-2 small prompts with raw states, 57-85% with the sink gauged out. A
# smooth density over the whole cloud plateaus, and even 30 states pulled
# 99% of the way together did not shrink it below 71%.
MAX_REGION_SHARE = 0.5


@dataclass
class Concentration:
    """Whether a basin's members from different tokens sit closer together
    than the same pattern of states on other tokens (see `concentration`)."""

    status: str               # "tested" | "one-token" | "depth-band" | "no-members"
                              # | "scale-outlier" | "not-a-region" (basin_null only)
    n_member_tokens: int      # distinct token positions among the members
    distance: float | None    # mean distance between members of different tokens
    ratio: float | None       # distance / its mean over relabelled sets; < 1 is closer
    draws: int
    outlier_token: int | None = None    # the token that made it "scale-outlier"
    outlier_scale: float | None = None  # its median state norm / every token's median
    n_members: int = 0                  # basin size, in states
    n_states: int = 0                   # L * T


@dataclass
class BasinNull:
    """One run's basin against the step-direction null (see `basin_null`)."""

    observed: Concentration
    surrogate_ratios: np.ndarray | None  # (draws,); nan where a surrogate is not comparable
    p_surrogate: float | None            # share of comparable surrogates at least as concentrated
    verdict: str        # "survives" | "within-null" | "underpowered", or observed.status
    draws: int
    alpha: float
    seed: int


def concentration(space: np.ndarray, members: np.ndarray, draws: int = NULL_DRAWS,
                  rng: np.random.Generator | None = None) -> Concentration:
    """A statistic that can come back empty, which `members_of` cannot: every
    density field has a peak, so every run has states above half of it.

    space: (L, T, D) states in the space the projection was fitted on.
    Within one token, consecutive layers are close whenever its steps are
    small. That is deceleration, which `BasinReport.step` already measures,
    so only pairs of members from *different* tokens count here. The
    reference is the members' own (layer, token) pattern moved onto other
    tokens by a random relabelling of positions: the same layers and the
    same shape of set, so depth alone cannot produce a low ratio.

    The ratio is a statistic, not a test: the members were chosen for being
    dense, so they beat relabelled sets far more often than chance would
    (38% of runs at p <= 0.05 on synthetic walks with no basin planted).
    `basin_null` supplies the null, by putting surrogates through the same
    selection.

    Three cases cannot be tested and are reported as such: one token (there
    are no pairs), every token at the same layers (a band of depth: every
    relabelling returns the same set), and a token on another scale from the
    rest (`SCALE_OUTLIER`; scaling each state to unit RMS, the `rms` gauge,
    removes it).
    """
    from scipy.spatial.distance import cdist

    space = np.asarray(space, dtype=np.float64)
    T = space.shape[1]
    members = np.asarray(members, dtype=np.int64).reshape(-1, 2)
    layers_of = {t: frozenset(members[members[:, 1] == t, 0].tolist())
                 for t in np.unique(members[:, 1]).tolist()}

    def untestable(status, **outlier):
        return Concentration(status, len(layers_of), None, None, draws, **outlier,
                             n_members=len(members), n_states=space.shape[0] * T)

    scale = np.median(np.linalg.norm(space, axis=-1), axis=0)          # (T,)
    worst = int(np.argmax(scale))
    if scale[worst] >= SCALE_OUTLIER * max(float(np.median(scale)), 1e-12):
        return untestable("scale-outlier", outlier_token=worst,
                          outlier_scale=float(scale[worst] / np.median(scale)))
    if not layers_of:
        return untestable("no-members")
    if len(layers_of) < 2:
        return untestable("one-token")
    if len(layers_of) == T and len(set(layers_of.values())) == 1:
        return untestable("depth-band")

    flat = space.reshape(-1, space.shape[-1])
    dist = cdist(flat, flat)

    def cross_token(tokens):
        idx = members[:, 0] * T + tokens
        return float(dist[np.ix_(idx, idx)][tokens[:, None] != tokens[None, :]].mean())

    rng = np.random.default_rng(0) if rng is None else rng
    observed = cross_token(members[:, 1])
    relabelled = np.array([cross_token(rng.permutation(T)[members[:, 1]])
                           for _ in range(draws)])
    return Concentration(
        status="tested",
        n_member_tokens=len(layers_of),
        distance=observed,
        ratio=observed / max(float(relabelled.mean()), 1e-12),
        draws=draws,
        n_members=len(members),
        n_states=space.shape[0] * T,
    )


def step_direction_surrogate(hidden: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A run with the same starts and the same step sizes, going other ways.

    Each token keeps its layer-0 state and the size of every step it took;
    the direction of each step is another token's at the same layer. What
    survives is everything step sizes and a layer's shared directions
    produce: late layers piling up because steps shrink, a position-0 state
    thrown far from the rest, a drift every token shares. What is destroyed
    is which token moved which way.

    Directions are permuted, not drawn at random: an isotropic walk in a
    768-dimensional space is unlike any residual stream, so beating it would
    show only that the model is not noise.
    """
    hidden = np.asarray(hidden)
    steps = np.diff(hidden.astype(np.float64), axis=0)            # (L-1, T, D)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    units = steps / np.maximum(norms, 1e-12)
    donors = np.stack([rng.permutation(hidden.shape[1]) for _ in range(len(steps))])
    moved = np.take_along_axis(units, donors[:, :, None], axis=1) * norms
    return np.concatenate([hidden[:1], hidden[:1] + np.cumsum(moved, axis=0)]) \
        .astype(hidden.dtype)


def basin_null(space: np.ndarray, basin_of, draws: int = NULL_DRAWS,
               alpha: float = NULL_ALPHA, seed: int = 0) -> BasinNull:
    """Is this run's basin more than slowing steps and shared drift?

    space: (L, T, D) states the projection is fitted on (the run's gauged
    residual stream). `basin_of(space)` returns the basin's members under
    the run's own projection, density estimator and cut. The run and every
    surrogate go through it, so the projection is refitted each time and no
    surrogate is drawn on the real run's axes. Surrogates are built in this
    space, not the raw stream, so step sizes are those of the picture.

    A basin holding more than `MAX_REGION_SHARE` of the states is not
    tested ("not-a-region"). The surrogates are held to the same condition:
    only those whose own basin is a testable region are compared. Comparing
    a run selected for being localized with surrogates that mostly are not
    biased p low (median 0.22 on synthetic walks with no basin planted).
    With fewer than `1 / alpha - 1` comparable surrogates p cannot reach
    `alpha`, and the verdict is "underpowered".

    The basin survives when its cross-token concentration (`concentration`)
    is greater than in all but `alpha` of the comparable surrogates
    (`step_direction_surrogate`).

    Calibrated at the declared criterion on synthetic walks (the generator
    is `_run` in tests/test_attractor.py). With no basin planted, 2 of the
    37 runs that got a p-value survived, and p was near uniform (deciles
    0.08, 0.28, 0.52, 0.73, 0.84). With five of twelve tokens converging,
    19 of 19 did. Most runs got no p-value: 203 of 240 and 21 of 40 were
    not regions, depth bands or underpowered. So "within-null" is evidence,
    and an untested basin is not.

    Surviving shows a concentration these tokens reach in this run that step
    sizes do not explain. It does not show the model uses it: that takes an
    intervention.
    """
    rng = np.random.default_rng(seed)
    observed = concentration(space, basin_of(space), draws=draws, rng=rng)
    if observed.status == "tested" and \
            observed.n_members > MAX_REGION_SHARE * observed.n_states:
        observed = replace(observed, status="not-a-region", distance=None, ratio=None)
    if observed.status != "tested":
        return BasinNull(observed, None, None, observed.status, draws, alpha, seed)

    ratios = np.full(draws, np.nan)
    for i in range(draws):
        surrogate = step_direction_surrogate(space, rng)
        c = concentration(surrogate, basin_of(surrogate), draws=draws, rng=rng)
        if c.status == "tested" and c.n_members <= MAX_REGION_SHARE * c.n_states:
            ratios[i] = c.ratio
    comparable = ratios[np.isfinite(ratios)]
    if len(comparable) < 1 / alpha - 1:
        return BasinNull(observed, ratios, None, "underpowered", draws, alpha, seed)
    p = float((1 + (comparable <= observed.ratio).sum()) / (len(comparable) + 1))
    return BasinNull(observed, ratios, p, "survives" if p <= alpha else "within-null",
                     draws, alpha, seed)


def explain_null(null: BasinNull) -> str:
    """One `basin_null` result → markdown prose, in docs/validity.md's terms."""
    head = "**Is it more than slowing down?** — "
    obs = null.observed
    if obs.status == "no-members":
        return head + "there is no basin to test."
    if obs.status == "one-token":
        return head + ("every member of this basin belongs to one token position. "
                       "It is one trajectory whose steps have become small, which "
                       "the step curve already measures; there is no concentration "
                       "across tokens to test.")
    if obs.status == "depth-band":
        return head + ("the basin holds every token at the same layers. It is a "
                       "band of depth, not a region some tokens reach and others "
                       "do not, so there is nothing token-specific to test.")
    if obs.status == "not-a-region":
        return head + (
            f"not tested: the basin holds **{obs.n_members} of {obs.n_states}** "
            f"states ({obs.n_members / obs.n_states:.0%}). That is most of the "
            f"picture, the plateau of a smooth density, not a region some states "
            f"reach and others do not, so there is no concentration to test.")
    if obs.status == "scale-outlier":
        return head + (
            f"not tested: token {obs.outlier_token}'s states are "
            f"**{obs.outlier_scale:.0f}×** the norm of the rest (an attention sink "
            f"looks like this). The picture is mostly the other tokens seen from "
            f"far away, and swapping step directions with it would make any run "
            f"look concentrated. Set the gauge to `rms` or `rms+center` and test "
            f"again.")
    comparable = null.surrogate_ratios[np.isfinite(null.surrogate_ratios)]
    lead = (
        f"members from different tokens sit at **{obs.ratio:.2f}×** the hidden-space "
        f"distance of the same pattern of states on other tokens. Of {null.draws} "
        f"surrogate runs that keep every token's starting state and step sizes but "
        f"give each step another token's direction, {len(comparable)} had a basin "
        f"that is a region at all"
    )
    if null.verdict == "underpowered":
        return head + lead + (
            f", too few for a p-value that could reach α = {null.alpha:g}: "
            f"untested, not refuted.")
    lead += (f", and **{int((comparable <= obs.ratio).sum())}** of those were at "
             f"least as concentrated (p = {null.p_surrogate:.3f}). ")
    if null.verdict == "survives":
        return head + lead + (
            f"At α = {null.alpha:g} the concentration is not reproduced by step "
            f"sizes and shared directions: it is a state concentration region "
            f"these tokens reach in this run. That is geometry, not mechanism — "
            f"whether the model uses it takes an intervention.")
    return head + lead + (
        f"At α = {null.alpha:g} that is within what slowing steps and shared "
        f"drift produce: this basin is not evidence of anything more.")
