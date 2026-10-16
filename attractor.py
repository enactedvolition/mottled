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

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import metrics as metrics_mod
from density import Landscape
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
    pts = np.asarray(coords, dtype=np.float64)[..., :2]

    dens = density_at(landscape, pts.reshape(-1, 2)).reshape(L, T)
    members = np.argwhere(dens >= member_threshold * max(dens.max(), 1e-12))
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
        members=members.astype(np.int64),
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
    picture — the exact failure mode the tool is most at risk of.
    """
    pres = np.asarray(quality.preservation)
    if report.n_members:
        m = report.members
        basin_pres = float(pres[m[:, 0], m[:, 1]].mean())
    else:
        basin_pres = float(pres[:, report.token].mean())

    ev = quality.explained_variance
    var_phrase = f"keeps **{ev:.0%}** of the variance globally and " if ev is not None else ""
    lead = (f"**How much to trust this** — the projection {var_phrase}the states "
            f"that make up this basin retain **{basin_pres:.0%}** of their "
            f"hidden-space nearest neighbors (k={quality.k}) on average. ")
    if basin_pres < 0.5:
        tail = ("That is low: read the basin's *shape* as suggestive, not "
                "established — much of the local structure did not survive the "
                "flattening.")
    elif basin_pres < 0.8:
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
