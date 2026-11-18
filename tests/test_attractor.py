"""Attractor analysis: why the basin forms and what it is made of."""

import numpy as np

import attractor as A
import density as D
import projection as P
import tiny as synthetic

PROMPT = "the capital of france is"


def _pipeline(prompt=PROMPT, **kw):
    traj = synthetic.capture(prompt, **kw)
    coords, _ = P.project(traj.hidden)
    landscape = D.compute_density(coords)
    return traj, coords, landscape


def test_analyze_report_is_sane():
    traj, coords, landscape = _pipeline(capture_components=True)
    r = A.analyze(traj, coords, landscape)
    L, T = traj.n_layers, traj.n_tokens
    assert r.token == T - 1
    assert r.n_states == L * T
    assert r.n_members >= 1 and 0 < r.member_share <= 1
    assert 0 <= r.layer_range[0] <= r.layer_range[1] <= L - 1
    assert 0.0 <= r.late_share <= 1.0
    assert r.step.shape == (L - 1,) and (r.step >= 0).all()
    assert 1 <= r.peak_step_layer <= L - 1
    if r.settle_layer is not None:
        assert 0 <= r.settle_layer <= L - 2
    assert r.entropy is not None and r.entropy.shape == (L,)
    assert r.top_token is not None
    assert 0 <= r.readout_stable_from <= L - 1
    assert r.late_attn_share is not None and 0.0 <= r.late_attn_share <= 1.0
    assert landscape.grid_x[0] <= r.center[0] <= landscape.grid_x[-1]
    assert landscape.grid_y[0] <= r.center[1] <= landscape.grid_y[-1]


def test_members_meet_the_density_cut():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape, member_threshold=0.6)
    dens = A.density_at(landscape, coords.reshape(-1, 2)).reshape(
        traj.n_layers, traj.n_tokens)
    cut = 0.6 * dens.max()
    member_set = {tuple(m) for m in r.members}
    for l in range(traj.n_layers):
        for t in range(traj.n_tokens):
            assert ((l, t) in member_set) == (dens[l, t] >= cut)


def test_settle_layer_definition():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape, settle_frac=0.5)
    small = 0.5 * r.step.max()
    if r.settle_layer is None:
        assert r.step[-1] > small  # the tail never stays below the cut
    else:
        assert r.step[r.settle_layer:].max() <= small
        if r.settle_layer > 0:
            assert r.step[r.settle_layer - 1:].max() > small


def test_readout_stability_is_the_minimal_suffix():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape)
    top1 = [traj.state(l, r.token).topk[0][0] for l in range(traj.n_layers)]
    assert r.top_token == top1[-1]
    assert all(t == r.top_token for t in top1[r.readout_stable_from:])
    if r.readout_stable_from > 0:
        assert top1[r.readout_stable_from - 1] != r.top_token


def test_explain_contains_the_measurements():
    traj, coords, landscape = _pipeline(capture_components=True)
    r = A.analyze(traj, coords, landscape)
    text = A.explain(r, traj)
    assert f"{r.n_members} of {r.n_states}" in text
    assert f"layers {r.layer_range[0]}–{r.layer_range[1]}" in text
    assert "density" in text
    assert r.top_token in text
    assert "MLP" in text  # components were captured
    # deterministic prose from deterministic measurements
    assert text == A.explain(A.analyze(traj, coords, landscape), traj)


def test_explain_fidelity_clause_reflects_preservation():
    """explain() threads projection distortion into the prose: a low-fidelity
    basin is called suggestive, a high-fidelity one faithful; absent quality,
    the trust clause is omitted entirely."""
    from projection import ProjectionQuality

    traj = synthetic.capture(PROMPT)
    coords, projector = P.project(traj.hidden)
    landscape = D.compute_density(coords)
    r = A.analyze(traj, coords, landscape)

    q = P.projection_quality(traj.hidden, coords, projector)
    text = A.explain(r, traj, quality=q)
    assert "How much to trust this" in text
    assert "How much to trust this" not in A.explain(r, traj)  # opt-in

    shape = (traj.n_layers, traj.n_tokens)
    low = ProjectionQuality(preservation=np.full(shape, 0.2, np.float32),
                            residual=None, explained_variance=0.5, k=10)
    assert "suggestive, not" in A.explain(r, traj, quality=low)

    high = ProjectionQuality(preservation=np.full(shape, 0.95, np.float32),
                             residual=None, explained_variance=0.99, k=10)
    high_text = A.explain(r, traj, quality=high)
    assert "faithful view" in high_text and "suggestive" not in high_text


def test_render_flags_low_fidelity_states_on_the_scene():
    from config import MarbleConfig
    from projection import ProjectionQuality
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_pipeline(cfg, PROMPT, **synthetic.mt())
    traj = result["traj"]
    shape = (traj.n_layers, traj.n_tokens)

    bad = ProjectionQuality(preservation=np.zeros(shape, np.float32),
                            residual=None, explained_variance=0.5, k=10)
    flagged = render(traj, result["mesh"], result["trajectories"],
                     result["fine_paths"], quality=bad)
    assert any("low fidelity" in (tr.name or "") for tr in flagged.data)

    good = ProjectionQuality(preservation=np.ones(shape, np.float32),
                             residual=None, explained_variance=0.99, k=10)
    clean = render(traj, result["mesh"], result["trajectories"],
                   result["fine_paths"], quality=good)
    assert not any("low fidelity" in (tr.name or "") for tr in clean.data)


def test_the_flag_and_the_prose_read_within_layer_preservation():
    """Pooled preservation can stay perfect while every layer is scrambled:
    cross-layer neighbors survive a within-layer shuffle. The ✕ markers and
    the trust sentence read the within-layer score where there is one (the
    pooled one drove both before, and could not tell PCA from random on
    GPT-2), and say so."""
    from config import MarbleConfig
    from projection import ProjectionQuality
    from ui import render, run_pipeline

    result = run_pipeline(MarbleConfig(model="tiny", use_cache=False), PROMPT,
                          **synthetic.mt())
    traj = result["traj"]
    shape = (traj.n_layers, traj.n_tokens)
    scrambled = ProjectionQuality(
        preservation=np.ones(shape, np.float32), residual=None,
        explained_variance=0.9, k=10, per_layer=np.full(shape, 0.2, np.float32),
        per_layer_cosine=np.full(shape, 0.4, np.float32), k_layer=5, chance_layer=0.3)
    fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                 quality=scrambled)
    marks = [tr for tr in fig.data if "low fidelity" in (tr.name or "")]
    assert marks and "within-layer" in marks[0].name
    assert sum(len(tr.x) for tr in marks) == sum(len(t.points) for t in result["trajectories"])

    r = A.analyze(traj, result["coords"], result["landscape"])
    text = A.explain(r, traj, quality=scrambled)
    assert "suggestive, not" in text and "within-layer" in text
    assert "no better than a random placement" in text
    assert "**20%**" in text and "keep 30%" in text   # the worse metric; chance


def test_density_at_matches_grid_nodes():
    _, coords, landscape = _pipeline()
    gx, gy = landscape.grid_x, landscape.grid_y
    nodes = np.array([[gx[3], gy[5]], [gx[10], gy[0]]])
    got = A.density_at(landscape, nodes)
    assert np.allclose(got, [landscape.density[5, 3], landscape.density[0, 10]],
                       atol=1e-6)
    # outside the grid the field is empty, not extrapolated
    far = np.array([[gx[-1] + 10 * (gx[-1] - gx[0]), gy[0]]])
    assert A.density_at(landscape, far)[0] == 0.0


def test_single_layer_trajectory_has_no_late_attn_share_nan():
    """A one-layer run has no block writes to share between attn/MLP —
    that must come back as None, never nan (empty-slice .mean() silently
    gives nan and would leak into explain()'s prose as "nan% attention")."""
    traj, coords, landscape = _pipeline(n_layers=1, capture_components=True)
    r = A.analyze(traj, coords, landscape)
    assert r.step.shape == (0,)
    assert r.settle_layer is None
    assert r.late_attn_share is None
    assert "nan" not in A.explain(r, traj).lower()


def test_zero_members_above_threshold_is_reported_as_absent_not_zero_zero():
    """An unreachable threshold must not fall back to the (0, 0) sentinel
    layer range, which would read as "the basin spans exactly layer 0"
    instead of "there is no basin here"."""
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape, member_threshold=10.0)
    assert r.n_members == 0
    assert r.layer_range is None
    text = A.explain(r, traj)
    assert "no basin" in text or "no state" in text
    assert "layers 0" not in text


def test_explain_does_not_overclaim_deceleration_when_the_token_never_settles():
    """If a token's step never drops below the settle threshold, explain()
    must not still assert it "decelerates" / "has stopped travelling" — the
    demo prompt's default token (settle_frac=0.25) never settles, so this
    exercises the actual default path, not a contrived one."""
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape)
    assert r.settle_layer is None  # the case this test targets
    text = A.explain(r, traj)
    assert "decelerates" not in text
    assert "stopped travelling" not in text
    assert "isn't what settles here" in text


def test_render_pins_the_basin_annotation():
    from config import MarbleConfig
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_pipeline(cfg, PROMPT, **synthetic.mt())
    r = A.analyze(result["traj"], result["coords"], result["landscape"])
    fig = render(result["traj"], result["mesh"], result["trajectories"],
                 result["fine_paths"], basin=r)
    assert len(fig.layout.scene.annotations) == 1
    note = fig.layout.scene.annotations[0].text
    assert "attractor basin" in note and f"{r.n_members}/{r.n_states}" in note
    assert any(tr.name == "attractor" for tr in fig.data)
    # without a report the scene stays unannotated
    plain = render(result["traj"], result["mesh"], result["trajectories"],
                   result["fine_paths"])
    assert len(plain.layout.scene.annotations) == 0


# ------------------------------------------------------------ the basin null
def _walks(seed, converge=(), shared=0.0, L=13, T=12, D=64):
    """Synthetic residual streams with known answers. Every token takes steps
    of the same shrinking size; `converge` tokens spend theirs heading for
    one point, the rest go their own way, or with `shared` mostly the
    layer's common way (drift every token shares, which the null must not
    count as a basin)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((L, T, D))
    h[0] = rng.normal(size=(T, D)) * 3.0 / np.sqrt(D)
    target = rng.normal(size=D) * 3.0 / np.sqrt(D)
    for layer in range(1, L):
        size = 6.0 * 0.8 ** layer
        common = rng.normal(size=D)
        common /= np.linalg.norm(common)
        for t in range(T):
            own = rng.normal(size=D)
            own /= np.linalg.norm(own)
            if t in converge:
                gap = target - h[layer - 1, t]
                far = np.linalg.norm(gap)
                d = gap / max(far, 1e-9) * min(1.0, far / size) + 0.05 * own
            else:
                d = shared * common + (1 - shared) * own
                d /= np.linalg.norm(d)
            h[layer, t] = h[layer - 1, t] + size * d
    return h.astype(np.float32)


def _basin_of(space):
    coords, _ = P.project(space)
    return A.members_of(coords, D.compute_density(coords))


def test_concentration_names_what_it_cannot_test():
    """Every density field has a peak, so `members_of` always finds a basin;
    the statistic has to be able to say there is nothing to test."""
    space = np.random.default_rng(0).normal(size=(4, 3, 8))
    none = A.concentration(space, np.empty((0, 2), dtype=np.int64))
    one = A.concentration(space, np.array([[1, 2], [2, 2], [3, 2]]))
    band = A.concentration(space, np.array([[l, t] for l in (2, 3) for t in range(3)]))
    assert (none.status, one.status, band.status) == ("no-members", "one-token", "depth-band")
    assert one.n_member_tokens == 1 and one.ratio is None
    assert A.concentration(space, np.array([[3, 0], [3, 1]])).status == "tested"


def test_concentration_is_low_only_where_tokens_meet():
    meet = _walks(0, converge=range(3, 8))
    apart = _walks(0)
    members = np.array([[l, t] for l in range(8, 13) for t in range(3, 8)])
    rng = np.random.default_rng(0)
    assert A.concentration(meet, members, draws=50, rng=rng).ratio < 0.5
    assert 0.8 < A.concentration(apart, members, draws=50, rng=rng).ratio < 1.2


def test_step_direction_surrogate_keeps_starts_and_step_sizes_and_moves_directions():
    h = _walks(1, converge=range(3, 8))
    s = A.step_direction_surrogate(h, np.random.default_rng(0))
    assert s.shape == h.shape and s.dtype == h.dtype
    np.testing.assert_array_equal(s[0], h[0])
    np.testing.assert_allclose(np.linalg.norm(np.diff(s, axis=0), axis=-1),
                               np.linalg.norm(np.diff(h, axis=0), axis=-1), rtol=1e-4)
    # each step's direction is one some token took at that layer
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    cos = np.einsum("ltd,lud->ltu", unit(np.diff(s, axis=0)), unit(np.diff(h, axis=0)))
    np.testing.assert_allclose(cos.max(axis=-1), 1.0, atol=1e-4)
    assert not np.allclose(s, h)


def test_basin_null_tells_convergence_from_slowing_and_shared_drift():
    """Five of twelve tokens converging survives; independent walks and a
    drift every token shares do not, though all three slow down alike. The
    calibration this rests on (2 of 40 null runs survive at alpha 0.05) is in
    `basin_null`'s docstring; this pins the three cases it was measured on."""
    draws = 60
    met = A.basin_null(_walks(0, converge=range(3, 8)), _basin_of, draws=draws, seed=0)
    walk = A.basin_null(_walks(0), _basin_of, draws=draws, seed=0)
    drift = A.basin_null(_walks(0, shared=0.8), _basin_of, draws=draws, seed=0)
    assert met.verdict == "survives" and met.p_surrogate <= 1 / draws + 1e-9
    assert walk.verdict == "within-null" and walk.p_surrogate > 0.2
    assert drift.verdict == "within-null" and drift.p_surrogate > 0.2
    for r in (met, walk, drift):
        assert r.surrogate_ratios.shape == (draws,) and r.draws == draws
        assert (r.alpha, r.seed) == (A.NULL_ALPHA, 0)


def test_basin_null_does_not_test_what_it_cannot():
    one = A.basin_null(np.ones((4, 3, 8), np.float32),
                       lambda h: np.array([[2, 1], [3, 1]]), draws=5)
    assert one.verdict == "one-token" and one.surrogate_ratios is None
    assert "one token position" in A.explain_null(one)


def test_a_token_on_another_scale_is_refused_not_tested():
    """GPT-2's position 0 sits at ~35x the other tokens' norm; with it in the
    picture every raw-space basin "survived". The null declines instead, and
    unit-RMS states (the `rms` gauge) take the outlier away."""
    h = _walks(0, converge=range(3, 8))
    h[:, 0] *= 40.0
    r = A.basin_null(h, _basin_of, draws=5)
    assert r.verdict == "scale-outlier" and r.surrogate_ratios is None
    assert r.observed.outlier_token == 0 and r.observed.outlier_scale > 30
    text = A.explain_null(r)
    assert "not tested" in text and f"{r.observed.outlier_scale:.0f}×" in text
    assert "`rms`" in text
    unit = h / np.sqrt((h.astype(np.float64) ** 2).mean(axis=-1, keepdims=True))
    assert A.basin_null(unit.astype(np.float32), _basin_of, draws=5).verdict != "scale-outlier"


def test_explain_null_claims_geometry_not_mechanism():
    met = A.basin_null(_walks(0, converge=range(3, 8)), _basin_of, draws=30, seed=0)
    walk = A.basin_null(_walks(0), _basin_of, draws=30, seed=0)
    yes, no = A.explain_null(met), A.explain_null(walk)
    assert "state concentration region" in yes and "not mechanism" in yes
    assert "intervention" in yes and "attractor" not in yes.lower()
    assert "not evidence of anything more" in no
    assert f"{met.observed.ratio:.2f}" in yes and "30 surrogate runs" in yes


def test_pipeline_basin_null_projects_every_surrogate_afresh(monkeypatch):
    """The null goes through the run's own gauge, projection and density, so a
    surrogate is never drawn on the real run's axes."""
    import pipeline
    from config import MarbleConfig

    calls, real = [], P.project
    monkeypatch.setattr(P, "project", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = pipeline.basin_null(MarbleConfig(model="tiny", use_cache=False),
                            synthetic.capture(PROMPT), draws=7)
    assert r.draws == 7
    assert len(calls) == (1 if r.surrogate_ratios is None else 8)
