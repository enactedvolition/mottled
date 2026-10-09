"""Cross-model comparison (crossmodel.py): the shared space and its limits.

Two models share no hidden space and often no tokenizer. These tests pin the
two things that make comparing them honest: readout space is a *real* shared
coordinate system (mass is conserved, the unshared part stays visible), and
CKA reports whether its own answer is identified instead of always returning
an argmax.
"""
import numpy as np
import pytest

import crossmodel as X
import tiny as synthetic
from trajectory import StateTrajectory

PROMPT = "the capital of france is"


def _traj(prompt=PROMPT, **kw):
    return synthetic.capture(prompt, **kw)


def _relabel(traj: StateTrajectory, vocab: list[str]) -> StateTrajectory:
    """Same trajectory, different vocabulary strings — a stand-in for a model
    with another tokenizer."""
    return StateTrajectory(
        hidden=traj.hidden, tokens=traj.tokens, logits=traj.logits,
        entropy=traj.entropy, topk=traj.topk, vocab=vocab,
        embedding_matrix=traj.embedding_matrix, meta=dict(traj.meta))


# ------------------------------------------------------------ shared vocab
def test_shared_vocab_is_the_intersection_of_strings():
    a = _traj()
    b = _relabel(a, [t if i % 2 else f"only_b_{i}" for i, t in enumerate(a.vocab)])
    shared = X.shared_vocab([a, b])
    assert shared == sorted(t for i, t in enumerate(a.vocab) if i % 2)
    assert all(not s.startswith("only_b") for s in shared)


def test_disjoint_vocabularies_refuse():
    a = _traj()
    b = _relabel(a, [f"other_{i}" for i in range(len(a.vocab))])
    with pytest.raises(ValueError, match="share no vocabulary"):
        X.shared_vocab([a, b])


# ---------------------------------------------------------- readout space
def test_readout_space_gives_models_one_coordinate_system():
    a, b = _traj(), _traj("the capital of germany is")
    (ra, rb), vocab = X.readout_space([a, b])
    assert ra.dim == rb.dim == len(vocab) + 1     # + the unshared bucket
    assert ra.vocab[-1] == X.UNSHARED_TOKEN
    ra.validate(); rb.validate()
    # a real distribution at every state
    np.testing.assert_allclose(ra.hidden.sum(axis=-1), 1.0, rtol=1e-4)
    assert (ra.hidden >= 0).all()


def test_unshared_mass_is_kept_visible_not_renormalised():
    """A model can look identical on shared tokens while spending its mass
    elsewhere; the picture has to be able to show that."""
    a = _traj()
    half = [t if i % 2 else f"only_a_{i}" for i, t in enumerate(a.vocab)]
    b = _relabel(a, half)
    (ra, _), vocab = X.readout_space([a, b])
    unshared = ra.hidden[..., -1]
    assert (unshared > 0).any(), "dropping half the vocab must leave visible mass"
    np.testing.assert_allclose(ra.hidden.sum(axis=-1), 1.0, rtol=1e-4)
    # and it is reported per layer in the metadata
    assert len(ra.meta["unshared_mass"]) == ra.n_layers
    assert ra.meta["shared_vocab"] == len(vocab)


def test_readout_space_needs_logits():
    a = synthetic.capture(PROMPT, keep_logits=False)
    with pytest.raises(ValueError, match="logit-lens"):
        X.readout_space([a, _traj()])


def test_readout_trajectory_flows_through_the_pipeline():
    from density import compute_density
    from projection import project_joint
    from terrain import mesh

    (ra, rb), _ = X.readout_space([_traj(), _traj("the capital of germany is")])
    coords, _ = project_joint([ra.hidden, rb.hidden])   # accepts a shared D
    assert coords[0].shape == (ra.n_layers, ra.n_tokens, 2)
    surface = mesh(compute_density(coords[0].reshape(-1, 2), grid_size=16))
    assert surface.z.shape == (16, 16)


# -------------------------------------------------------- readout compare
def test_compare_models_measures_divergence_per_layer():
    a, b = _traj(), _traj("the capital of germany is")
    cmp = X.compare_models(a, b)
    assert cmp.divergence.shape == (a.n_layers,)
    assert (cmp.divergence >= 0).all()
    assert cmp.shared_vocab > 0
    # identical models diverge nowhere
    same = X.compare_models(a, _traj())
    np.testing.assert_allclose(same.divergence, 0.0, atol=1e-6)
    assert same.agree_from == 0 and same.top_a == same.top_b


def test_compare_models_normalises_different_depths():
    """Layer 6 of a 12-layer model is not layer 6 of a 32-layer model, so the
    comparison runs on a resampled common depth."""
    deep = synthetic.capture(PROMPT, n_layers=13)
    shallow = synthetic.capture(PROMPT, n_layers=7)
    cmp = X.compare_models(deep, shallow)
    assert cmp.divergence.shape == (7,)
    assert cmp.unshared_a.shape == cmp.unshared_b.shape == (7,)


# ------------------------------------------------------------------- CKA
def test_cka_is_one_for_a_layer_against_itself():
    a = _traj()
    assert X.cka(a.hidden[5], a.hidden[5]) == pytest.approx(1.0, abs=1e-9)


def test_raw_cka_is_exactly_invariant_to_rotation_and_scale():
    """The invariances that let CKA compare models at all — exact in the raw
    form, which is what makes it a legitimate cross-model measure."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))       # orthogonal
    assert X.cka(x, 7.5 * (x @ q), standardize=False) == pytest.approx(1.0, abs=1e-9)
    # width-agnostic: the same structure embedded in more dimensions
    wide = np.concatenate([x, np.zeros((40, 16))], axis=1)
    assert X.cka(x, wide, standardize=False) == pytest.approx(1.0, abs=1e-9)


def test_standardizing_trades_rotation_invariance_for_resolution():
    """The default is a trade, and the cost is measured, not assumed:
    z-scoring keeps isotropic-scale invariance exactly but only approximates
    rotation invariance, because rotating mixes dimensions whose scales are
    then equalised differently."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert X.cka(x, 7.5 * x, standardize=True) == pytest.approx(1.0, abs=1e-9)
    rotated = X.cka(x, x @ q, standardize=True)
    assert 0.9 < rotated < 1.0, f"expected near-but-not-exact invariance, got {rotated}"


def test_cka_rejects_unpaired_or_degenerate_input():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="paired"):
        X.cka(rng.normal(size=(10, 4)), rng.normal(size=(9, 4)))
    with pytest.raises(ValueError, match="at least 2"):
        X.cka(rng.normal(size=(1, 4)), rng.normal(size=(1, 4)))


def test_layer_similarity_reports_whether_the_match_is_identified():
    """The value that keeps this honest: a flat row's argmax is noise, and
    the alignment says so rather than presenting it as a correspondence."""
    a = _traj("the capital of france is paris and the capital of germany is berlin")
    b = _traj("the capital of france is paris and the capital of germany is berlin")
    align = X.layer_similarity(a, b)
    assert align.matrix.shape == (a.n_layers, b.n_layers)
    assert align.contrast.shape == (a.n_layers,)
    assert align.n_states == a.n_tokens
    # Identical runs: every layer matches itself, in order. That much is the
    # code's job and is exact.
    np.testing.assert_array_equal(align.best, np.arange(a.n_layers))
    assert align.monotone
    np.testing.assert_allclose(np.diag(align.matrix), 1.0, atol=1e-5)

    # Identification is a different claim, and not every row clears the bar.
    # This used to assert `.all()`, which held only because the synthetic
    # backend took large steps between layers. In a small model consecutive
    # layers barely change the representation, so CKA genuinely cannot resolve
    # some of them even against an identical run -- and saying so is exactly
    # what `identified()` exists for. What is pinned is that the report is
    # consistent with the contrast it was computed from, and that most layers
    # do resolve.
    ident = align.identified()
    np.testing.assert_array_equal(ident, align.contrast >= 0.02)
    assert ident.sum() >= 0.7 * len(ident)
    assert "identified" in align.summary()
    if not ident.all():
        assert "unresolved" in align.summary()


def test_layer_similarity_refuses_states_that_are_not_counterparts():
    a = _traj()
    b = _traj("a much longer prompt with rather more tokens in it than that")
    with pytest.raises(ValueError, match="paired states"):
        X.layer_similarity(a, b)

    # same token count, different segmentation -> still not counterparts
    c = _traj("the capital of spain is")
    c.tokens = ["the", "capital", "of", "SPAIN", "is"]
    with pytest.raises(ValueError, match="split the text differently"):
        X.layer_similarity(a, c)


def test_standardizing_is_what_makes_the_alignment_resolvable():
    """Residual streams carry a few very-high-variance dimensions shared by
    every layer; without standardizing they dominate and every layer looks
    similar to every other. Pinned because it drives the default."""
    rng = np.random.default_rng(2)
    n, D, L = 60, 24, 6
    # a few outlier dimensions with enormous variance, identical in every
    # layer — the shape of a real residual stream's high-norm features
    outlier = rng.normal(size=(n, 3)) * 50.0
    layers = []
    for _ in range(L):
        own = rng.normal(size=(n, D - 3)) * 0.5        # each layer's own structure
        layers.append(np.concatenate([outlier, own], axis=1))
    stack = np.stack(layers)

    raw = np.array([[X.cka(stack[i], stack[j], standardize=False)
                     for j in range(L)] for i in range(L)])
    std = np.array([[X.cka(stack[i], stack[j], standardize=True)
                     for j in range(L)] for i in range(L)])
    off = ~np.eye(L, dtype=bool)
    # raw: everything looks alike; standardized: unrelated layers separate
    assert raw[off].mean() > 0.9
    assert std[off].mean() < raw[off].mean() - 0.3


# --------------------------------------------------------------- pipeline
def test_run_model_scene_puts_models_on_one_terrain():
    from config import MarbleConfig
    from pipeline import run_model_scene

    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)
    result = run_model_scene(cfg, PROMPT, ["tiny", "tiny"], loaded=synthetic.loaded())
    assert result["model_names"] == ["tiny", "tiny"]
    assert result["space"] == "readout"
    assert len(result["trajs"]) == 2
    # one shared terrain, one shared projection
    assert result["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)
    assert result["coords_list"][0].shape[-1] == cfg.n_components
    assert len(result["model_comparisons"]) == 1
    assert result["shared_vocab"] > 0
    # the untouched source captures come along for analyses that need them
    assert len(result["source_trajs"]) == 2


def test_run_model_scene_needs_models():
    from config import MarbleConfig
    from pipeline import run_model_scene

    with pytest.raises(ValueError, match="at least one model"):
        run_model_scene(MarbleConfig(use_cache=False), PROMPT, [])


# ------------------------------------------------------- real models (net)
@pytest.mark.network
def test_gpt2_and_distilgpt2_align_proportionally():
    """The finding this module is built around, on real models: GPT-2 and its
    distillation pass through comparable stages in order, and standardizing
    is what makes that visible."""
    from capture import capture

    # a few dozen states: CKA cannot resolve a correspondence from a handful
    text = ("The capital of France is Paris, and the capital of Germany is "
            "Berlin. Water freezes at zero degrees Celsius. The Roman Empire "
            "fell in the fifth century, and the printing press was invented "
            "much later in Mainz by Gutenberg.")
    big = capture("gpt2", text)
    small = capture("distilgpt2", text)
    assert big.n_layers > small.n_layers

    align = X.layer_similarity(big, small)
    assert align.matrix.shape == (big.n_layers, small.n_layers)
    assert align.monotone, f"expected an ordered path, got {align.best.tolist()}"
    assert align.identified().all(), f"contrast too flat: {align.contrast.tolist()}"
    # the path spans the shallower model rather than collapsing onto one layer
    assert align.best[0] == 0 and align.best[-1] >= small.n_layers - 2

    raw = X.layer_similarity(big, small, standardize=False)
    assert raw.identified().sum() < align.identified().sum(), (
        "raw CKA should resolve fewer layers than standardized")


@pytest.mark.network
def test_models_with_different_tokenizers_meet_in_readout_space():
    """Different family, different width, different depth, different
    vocabulary — and still comparable, because they read out into text."""
    from capture import capture

    prompt = "The capital of France is"
    gpt2 = capture("gpt2", prompt)
    pythia = capture("EleutherAI/pythia-70m", prompt)
    assert gpt2.dim != pythia.dim and gpt2.n_layers != pythia.n_layers

    cmp = X.compare_models(gpt2, pythia)
    assert 0 < cmp.shared_vocab < min(len(gpt2.vocab), len(pythia.vocab))
    assert cmp.divergence.shape == (min(gpt2.n_layers, pythia.n_layers),)
    # both are competent enough on this prompt to converge by the last layer
    assert cmp.final_divergence < cmp.divergence[0]
    # and the mass each spends outside the shared vocabulary is visible
    assert (cmp.unshared_a >= 0).all() and (cmp.unshared_b >= 0).all()


@pytest.mark.network
def test_cli_export_models(tmp_path, capsys):
    """`mottled export --models a,b` writes a cross-model scene."""
    import statefile
    from cli import main

    out = tmp_path / "models.mtj"
    assert main(["export", PROMPT, "-o", str(out),
                 "--models", "gpt2,distilgpt2"]) == 0
    # the comparison is reported on stderr: stdout is kept for data
    assert "shared vocabulary" in capsys.readouterr().err
    scene = statefile.load_scene(out)
    assert len(scene["runs"]) == 2


# ------------------------------------------------------------ decode axis
def _gen(prompt=PROMPT, n=4, **kw):
    return synthetic.generate_and_capture(prompt, max_new_tokens=n, **kw)


def test_identical_generations_never_diverge():
    a, b = _gen(), _gen()
    g = X.compare_generations(a, b)
    assert g.tokens_a == g.tokens_b
    assert g.agree.all()
    assert g.first_divergence is None
    assert g.comparable_steps == len(g.agree)
    assert "identical" in g.summary()


def test_generation_comparison_marks_where_comparability_ends():
    """The honest boundary: after the models choose differently they are
    continuing different texts, so later steps are not a comparison."""
    a = _gen()
    b = _gen()
    # force a disagreement at step 1
    b.meta["generation"]["steps"][1]["token"] = "divergent"
    g = X.compare_generations(a, b)
    assert g.first_divergence == 1
    # step 1 is still like-for-like (same context, different choice); step 2 on
    # is each model continuing its own text
    assert g.comparable_steps == 2
    assert "not a comparison" in g.summary()
    assert g.p_a.shape == g.p_b.shape == g.agree.shape
    assert (g.entropy_a >= 0).all()


def test_generation_comparison_needs_a_decode_record():
    with pytest.raises(ValueError, match="no decode record"):
        X.compare_generations(_traj(), _gen())


def test_forced_divergence_scores_both_models_on_one_text():
    a, b = _traj(), _traj()
    f = X.forced_divergence(a, b)
    assert f.divergence.shape == (a.n_tokens - 1,)     # each position predicts
    assert f.top_agree.shape == f.divergence.shape     # the next one
    assert f.tokens == list(a.tokens)
    # identical models agree everywhere
    np.testing.assert_allclose(f.divergence, 0.0, atol=1e-6)
    assert f.top_agree.all()
    assert 0 <= f.worst_position < len(f.divergence)


def test_forced_divergence_refuses_mismatched_text():
    a = _traj()
    b = _traj("a longer prompt with more tokens than the other one has")
    with pytest.raises(ValueError, match="tokenized the same text"):
        X.forced_divergence(a, b)


@pytest.mark.network
def test_gpt2_and_distilgpt2_part_company_mid_sentence():
    """The capstone question, on real models: both complete the prompt the
    same way for a step, then continue differently — and teacher forcing
    stays a like-for-like comparison where free generation cannot."""
    from capture import capture, generate_and_capture

    prompt = "The residual stream moves, turns, and settles"
    a = generate_and_capture("gpt2", prompt, max_new_tokens=8)
    b = generate_and_capture("distilgpt2", prompt, max_new_tokens=8)

    g = X.compare_generations(a, b)
    assert g.tokens_a[0] == g.tokens_b[0], "both should complete the thesis alike"
    assert g.first_divergence is not None, "different models should part company"
    assert g.comparable_steps == g.first_divergence + 1
    assert "diverge at step" in g.summary()

    # scored on one fixed text instead, every position stays comparable
    text = prompt + "".join(g.tokens_a)
    f = X.forced_divergence(capture("gpt2", text), capture("distilgpt2", text))
    assert f.divergence.shape[0] == len(f.tokens) - 1
    assert (f.divergence >= 0).all()
    assert not f.top_agree.all(), "two different models should disagree somewhere"


# ------------------------------------------------------- the explorer panel
class _FakeSt:
    """Records what the panel would render, so its wording is testable
    without a browser."""

    def __init__(self):
        self.text: list[str] = []
        self.charts = 0
        self.tables = 0

    def caption(self, s, *a, **kw):
        self.text.append(str(s))

    write = markdown = caption

    def line_chart(self, *a, **kw):
        self.charts += 1

    bar_chart = line_chart

    def dataframe(self, *a, **kw):
        self.tables += 1

    def all_text(self) -> str:
        return "\n".join(self.text)


def test_model_panel_states_the_shared_space_and_its_limits():
    from config import MarbleConfig
    from pipeline import run_model_scene
    from ui import render_model_comparison

    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)
    result = run_model_scene(cfg, PROMPT, ["tiny", "tiny"], loaded=synthetic.loaded())
    st = _FakeSt()
    render_model_comparison(st, result)
    body = st.all_text()

    # names the shared coordinate system and why it is the shared one
    assert "readout space" in body
    assert X.UNSHARED_TOKEN in body
    # reports the divergence and the depth-normalisation caveat
    assert "divergence" in body and "Depth is normalized" in body
    # reports the alignment together with whether it is identified
    assert "identified" in body
    assert st.charts >= 2


def test_model_panel_explains_an_unavailable_alignment():
    """When the models tokenize differently the alignment is undefined, and
    the panel says why instead of omitting it silently."""
    from ui import render_model_comparison

    a = _traj()
    b = _traj("a much longer prompt with rather more tokens in it than that")
    result = {"model_names": ["a", "b"], "shared_vocab": 10,
              "source_trajs": [a, b], "trajs": [a, b], "model_comparisons": []}
    st = _FakeSt()
    render_model_comparison(st, result)
    assert "unavailable" in st.all_text()
    assert "paired states" in st.all_text()


def test_model_panel_shows_the_generation_split():
    from ui import render_model_comparison

    a = _gen()
    b = _gen()
    b.meta["generation"]["steps"][1]["token"] = "divergent"
    result = {"model_names": ["a", "b"], "shared_vocab": 10,
              "source_trajs": [a, b], "trajs": [a, b], "model_comparisons": []}
    st = _FakeSt()
    render_model_comparison(st, result)
    assert "diverge at step 1" in st.all_text()
    assert st.tables == 1


# --------------------------------------------- within-model assumptions
def test_scrubber_spans_the_deepest_run_not_the_shallowest():
    """Runs of unequal depth are normal now (two models on one terrain).
    Taking the minimum silently made the deeper model's late layers
    unreachable, with no sign anything had been cut."""
    from config import MarbleConfig
    from pipeline import _assemble_scene
    from render import render

    deep = synthetic.capture(PROMPT, n_layers=13)
    shallow = synthetic.capture(PROMPT, n_layers=7)
    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)

    for order in ([deep, shallow], [shallow, deep]):   # either run may be first
        trajs, _ = X.readout_space(order)
        r = _assemble_scene(cfg, trajs)
        fig = render(r["traj"], r["mesh"], r["trajectories"], r["fine_paths"],
                     traj_b=r["traj_b"], trajectories_b=r["trajectories_b"],
                     fine_paths_b=r["fine_paths_b"],
                     frames_per_layer=cfg.frames_per_layer)
        steps = fig.layout.sliders[0].steps
        assert len(steps) == 13, "the deepest run must stay reachable"
        assert steps[-1].label == "12"


def test_scene_records_which_model_each_run_came_from():
    """Scene-level meta describes run 0 — wrong for a scene whose runs are
    different models."""
    import io

    import statefile
    from config import MarbleConfig
    from pipeline import run_model_scene

    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)
    result = run_model_scene(cfg, PROMPT, ["tiny", "tiny"], loaded=synthetic.loaded())
    buf = io.BytesIO()
    statefile.save_scene(result, buf)
    buf.seek(0)
    runs = statefile.load_scene(buf)["runs"]
    assert [r["model"] for r in runs] == ["tiny", "tiny"]


def test_readout_space_is_never_offered_a_residual_sae():
    """The trained dictionary is keyed on hidden width, which only identifies
    a residual stream when the states *are* one. Readout-space axes are
    vocabulary entries; a width collision must not offer a residual SAE for
    probability vectors."""
    (ra, _), _ = X.readout_space([_traj(), _traj("the capital of germany is")])
    assert ra.meta["space"] == "readout"
    # the guard keys off that marker, not the width
    assert ra.meta.get("space") == "readout" and ra.dim != _traj().dim


def test_readout_space_does_not_materialise_an_identity_embedding():
    """Regression: readout space once carried `np.eye(V_shared)` as its
    embedding matrix — conceptually tidy (each axis *is* a vocabulary entry)
    and O(V^2) in memory. For two models sharing 42k tokens that is a 7 GB
    allocation, which killed the process on a real 1.5B-parameter pair. It
    buys nothing either: the nearest vocabulary token to a readout state is
    just its largest component, which `topk` already reports."""
    (ra, rb), vocab = X.readout_space([_traj(), _traj("the capital of germany is")])
    assert ra.embedding_matrix is None and rb.embedding_matrix is None
    ra.validate()          # optional everywhere it matters
    # the information the identity would have carried is already present
    assert ra.topk is not None and ra.vocab[-1] == X.UNSHARED_TOKEN


def test_attach_inspector_skips_neighbors_without_an_embedding_matrix():
    """The inspector layer degrades rather than failing when a producer has
    no embedding table — which readout space now deliberately does not."""
    from config import MarbleConfig
    from pipeline import attach_inspector, run_model_scene

    cfg = MarbleConfig(use_cache=False, density_bootstrap=0)
    result = attach_inspector(
        run_model_scene(cfg, PROMPT, ["tiny", "tiny"], loaded=synthetic.loaded()))
    for entry in result["inspector_list"]:
        assert "neighbor_idx" not in entry      # no embeddings -> no neighbors
