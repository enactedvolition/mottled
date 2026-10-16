"""Phase 3 — SAE features and residual decomposition."""

import numpy as np
import pytest

import metrics as M
import sae as S
import tiny as synthetic

PROMPT = "the capital of france is"


# ------------------------------------------------------------------ SAE math
def _identity_sae(dim: int) -> S.SAE:
    eye = np.eye(dim, dtype=np.float32)
    return S.SAE(w_enc=eye.copy(), b_enc=np.zeros(dim, np.float32),
                 w_dec=eye.copy(), b_dec=np.zeros(dim, np.float32))


def test_encode_is_relu_of_affine():
    sae = S.SAE(
        w_enc=np.array([[1.0, -1.0]], dtype=np.float32),   # D=1, F=2
        b_enc=np.array([0.5, 0.5], dtype=np.float32),
        w_dec=np.zeros((2, 1), dtype=np.float32),
        b_dec=np.array([2.0], dtype=np.float32),
    )
    sae.validate()
    # h=3: pre-bias 3-2=1 -> [1*1+0.5, 1*-1+0.5] = [1.5, -0.5] -> ReLU
    acts = sae.encode(np.array([3.0]))
    assert np.allclose(acts, [1.5, 0.0])


def test_identity_sae_reconstructs_nonnegative_inputs():
    sae = _identity_sae(4)
    h = np.abs(np.random.default_rng(0).normal(size=(5, 4))).astype(np.float32)
    assert np.allclose(sae.reconstruct(h), h, atol=1e-6)
    assert np.allclose(sae.reconstruction_error(h), 0.0, atol=1e-5)


def test_validate_rejects_bad_shapes():
    sae = _identity_sae(4)
    sae.b_enc = np.zeros(3, np.float32)
    with pytest.raises(ValueError):
        sae.validate()


def test_demo_sae_deterministic_sparse_unit_dictionary():
    a = S.demo_sae(16, n_features=64, seed=3)
    b = S.demo_sae(16, n_features=64, seed=3)
    assert np.array_equal(a.w_enc, b.w_enc)
    assert (a.dim, a.n_features) == (16, 64)
    assert np.allclose(np.linalg.norm(a.w_dec, axis=1), 1.0, atol=1e-6)
    acts = a.encode(np.random.default_rng(1).normal(size=(32, 16)))
    assert (acts >= 0).all()
    assert 0 < (acts > 0).mean() < 0.6  # the negative bias keeps it sparse


def test_npz_roundtrip(tmp_path):
    sae = S.demo_sae(8, n_features=12, seed=1)
    sae.labels = [f"feat-{i}" for i in range(12)]
    path = tmp_path / "sae.npz"
    S.save_npz(sae, path)
    back = S.load_npz(path)
    assert np.allclose(back.w_enc, sae.w_enc)
    assert np.allclose(back.b_dec, sae.b_dec)
    assert back.labels == sae.labels
    assert back.feature_label(3) == "feat-3"
    assert S.demo_sae(8, 12, 1).feature_label(3) == "f3"  # unlabeled fallback


# -------------------------------------------------- real-weights loading (1a)
def test_from_state_dict_maps_and_orients():
    """The standard four-array convention maps directly; either weight
    orientation is accepted and normalized to Mottled's (D, F) / (F, D)."""
    rng = np.random.default_rng(0)
    D, F = 6, 10
    w_enc = rng.normal(size=(D, F)).astype(np.float32)
    w_dec = rng.normal(size=(F, D)).astype(np.float32)
    b_enc = rng.normal(size=F).astype(np.float32)
    b_dec = rng.normal(size=D).astype(np.float32)

    sae = S.from_state_dict({"W_enc": w_enc, "b_enc": b_enc,
                             "W_dec": w_dec, "b_dec": b_dec})
    assert (sae.dim, sae.n_features) == (D, F)
    h = rng.normal(size=(4, D)).astype(np.float32)
    expected = np.maximum((h - b_dec) @ w_enc + b_enc, 0.0)
    assert np.allclose(sae.encode(h), expected, atol=1e-5)

    # a transposed encoder (F, D) is auto-oriented back to (D, F)
    flipped = S.from_state_dict({"W_enc": w_enc.T, "b_enc": b_enc,
                                 "W_dec": w_dec, "b_dec": b_dec})
    assert np.allclose(flipped.w_enc, w_enc, atol=1e-6)

    with pytest.raises(KeyError):
        S.from_state_dict({"b_enc": b_enc, "b_dec": b_dec})  # missing weights


def test_from_sae_lens_stub_and_rejects_nonstandard():
    rng = np.random.default_rng(1)
    D, F = 5, 8

    class _Cfg:
        architecture = "standard"

    class _StubSAE:
        cfg = _Cfg()
        W_enc = rng.normal(size=(D, F)).astype(np.float32)
        b_enc = rng.normal(size=F).astype(np.float32)
        W_dec = rng.normal(size=(F, D)).astype(np.float32)
        b_dec = rng.normal(size=D).astype(np.float32)

    stub = _StubSAE()
    sae = S.from_sae_lens(stub)
    assert np.allclose(sae.w_enc, stub.W_enc) and np.allclose(sae.b_dec, stub.b_dec)

    stub.cfg.architecture = "jumprelu"   # different nonlinearity → refuse
    with pytest.raises(ValueError, match="architecture"):
        S.from_sae_lens(stub)


def _saelens_stub(D=5, F=8, seed=1, **cfg_attrs):
    rng = np.random.default_rng(seed)
    cfg = type("Cfg", (), {"architecture": "standard", **cfg_attrs})()
    return type("Stub", (), {
        "cfg": cfg,
        "W_enc": rng.normal(size=(D, F)).astype(np.float32),
        "b_enc": rng.normal(size=F).astype(np.float32),
        "W_dec": rng.normal(size=(F, D)).astype(np.float32),
        "b_dec": rng.normal(size=D).astype(np.float32),
    })()


def test_from_sae_lens_folds_b_dec_when_not_applied_to_input():
    """A standard SAE trained with apply_b_dec_to_input=False encodes
    `h @ W_enc + b_enc` (no b_dec subtraction). Mottled's forward always
    subtracts b_dec, so the converter must fold that constant into b_enc and
    reproduce the source encode exactly — not silently mis-encode it."""
    stub = _saelens_stub(apply_b_dec_to_input=False)
    sae = S.from_sae_lens(stub)
    h = np.random.default_rng(7).normal(size=(4, 5)).astype(np.float32)
    source = np.maximum(h @ stub.W_enc + stub.b_enc, 0.0)  # no b_dec on input
    assert np.allclose(sae.encode(h), source, atol=1e-5)
    assert np.allclose(sae.b_dec, stub.b_dec)              # decode still uses b_dec
    # the default (apply_b_dec_to_input=True) path is an untouched copy
    plain = S.from_sae_lens(_saelens_stub(apply_b_dec_to_input=True))
    ref = _saelens_stub(apply_b_dec_to_input=True)
    assert np.allclose(plain.b_enc, ref.b_enc)


def test_from_sae_lens_rejects_unrepresentable_configs():
    """The gate refuses forwards Mottled cannot reproduce, and fails closed on a
    gated SAE even when the config does not name its architecture."""
    with pytest.raises(ValueError, match="normalize"):
        S.from_sae_lens(_saelens_stub(normalize_activations="layer_norm"))
    with pytest.raises(ValueError, match="activation_fn"):
        S.from_sae_lens(_saelens_stub(activation_fn="topk"))
    gated = _saelens_stub()                 # cfg says "standard" …
    gated.W_gate = gated.W_enc.copy()       # … but the gate weight gives it away
    with pytest.raises(ValueError, match="gated"):
        S.from_sae_lens(gated)


def test_convert_sae_cli_accepts_out_after_subcommand():
    """Regression: -o/--out must parse *after* the subcommand name. argparse's
    subparser action consumes everything following the subcommand, so a required
    option on the parent parser (before add_subparsers) is unreachable."""
    import convert_sae

    p = convert_sae._build_parser()
    a = p.parse_args(["from-file", "sae.safetensors", "-o", "out.npz"])
    assert a.out == "out.npz" and a.path == "sae.safetensors"
    a = p.parse_args(["from-saelens", "rel", "sid", "-o", "d.npz"])
    assert a.out == "d.npz" and a.release == "rel" and a.sae_id == "sid"


def _fitted_sae(hidden_flat: np.ndarray) -> S.SAE:
    """A data-aligned dictionary built (in-test) from the states' own SVD.

    Antipodal singular directions + a ReLU encoder reconstruct signed hidden
    states in their own subspace — this is *not* training shipped in the
    library (a non-goal); it stands in for a real trained SAE so we can show
    the reconstruction metric separates a meaningful dictionary from
    `demo_sae`'s random one."""
    H = np.asarray(hidden_flat, dtype=np.float64)
    mean = H.mean(axis=0)
    _, _, vt = np.linalg.svd(H - mean, full_matrices=False)
    dirs = np.vstack([vt, -vt]).astype(np.float32)      # (2k, D) antipodal
    return S.SAE(w_enc=dirs.T.copy(), b_enc=np.zeros(len(dirs), np.float32),
                 w_dec=dirs, b_dec=mean.astype(np.float32))


def test_fitted_sae_beats_demo_reconstruction():
    """The acceptance check: a real (data-aligned) dictionary reconstructs
    captured states far better than the untrained demo dictionary."""
    traj = synthetic.capture(PROMPT)
    H = traj.flat_hidden()

    real = _fitted_sae(H)
    demo = S.demo_sae(traj.dim, n_features=256)

    real_err = float(real.reconstruction_error(H).mean())
    demo_err = float(demo.reconstruction_error(H).mean())
    assert real_err < 0.05                     # near-exact on the real states
    assert real_err < 0.3 * demo_err           # and materially below the demo


# ------------------------------------------------------------- features
def test_feature_trajectory_shapes():
    traj = synthetic.capture(PROMPT)
    sae = S.demo_sae(traj.dim, n_features=32)
    acts = S.feature_trajectory(traj, sae)
    assert acts.shape == (traj.n_layers, traj.n_tokens, 32)
    assert np.isfinite(acts).all() and (acts >= 0).all()
    with pytest.raises(ValueError):
        S.feature_trajectory(traj, S.demo_sae(traj.dim + 1, 32))


def test_top_and_active_features():
    acts = np.zeros((2, 2, 5))
    acts[1, 0] = [0.0, 3.0, 1.0, 0.0, 2.0]
    top = S.top_features(acts, layer=1, token=0, k=3)
    assert top == [(1, 3.0), (4, 2.0), (2, 1.0)]
    assert S.top_features(acts, layer=0, token=0, k=3) == []  # nothing fires
    ranked = S.active_features(acts, k=10)
    assert list(ranked[:2]) == [1, 4]
    assert 0 not in ranked and 3 not in ranked  # silent features excluded


# ---------------------------------------------------------- feature field
def test_pca_inverse_transform_roundtrip():
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden)
    flat = coords.reshape(-1, 2)
    assert np.allclose(proj.transform(proj.inverse_transform(flat)), flat, atol=1e-4)


def test_feature_field_shapes_and_determinism():
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden)
    sae = S.demo_sae(traj.dim, n_features=64)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 24)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 16)
    fld = S.feature_field(sae, proj, gx, gy)
    assert fld.magnitude.shape == fld.dominant.shape == fld.n_active.shape == (16, 24)
    assert np.isfinite(fld.magnitude).all() and (fld.magnitude >= 0).all()
    assert fld.dominant.min() >= -1 and fld.dominant.max() < 64
    assert ((fld.dominant == -1) == (fld.magnitude <= 0)).all()
    assert (fld.n_active >= (fld.magnitude > 0)).all()
    assert set(fld.features) == set(np.unique(fld.dominant[fld.dominant >= 0]))
    assert len(fld.features) > 1  # a field, not a single domain
    again = S.feature_field(sae, proj, gx, gy)
    assert np.array_equal(fld.dominant, again.dominant)
    assert np.array_equal(fld.magnitude, again.magnitude)


@pytest.mark.skipif("umap" not in __import__("projection").PROJECTIONS, reason="umap not registered")
def test_feature_field_with_umap_inverse_is_approximate_but_shape_safe():
    """umap's inverse_transform is an optimization-based approximation, not
    an exact map — the field must still come back finite and correctly
    shaped, even though the values away from the fitted data are not to be
    trusted as measurement (see the UI caption)."""
    pytest.importorskip("umap")
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden, method="umap")
    sae = S.demo_sae(traj.dim, n_features=32)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 12)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 10)
    fld = S.feature_field(sae, proj, gx, gy)
    assert fld.magnitude.shape == (10, 12)
    assert np.isfinite(fld.magnitude).all()
    assert fld.dominant.min() >= -1


def test_feature_field_requires_an_invertible_projection():
    traj = synthetic.capture(PROMPT)
    sae = S.demo_sae(traj.dim, 16)

    class NoInverse:
        pass

    with pytest.raises(ValueError, match="inverse"):
        S.feature_field(sae, NoInverse(), np.zeros(4), np.zeros(4))

    class WrongDim:
        def inverse_transform(self, Y):
            return np.zeros((len(Y), traj.dim + 1), np.float32)

    with pytest.raises(ValueError, match="dim"):
        S.feature_field(sae, WrongDim(), np.zeros(4), np.zeros(4))


def test_field_rgb_domain_coloring():
    from ui import field_rgb

    fld = S.FeatureField(
        grid_x=np.array([0.0, 1.0], np.float32),
        grid_y=np.array([0.0, 1.0], np.float32),
        magnitude=np.array([[0.0, 1.0], [0.5, 2.0]], np.float32),
        dominant=np.array([[-1, 3], [5, 3]], np.int32),
        n_active=np.array([[0, 2], [1, 3]], np.int32),
    )
    rgb = field_rgb(fld)
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 2, 3)
    dead = rgb[0, 0]
    assert dead[0] == dead[1] == dead[2] and dead[0] < 20  # achromatic void
    assert not np.array_equal(rgb[0, 1], rgb[1, 0])  # different features, different hue
    assert rgb[1, 1].max() > dead.max()  # firing cells outshine the void


def test_render_feature_field_modes():
    from config import MarbleConfig
    from ui import render_feature_field, run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_pipeline(cfg, PROMPT, **synthetic.mt())
    traj = result["traj"]
    assert hasattr(result["projector"], "inverse_transform")
    sae = S.demo_sae(traj.dim, 64)
    land = result["landscape"]
    fld = S.feature_field(sae, result["projector"], land.grid_x, land.grid_y)
    path = result["coords"][:, -1, :2]

    flat = render_feature_field(fld, sae, path=path)
    assert flat.data[0].type == "image"
    assert np.asarray(flat.data[0].z).shape == (len(land.grid_y), len(land.grid_x), 3)
    assert flat.data[1].type == "scatter" and len(flat.data[1].x) == traj.n_layers

    relief = render_feature_field(fld, sae, path=path, relief=True)
    assert relief.data[0].type == "surface"
    assert relief.data[1].type == "scatter3d"


def test_streamlit_app_feature_field():
    """Drive the real app with the domain-coloring field enabled."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPT)
    at.selectbox(key="model").select("gpt2")
    at.checkbox(key="sae_field").check()
    at.button(key="run").click()
    at.run()
    assert not at.exception
    at.radio(key="field_mode").set_value("relief")
    at.run()
    assert not at.exception


# ----------------------------------------------- residual decomposition
def test_tiny_components_sum_to_updates():
    traj = synthetic.capture(PROMPT, capture_components=True)
    traj.validate()
    comps = traj.components
    L, T, D = traj.hidden.shape
    assert set(comps) == {"attn", "mlp"}
    assert comps["attn"].shape == comps["mlp"].shape == (L - 1, T, D)
    updates = np.diff(traj.hidden, axis=0)
    assert np.allclose(comps["attn"] + comps["mlp"], updates, atol=1e-5)
    # off by default, and deterministic when on
    assert synthetic.capture(PROMPT).components is None
    again = synthetic.capture(PROMPT, capture_components=True)
    assert np.array_equal(comps["attn"], again.components["attn"])


def test_component_shares():
    traj = synthetic.capture(PROMPT, capture_components=True)
    shares = M.component_shares(traj, token=-1)
    assert shares.shape == (traj.n_layers - 1, 2)
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert (shares >= 0).all() and (shares <= 1).all()
    with pytest.raises(ValueError):
        M.component_shares(synthetic.capture(PROMPT))


def test_summary_includes_attn_share():
    from projection import project

    traj = synthetic.capture(PROMPT, capture_components=True)
    coords, _ = project(traj.hidden)
    out = M.summarize(traj, coords, token=-1)
    assert 0.0 <= out["avg_attn_share"] <= 1.0


# ------------------------------------------------- torch component capture
def test_torch_component_capture_exact():
    """attn + mlp writes reproduce the residual stream exactly (pre-norm)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from capture import capture

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_components=True)
    traj.validate()
    updates = np.diff(traj.hidden, axis=0)
    total = traj.components["attn"] + traj.components["mlp"]
    assert np.allclose(total, updates, atol=1e-4)


def test_torch_component_capture_gpt2_layout():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from capture import capture

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=VOCAB_SIZE, n_embd=32,
                                  n_layer=2, n_head=4, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_components=True)
    updates = np.diff(traj.hidden, axis=0)
    total = traj.components["attn"] + traj.components["mlp"]
    assert np.allclose(total, updates, atol=1e-4)


# ----------------------------------------------------------------- UI
def test_pipeline_and_overlay_render():
    from config import MarbleConfig
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_pipeline(cfg, PROMPT, **synthetic.mt())
    traj = result["traj"]
    assert traj.components is not None  # capture_components defaults on

    sae = S.demo_sae(traj.dim, cfg.sae_features)
    acts = S.feature_trajectory(traj, sae)
    feat = int(S.active_features(acts, k=1)[0])
    overlay = [acts[:, t.token, feat] for t in result["trajectories"]]

    fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                 overlay=overlay, overlay_label=f"f{feat}")
    lines = [tr for tr in fig.data if tr.type == "scatter3d" and tr.mode == "lines+markers"]
    assert len(lines) == traj.n_tokens
    assert all(len(tr.marker.color) == traj.n_layers for tr in lines)
    assert lines[0].marker.showscale and not lines[1].marker.showscale


def test_streamlit_app_sae_overlay():
    """Drive the real app with the SAE overlay enabled."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPT)
    at.selectbox(key="model").select("gpt2")
    at.checkbox(key="sae_overlay").check()
    at.button(key="run").click()
    at.run()
    assert not at.exception
    at.selectbox(key="feature").select(at.selectbox(key="feature").options[1])
    at.run()
    assert not at.exception


# ------------------------------------------------------------- hub fetching

def _fake_saelens_repo(tmp_path, cfg: dict, seed=3):
    """Write sae_weights.safetensors + cfg.json the way SAELens releases do."""
    import json

    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    d, f = 8, 16
    sd = {"W_enc": rng.normal(size=(d, f)).astype(np.float32),
          "b_enc": rng.normal(size=f).astype(np.float32),
          "W_dec": rng.normal(size=(f, d)).astype(np.float32),
          "b_dec": rng.normal(size=d).astype(np.float32)}
    folder = tmp_path / "blocks.8.hook_resid_pre"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(folder / "sae_weights.safetensors"))
    (folder / "cfg.json").write_text(json.dumps(cfg))
    return sd, folder


def _patch_hub(monkeypatch, folder):
    import huggingface_hub

    def fake_download(filename, repo_id=None, subfolder=None, revision=None,
                      cache_dir=None, **kw):
        return str(folder / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)


def test_fetch_from_hub_converts_exactly(tmp_path, monkeypatch):
    sd, folder = _fake_saelens_repo(
        tmp_path, {"architecture": "standard", "activation_fn_str": "relu",
                   "apply_b_dec_to_input": True, "normalize_activations": "none"})
    _patch_hub(monkeypatch, folder)
    out = S.fetch_from_hub("any/repo", "blocks.8.hook_resid_pre")
    h = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    ref = np.maximum((h - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"], 0.0)
    np.testing.assert_allclose(out.encode(h), ref, rtol=1e-6)


def test_fetch_from_hub_folds_b_dec(tmp_path, monkeypatch):
    """apply_b_dec_to_input=False sources still convert exactly."""
    sd, folder = _fake_saelens_repo(
        tmp_path, {"architecture": "standard", "apply_b_dec_to_input": False})
    _patch_hub(monkeypatch, folder)
    out = S.fetch_from_hub("any/repo", "blocks.8.hook_resid_pre")
    h = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
    ref = np.maximum(h @ sd["W_enc"] + sd["b_enc"], 0.0)  # the source forward
    np.testing.assert_allclose(out.encode(h), ref, rtol=1e-5, atol=1e-6)


def test_fetch_from_hub_rejects_nonstandard(tmp_path, monkeypatch):
    _, folder = _fake_saelens_repo(tmp_path, {"architecture": "jumprelu"})
    _patch_hub(monkeypatch, folder)
    with pytest.raises(ValueError, match="architecture"):
        S.fetch_from_hub("any/repo", "blocks.8.hook_resid_pre")


def test_fetch_from_hub_rejects_normalized(tmp_path, monkeypatch):
    _, folder = _fake_saelens_repo(
        tmp_path, {"architecture": "standard",
                   "normalize_activations": "expected_average_only_in"})
    _patch_hub(monkeypatch, folder)
    with pytest.raises(ValueError, match="normalize"):
        S.fetch_from_hub("any/repo", "blocks.8.hook_resid_pre")


# ------------------------------------------------------------ fit reporting

def test_fit_report_identity_reconstructs_perfectly():
    traj = synthetic.capture(PROMPT)
    fit = S.fit_report(_identity_sae(traj.dim), traj)
    L = traj.n_layers
    assert fit.recon_error.shape == (L,) and fit.active_frac.shape == (L,)
    # identity dictionary reconstructs every state exactly (up to ReLU halving:
    # only non-negative components survive, so error is not zero — but the
    # ordering vs an unrelated dictionary is what the UI relies on)
    assert 0 <= fit.best_layer < L
    assert fit.best_error == fit.recon_error[fit.best_layer]


def test_fit_report_flags_mismatched_dictionary():
    """A dictionary that fits the activations reads near-zero error; an
    unrelated one reads high — the signal the UI's warning rests on."""
    traj = synthetic.capture(PROMPT)
    D = traj.dim
    # exact ReLU autoencoder: acts = [ReLU(h), ReLU(-h)], decode restores h
    eye = np.eye(D, dtype=np.float32)
    signed = np.concatenate([eye, -eye], axis=1)          # (D, 2D)
    matched = S.SAE(w_enc=signed, b_enc=np.zeros(2 * D, np.float32),
                    w_dec=signed.T.copy(), b_dec=np.zeros(D, np.float32))
    mismatched = S.demo_sae(D, 2 * D, seed=9)
    fit_m = S.fit_report(matched, traj)
    fit_x = S.fit_report(mismatched, traj)
    assert fit_m.best_error < 1e-5
    assert fit_x.best_error > 10 * max(fit_m.best_error, 1e-6)


# ------------------------------------------------------- feature labels (M2)
def _explanation(desc="a description", model="gpt-3.5-turbo",
                 method="oai_token-act-pair", scores=None):
    return {"explanations": [{"description": desc, "explanationModelName": model,
                              "typeName": method, "scores": scores or []}]}


def test_fetch_labels_reads_and_caches(tmp_path):
    calls = []

    def fake(url):
        calls.append(url)
        idx = url.rsplit("/", 1)[-1]
        return _explanation(desc=f"feature {idx} fires on things")

    src = ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre")
    got = S.fetch_labels([7, 3], source=src, cache_dir=tmp_path, fetch=fake)
    assert set(got) == {3, 7}
    assert got[3].description == "feature 3 fires on things"
    assert got[3].explained_by == "gpt-3.5-turbo"      # provenance travels
    assert got[3].method == "oai_token-act-pair"
    assert len(calls) == 2

    # a second look is free, and does not touch the network at all
    def explode(url):
        raise AssertionError("should have been cached")

    again = S.fetch_labels([3, 7], source=src, cache_dir=tmp_path, fetch=explode)
    assert {i: l.description for i, l in again.items()} == \
           {i: l.description for i, l in got.items()}


def test_fetch_labels_never_raises_offline(tmp_path):
    """An unreachable source yields no names; it must not break the overlay."""
    def down(url):
        raise OSError("no network")

    src = ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre")
    assert S.fetch_labels([1, 2], source=src, cache_dir=tmp_path, fetch=down) == {}
    # an unknown source is simply not looked up
    assert S.fetch_labels([1], source=("who/what", "where"), cache_dir=tmp_path) == {}


def test_apply_labels_keeps_bare_names_for_unexplained_features():
    sae = S.demo_sae(8, n_features=6, seed=2)
    S.apply_labels(sae, {2: S.FeatureLabel(index=2, description="capital cities")})
    assert sae.feature_label(2) == "f2 · capital cities"
    assert sae.feature_label(5) == "f5"          # untouched, still an index
    sae.validate()                                # labels stay length-F


def test_label_provenance_names_the_writer_not_the_feature():
    """Auto-interp text describes correlates, not computation — the surface
    has to say so, and say who wrote it."""
    from ui import _label_provenance

    note = _label_provenance({1: S.FeatureLabel(1, "capitals", explained_by="gpt-3.5-turbo")})
    assert "auto-generated" in note
    assert "gpt-3.5-turbo" in note
    assert "not of what it computes" in note or "not what it computes" in note


@pytest.mark.network
def test_real_labels_name_the_capital_feature():
    """The point of the whole M2 milestone, checked against the live source:
    the feature that fires hardest on 'the capital of france is' is one whose
    published explanation is about capital cities."""
    from capture import capture

    src = ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre")
    real = S.fetch_from_hub(*src)
    traj = capture("gpt2", "The capital of France is")
    acts = S.feature_trajectory(traj, real)
    top = [int(f) for f, _ in S.top_features(acts, 8, -1, k=5)]

    labels = S.fetch_labels(top, source=src)
    assert labels, "expected published explanations for these features"
    text = " ".join(l.description.lower() for l in labels.values())
    assert "capital" in text, f"no capital-related feature among {top}: {text}"
    assert all(l.explained_by for l in labels.values())   # provenance present


def test_field_domains_rank_by_area_with_plane_centroids():
    from projection import project

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden, method="pca")
    d = S.demo_sae(traj.dim, 64, seed=1)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 24)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 24)
    field = S.feature_field(d, proj, gx, gy)

    doms = field.domains(k=4)
    assert 0 < len(doms) <= 4
    assert [x.area for x in doms] == sorted((x.area for x in doms), reverse=True)
    assert sum(x.area for x in field.domains(k=10**6)) == pytest.approx(1.0, abs=1e-6)
    for x in doms:                       # centroids land inside the plane
        assert gx.min() <= x.x <= gx.max() and gy.min() <= x.y <= gy.max()
        assert int(x.feature) in set(field.features.tolist())


def test_feature_field_names_domains_but_colours_by_identity():
    """The deliberate non-choice: hue stays keyed to the feature id. Recolouring
    by label meaning would imply a semantic metric the palette cannot carry and
    destroy the golden-angle separation that makes neighbouring ids legible."""
    from projection import project
    from render import _feature_hue, render_feature_field

    traj = synthetic.capture(PROMPT)
    coords, proj = project(traj.hidden, method="pca")
    d = S.demo_sae(traj.dim, 64, seed=1)
    gx = np.linspace(coords[..., 0].min(), coords[..., 0].max(), 20)
    gy = np.linspace(coords[..., 1].min(), coords[..., 1].max(), 20)
    field = S.feature_field(d, proj, gx, gy)
    top = field.domains(k=1)[0].feature

    bare = render_feature_field(field, sae=d)
    assert not bare.layout.annotations, "unlabelled dictionary names nothing"

    S.apply_labels(d, {top: S.FeatureLabel(top, "capital cities")})
    named = render_feature_field(field, sae=d, labels_k=3)
    texts = [a.text for a in named.layout.annotations]
    assert texts and any("capital cities" in t for t in texts)
    # hue is unchanged by naming: colour encodes which feature, not what it means
    assert _feature_hue(top) == _feature_hue(top)
    assert float(_feature_hue(np.array([top]))[0]) == pytest.approx(
        (top * 0.6180339887498949) % 1.0)


def test_failed_label_lookups_are_reported_not_dropped(tmp_path, capsys):
    """Offline is not an error, but it must not be silent: one line on stderr
    says how many lookups failed and why."""
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) % 2:
            raise TimeoutError("timed out")
        return {"explanations": [{"description": "a capital city"}]}

    src = ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre")
    got = S.fetch_labels([1, 2, 3, 4], source=src, cache_dir=tmp_path, fetch=flaky)
    assert sorted(got) == [2, 4]
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "2 of 4 Neuronpedia label lookups failed" in err and "TimeoutError" in err
