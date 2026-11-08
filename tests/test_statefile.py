"""`.mtj` interchange format: container layout, round-trips, scene export."""

import io
import json
import struct

import numpy as np
import pytest

import statefile as F
from config import MarbleConfig
import tiny as synthetic
from ui import run_scene

PROMPT = "the capital of france is"


@pytest.fixture(scope="module")
def traj():
    return synthetic.capture(PROMPT, capture_components=True, capture_attention=True)


# ----------------------------------------------------------------- container
def test_container_layout(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    raw = path.read_bytes()
    assert raw[:4] == b"MTRJ"
    version, mlen = struct.unpack_from("<II", raw, 4)
    assert version == 1
    assert (12 + mlen) % 16 == 0                      # blob starts aligned
    manifest = json.loads(raw[12 : 12 + mlen].decode("utf-8"))
    assert manifest["format"] == "mottled-trajectory"
    assert manifest["kind"] == "trajectory"
    for ref in manifest["arrays"].values():
        assert ref["offset"] % 16 == 0                # every array aligned
        size = np.dtype(ref["dtype"]).itemsize * int(np.prod(ref["shape"]))
        assert ref["length"] == size


def test_reader_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mtj"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError, match="magic"):
        F.read_container(bad)
    versioned = tmp_path / "v9.mtj"
    versioned.write_bytes(b"MTRJ" + struct.pack("<II", 9, 0))
    with pytest.raises(ValueError, match="version"):
        F.read_container(versioned)


def test_kind_mismatch(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    with pytest.raises(ValueError, match="scene"):
        F.load_scene(path)


# ---------------------------------------------------------------- round-trip
def test_trajectory_roundtrip(tmp_path, traj):
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    back = F.load(path)
    back.validate()
    assert np.array_equal(back.hidden, traj.hidden)
    assert np.array_equal(back.entropy, traj.entropy)
    assert np.array_equal(back.logits, traj.logits)
    assert np.array_equal(back.attention, traj.attention)
    assert np.array_equal(back.components["attn"], traj.components["attn"])
    assert np.array_equal(back.embedding_matrix, traj.embedding_matrix)
    assert back.tokens == traj.tokens
    assert back.vocab == traj.vocab
    assert back.topk[0][0][0][0] == traj.topk[0][0][0][0]
    assert back.meta["prompt"] == PROMPT


def test_roundtrip_minimal_and_file_objects():
    lean = synthetic.capture(PROMPT, keep_logits=False)
    lean.entropy = None
    lean.topk = None
    buf = io.BytesIO()
    F.save(lean, buf, include_embeddings=False)
    buf.seek(0)
    back = F.load(buf)
    assert back.logits is None and back.entropy is None
    assert back.embedding_matrix is None and back.components is None
    assert np.array_equal(back.hidden, lean.hidden)


def test_reader_ignores_unknown_fields(tmp_path, traj):
    """Additive evolution: unknown manifest fields must not break readers."""
    path = tmp_path / "run.mtj"
    F.save(traj, path)
    raw = bytearray(path.read_bytes())
    _, mlen = struct.unpack_from("<II", raw, 4)
    manifest = json.loads(raw[12 : 12 + mlen].decode("utf-8"))
    manifest["future_field"] = {"anything": [1, 2, 3]}
    blob = json.dumps(manifest).encode("utf-8")
    blob += b" " * ((-(12 + len(blob))) % 16)
    patched = raw[:8] + struct.pack("<I", len(blob)) + blob + raw[12 + mlen:]
    out = tmp_path / "future.mtj"
    out.write_bytes(patched)
    assert np.array_equal(F.load(out).hidden, traj.hidden)


# -------------------------------------------------------------------- scenes
@pytest.fixture(scope="module")
def scene_result():
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=8)
    return run_scene(cfg, [PROMPT, "the capital of germany is"], **synthetic.mt())


def test_scene_roundtrip(tmp_path, scene_result):
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    scene = F.load_scene(path)
    assert scene["kind"] == "scene"
    assert len(scene["runs"]) == 2
    mesh = scene_result["mesh"]
    assert np.array_equal(scene["terrain"]["z"], mesh.z)
    for run, traj, trajectories in zip(scene["runs"], scene_result["trajs"],
                                       scene_result["trajectories_list"]):
        assert run["points"].shape == (len(trajectories), traj.n_layers, 3)
        assert np.array_equal(run["points"][0], trajectories[0].points)
        assert np.array_equal(run["entropy"], traj.entropy)
        assert run["attention"].shape == (traj.n_layers - 1, traj.n_tokens, traj.n_tokens)
        assert run["tokens"] == traj.tokens
        assert len(run["topk"]) == traj.n_layers
    cmp = scene["comparisons"][0]
    assert cmp["label"] == "B"
    assert cmp["hausdorff"] == pytest.approx(scene_result["comparisons"][0].hausdorff)
    # the per-layer divergence curves travel as arrays, not summaries
    assert np.array_equal(cmp["relative"], scene_result["comparisons"][0].relative)
    assert np.array_equal(cmp["cosine"], scene_result["comparisons"][0].cosine)
    import mtjschema
    assert mtjschema.validate_bytes(path.read_bytes(), strict=True) == []


def test_scene_carries_uncertainty_layers(tmp_path, scene_result):
    """Bootstrap SE, density and per-run projection quality survive export."""
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    scene = F.load_scene(path)
    assert np.array_equal(scene["terrain"]["se"], scene_result["landscape"].density_se)
    assert np.array_equal(scene["terrain"]["density"], scene_result["landscape"].density)
    for run, q in zip(scene["runs"], scene_result["quality_list"]):
        assert np.array_equal(run["quality"], q.preservation)


def test_scene_without_bootstrap_omits_se(tmp_path):
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, [PROMPT], **synthetic.mt())
    buf = io.BytesIO()
    F.save_scene(result, buf)
    buf.seek(0)
    scene = F.load_scene(buf)
    assert "se" not in scene["terrain"]
    # quality is always present; density too
    assert "density" in scene["terrain"]
    assert "quality" in scene["runs"][0]


def test_scene_contains_no_hidden_states(tmp_path, scene_result):
    """Scene files are viewer bundles: heavy capture arrays must stay out."""
    path = tmp_path / "scene.mtj"
    F.save_scene(scene_result, path)
    manifest, arrays = F.read_container(path)
    assert not any("hidden" in name or "logits" in name or "embedding" in name
                   for name in arrays)


def test_scene_from_single_run_pipeline(tmp_path):
    from ui import run_pipeline

    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_pipeline(cfg, PROMPT, **synthetic.mt())
    buf = io.BytesIO()
    F.save_scene(result, buf)
    buf.seek(0)
    scene = F.load_scene(buf)
    assert len(scene["runs"]) == 1
    assert "comparisons" not in scene


def test_scene_carries_sae_feature_layer(tmp_path):
    """ui.attach_features -> save_scene -> load_scene round-trips the
    dominant-feature layer and its measured fit; absent, nothing changes."""
    import sae as S
    from ui import attach_features, run_scene

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, [PROMPT, "the capital of germany is"], **synthetic.mt())
    sae = S.demo_sae(result["traj"].dim, 32, seed=4)
    attach_features(result, sae, source="test/repo", hook="blocks.8.hook_resid_pre")

    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)
    scene = F.load_scene(path)
    for run, traj in zip(scene["runs"], result["trajs"]):
        feats = run["features"]
        L, T = traj.n_layers, traj.n_tokens
        assert feats["source"] == "test/repo"
        assert feats["top_id"].shape == (L, T) and feats["top_id"].dtype == np.int32
        assert feats["top_act"].shape == (L, T)
        assert feats["recon_error"].shape == (L,)
        assert 0 <= feats["best_layer"] < L
        acts = sae.encode(traj.hidden)
        assert np.array_equal(feats["top_id"], acts.argmax(axis=-1))

    plain = run_scene(cfg, [PROMPT], **synthetic.mt())
    F.save_scene(plain, path)
    assert "features" not in F.load_scene(path)["runs"][0]


def test_scene_carries_inspector_layers(tmp_path):
    """Neighbors and the attn/MLP split cannot be recomputed from a scene
    file — the embedding matrix and the residual components are far too big
    to ship — so they are resolved before export and round-trip as compact
    arrays."""
    import numpy as np

    from neighbors import TokenNeighbors
    from pipeline import attach_inspector

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = attach_inspector(run_scene(cfg, [PROMPT], **synthetic.mt()), n_neighbors=5)
    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)
    run = F.load_scene(path)["runs"][0]
    traj = result["traj"]
    L, T = traj.n_layers, traj.n_tokens

    ins = run["inspector"]
    assert ins["idx"].shape == (L, T, 5) and ins["idx"].dtype == np.int32
    assert ins["sim"].shape == (L, T, 5)
    assert 0 < len(ins["tokens"]) <= L * T * 5      # compact table, not the vocab
    assert int(ins["idx"].max()) < len(ins["tokens"])

    # the stored neighbors are the ones the live index gives
    tn = TokenNeighbors(traj.embedding_matrix, traj.vocab)
    live = tn.nearest(traj.hidden[3, 2], k=5)
    stored = [ins["tokens"][j] for j in ins["idx"][3, 2]]
    assert [t for t, _ in live] == stored
    np.testing.assert_allclose([s for _, s in live], ins["sim"][3, 2], rtol=1e-5)

    shares = ins["component_shares"]
    assert shares.shape == (L - 1, T, 2)
    np.testing.assert_allclose(shares.sum(axis=-1), 1.0, atol=1e-5)


def test_scene_without_inspector_is_unchanged(tmp_path):
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    path = tmp_path / "plain.mtj"
    F.save_scene(run_scene(cfg, [PROMPT], **synthetic.mt()), path)
    assert "inspector" not in F.load_scene(path)["runs"][0]


def test_attach_inspector_skips_runs_without_the_source_data(tmp_path):
    """A run captured without components (or without an embedding matrix)
    contributes what it has instead of failing the whole export."""
    import tiny as synthetic
    from pipeline import _assemble_scene, attach_inspector

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    plain = synthetic.capture(PROMPT)              # no components captured
    result = {"prompts": [PROMPT], "prompt": PROMPT, **_assemble_scene(cfg, [plain])}
    attach_inspector(result)
    entry = result["inspector_list"][0]
    assert "neighbor_idx" in entry                  # embeddings were available
    assert "component_shares" not in entry          # components were not
    path = tmp_path / "partial.mtj"
    F.save_scene(result, path)
    ins = F.load_scene(path)["runs"][0]["inspector"]
    assert "idx" in ins and "component_shares" not in ins
