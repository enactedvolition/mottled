"""The analysis record: what `docs/validity.md` asks a user to version-lock,
made an affordance instead of a norm (provenance.py, pipeline.attach_manifest,
the `analysis` key in a .mtj manifest, `mottled export-manifest`)."""

import io
import json
from dataclasses import fields

import numpy as np
import pytest

import provenance as P
import sae as S
import statefile as F
import tiny as synthetic
from config import MarbleConfig
from ui import attach_manifest, run_pipeline, run_scene

PROMPT = "the capital of france is"


@pytest.fixture(scope="module")
def traj():
    return synthetic.capture(PROMPT)


# -------------------------------------------------------------------- record
def test_record_carries_the_whole_parameterization(traj):
    """Every knob, not a chosen subset: a reproduction attempt that has to
    guess one field is no better off than one that has to guess them all."""
    cfg = MarbleConfig(model="tiny", projection="pca", seed=7, grid_size=32)
    rec = P.record(cfg, prompts=[PROMPT], trajs=[traj])

    assert rec["schema"] == P.SCHEMA
    assert rec["created"].endswith("Z")
    assert rec["prompts"] == [PROMPT]
    assert set(rec["config"]) == {f.name for f in fields(MarbleConfig)}
    assert rec["config"]["seed"] == 7 and rec["config"]["grid_size"] == 32


def test_record_is_json_serializable(traj):
    """It is only a citable artifact if it survives `json.dumps` unchanged."""
    rec = P.record(MarbleConfig(model="tiny"), trajs=[traj])
    assert json.loads(json.dumps(rec)) == rec


def test_record_reports_the_environment():
    rec = P.record(MarbleConfig())
    env = rec["environment"]
    assert env["python"].startswith("3.")
    assert env["platform"]
    # the libraries that can move a number in a scene, and only those
    assert env["packages"]["numpy"] == np.__version__
    assert set(env["packages"]) <= set(P._PACKAGES)


def test_record_reports_the_captures_own_weights_identity(traj):
    """Device and precision come from the capture, not from the config: the
    config may say "auto"/"float32" while the model ran somewhere else."""
    rec = P.record(MarbleConfig(model="tiny", device="auto"), trajs=[traj])
    model, = rec["models"]
    assert model["backend"] == "transformers"
    assert model["device"] == traj.meta["device"]
    assert model["dtype"] == traj.meta["dtype"]
    assert model["family"] == traj.meta["family"]
    # locally-built weights resolve to no hub commit; the field is still there
    assert model["revision"] is None


def test_record_names_every_model_in_the_scene(traj):
    """A cross-model scene has one identity per run — recording only run 0's
    would name the wrong weights for every other one."""
    other = synthetic.capture(PROMPT, seed=3)
    other.meta = {**other.meta, "model": "other/model", "revision": "abc123"}
    rec = P.record(MarbleConfig(model="tiny"), trajs=[traj, other])
    # run 0 is a model built in-process, whose `name_or_path` is empty: the
    # config's name is the only identity there is, and beats recording nothing
    assert [m["id"] for m in rec["models"]] == ["tiny", "other/model"]
    assert rec["models"][1]["revision"] == "abc123"


def test_record_falls_back_to_the_config_without_a_capture():
    rec = P.record(MarbleConfig(model="gpt2"))
    assert rec["models"] == [{"id": "gpt2", "revision": None, "backend": None,
                              "family": None, "device": None, "dtype": None}]
    assert rec["prompts"] == []


# ----------------------------------------------------------------- SAE hashes
def test_sae_digest_is_content_addressed():
    sae = S.demo_sae(8, 16, seed=1)
    same = S.demo_sae(8, 16, seed=1)
    assert P.sae_digest(sae) == P.sae_digest(same)

    moved = S.demo_sae(8, 16, seed=1)
    moved.w_enc = moved.w_enc.copy()
    moved.w_enc[0, 0] += 1e-3
    assert P.sae_digest(moved) != P.sae_digest(sae)
    assert P.sae_digest(S.demo_sae(8, 32, seed=1)) != P.sae_digest(sae)


def test_record_attaches_the_dictionary_identity(traj):
    sae = S.demo_sae(traj.dim, 16, seed=2)
    rec = P.record(MarbleConfig(model="tiny"), trajs=[traj], sae=sae,
                   sae_source="test/repo", sae_hook="blocks.8.hook_resid_pre")
    assert rec["sae"] == {"source": "test/repo",
                          "hook": "blocks.8.hook_resid_pre",
                          "sha256": P.sae_digest(sae),
                          "n_features": 16}


def test_record_omits_the_sae_block_when_no_dictionary_ran(traj):
    assert "sae" not in P.record(MarbleConfig(model="tiny"), trajs=[traj])


# ------------------------------------------------------------------ .mtj wire
def test_scene_carries_the_analysis_record(tmp_path):
    """attach_manifest -> save_scene -> load_scene: a shared scene states its
    own parameterization instead of relying on the sender's notes."""
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0, seed=5)
    result = run_scene(cfg, [PROMPT, "the capital of germany is"], **synthetic.mt())
    attach_manifest(result, cfg)

    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)
    rec = F.load_scene(path)["analysis"]
    assert rec["schema"] == P.SCHEMA
    assert rec["config"]["seed"] == 5
    assert rec["prompts"] == [PROMPT, "the capital of germany is"]
    assert len(rec["models"]) == 2


def test_explorer_export_sequence_round_trips(tmp_path):
    """The three layers the explorer attaches before writing a scene, in the
    order `ui.py` attaches them: inspector, features, then the record naming
    the dictionary those features came from."""
    from ui import attach_features, attach_inspector

    hook = "blocks.8.hook_resid_pre"
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, [PROMPT], **synthetic.mt())
    sae = S.demo_sae(result["traj"].dim, 16, seed=3)
    attach_inspector(result, n_neighbors=cfg.n_neighbors)
    attach_features(result, sae, source="test/repo", hook=hook)
    attach_manifest(result, cfg, sae=sae, sae_source="test/repo", sae_hook=hook)

    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)
    scene = F.load_scene(path)
    assert scene["analysis"]["sae"] == {"source": "test/repo", "hook": hook,
                                        "sha256": P.sae_digest(sae),
                                        "n_features": 16}
    assert scene["runs"][0]["features"]["source"] == "test/repo"
    assert "inspector" in scene["runs"][0]


def test_intervention_scene_records_what_ran(tmp_path):
    """The explorer attaches the record with its session config. With the
    decode slider up, that config asks for a continuation, but an
    intervention runs the prompt pass only (the edit replays that pass). The
    record has to state the config that ran, and the edits: without them it
    describes an untouched run rather than the one drawn."""
    from intervene import Perturb
    from ui import run_intervention

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0,
                       generate_tokens=3, generate_temperature=0.7)
    edits = [Perturb(2, np.full(32, 0.5, np.float32), token=-1)]
    result = run_intervention(cfg, PROMPT, edits, **synthetic.mt())
    attach_manifest(result, cfg)                    # as ui.py's export does

    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)
    rec = F.load_scene(path)["analysis"]
    assert not any("generation" in t.meta for t in result["trajs"])  # no decode
    assert rec["config"]["generate_tokens"] == 0
    assert rec["config"]["generate_temperature"] == 0.0
    assert rec["interventions"] == [[], [edits[0].record()]]


def test_scene_without_a_record_is_unchanged(tmp_path):
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    buf = io.BytesIO()
    F.save_scene(run_pipeline(cfg, PROMPT, **synthetic.mt()), buf)
    buf.seek(0)
    assert "analysis" not in F.load_scene(buf)


def test_single_run_result_records_its_one_prompt():
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = attach_manifest(run_pipeline(cfg, PROMPT, **synthetic.mt()), cfg)
    assert result["analysis"]["prompts"] == [PROMPT]


def test_trajectory_file_carries_the_record(tmp_path, traj):
    """The archival counterpart: a full-fidelity .mtj keeps its record too."""
    rec = P.record(MarbleConfig(model="tiny"), trajs=[traj])
    path = tmp_path / "run.mtj"
    F.save(traj, path, analysis=rec)
    manifest, _ = F.read_container(path)
    assert manifest["analysis"] == rec
    assert F.load(path).hidden.shape == traj.hidden.shape  # still a trajectory

    F.save(traj, path)
    assert "analysis" not in F.read_container(path)[0]


# -------------------------------------------------------------------- the CLI
def test_export_manifest_prints_the_record(tmp_path, capsys):
    import cli

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = attach_manifest(run_pipeline(cfg, PROMPT, **synthetic.mt()), cfg)
    path = tmp_path / "scene.mtj"
    F.save_scene(result, path)

    assert cli.main(["export-manifest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == result["analysis"]

    out = tmp_path / "methods.json"
    assert cli.main(["export-manifest", str(path), "-o", str(out)]) == 0
    assert json.loads(out.read_text()) == result["analysis"]


def test_export_writes_the_record_into_the_scene_and_beside_it(tmp_path, monkeypatch):
    """`mottled export` always embeds the record; `--manifest` also writes it
    out as the standalone JSON a methods section can cite."""
    import cli

    model_dir = synthetic.save_to(tmp_path / "tiny")
    scene, sidecar = tmp_path / "scene.mtj", tmp_path / "methods.json"
    assert cli.main(["export", PROMPT, "--model", model_dir,
                     "-o", str(scene), "--manifest", str(sidecar)]) == 0

    embedded = F.load_scene(scene)["analysis"]
    assert embedded["config"]["model"] == model_dir
    assert embedded["prompts"] == [PROMPT]
    assert json.loads(sidecar.read_text()) == embedded


def test_export_manifest_says_so_when_there_is_no_record(tmp_path, capsys):
    import cli

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    path = tmp_path / "bare.mtj"
    F.save_scene(run_pipeline(cfg, PROMPT, **synthetic.mt()), path)

    assert cli.main(["export-manifest", str(path)]) == 1
    assert "no analysis record" in capsys.readouterr().err
