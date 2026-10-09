"""The command line as a Unix tool: stages that pipe, stdout for data only,
quiet by default, conventional exit codes, no clobbering, one-line errors.

Runs offline against a tiny checkpoint saved to disk (`tiny.save_to`), so the
CLI loads it by path exactly as it would load a hub model.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cli
import statefile
import tiny

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ["the capital of france is", "the cat sat on the mat"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return tiny.save_to(tmp_path_factory.mktemp("tiny") / "model")


def _stdin(monkeypatch, data: bytes | str):
    raw = data.encode() if isinstance(data, str) else data
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))


def _same_scene(a: Path | bytes, b: Path | bytes):
    """Two scenes are the same if everything but the record's timestamp is."""
    def load(x):
        return statefile.read_container(io.BytesIO(x) if isinstance(x, bytes) else x)
    (ma, aa), (mb, ab) = load(a), load(b)
    ma["analysis"].pop("created"), mb["analysis"].pop("created")
    assert ma == mb
    assert set(aa) == set(ab)
    for k in aa:
        np.testing.assert_array_equal(aa[k], ab[k], err_msg=k)


# ------------------------------------------------------------ composition
def test_capture_then_project_is_export(tmp_path, model_dir):
    traj, piped, exported = tmp_path / "t.mtj", tmp_path / "p.mtj", tmp_path / "e.mtj"
    assert cli.main(["capture", *PROMPTS, "--model", model_dir, "-o", str(traj)]) == 0
    assert cli.main(["project", str(traj), "-o", str(piped)]) == 0
    assert cli.main(["export", *PROMPTS, "--model", model_dir, "-o", str(exported)]) == 0
    _same_scene(piped, exported)


def test_capture_writes_one_container_per_prompt(tmp_path, model_dir):
    out = tmp_path / "t.mtj"
    assert cli.main(["capture", *PROMPTS, "--model", model_dir, "-o", str(out)]) == 0
    loaded = statefile.load_stream(out)
    assert [t.meta["prompt"] for t, _, _ in loaded] == PROMPTS
    # lean by default: the inspector travels instead of the (V, D) matrix
    t, insp, manifest = loaded[0]
    assert t.embedding_matrix is None and insp and "neighbor_idx" in insp
    assert manifest["schema"] == "mottled-trajectory/1"
    assert manifest["analysis"]["config"]["model"] == model_dir


def test_streams_concatenate_like_text(tmp_path, model_dir):
    """`cat a.mtj b.mtj | mottled project` sees both runs."""
    a, b = tmp_path / "a.mtj", tmp_path / "b.mtj"
    cli.main(["capture", PROMPTS[0], "--model", model_dir, "-o", str(a)])
    cli.main(["capture", PROMPTS[1], "--model", model_dir, "-o", str(b)])
    both = tmp_path / "both.mtj"
    both.write_bytes(a.read_bytes() + b.read_bytes())
    s1, s2 = tmp_path / "s1.mtj", tmp_path / "s2.mtj"
    assert cli.main(["project", str(both), "-o", str(s1)]) == 0
    assert cli.main(["project", str(a), str(b), "-o", str(s2)]) == 0
    _same_scene(s1, s2)
    assert len(statefile.load_scene(s1)["runs"]) == 2


def test_stdin_and_stdout_in_process(monkeypatch, capsysbinary, model_dir):
    _stdin(monkeypatch, "\n".join(PROMPTS) + "\n\n")      # blank lines skipped
    assert cli.main(["capture", "--model", model_dir]) == 0
    out, err = capsysbinary.readouterr()
    assert out[:4] == b"MTRJ" and err == b""            # data only; silent
    assert len(statefile.read_stream(io.BytesIO(out))) == 2

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(out)))
    assert cli.main(["project"]) == 0
    scene, err = capsysbinary.readouterr()
    assert err == b"" and statefile.load_scene(io.BytesIO(scene))["runs"][1]["prompt"] == PROMPTS[1]


def test_a_real_pipe_between_processes(tmp_path, model_dir):
    """The claim in the README, run as a shell would run it."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    py = [sys.executable, str(ROOT / "cli.py")]
    cap = subprocess.Popen(py + ["capture", "--model", model_dir], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    proj = subprocess.Popen(py + ["project"], stdin=cap.stdout, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    cap.stdin.write(("\n".join(PROMPTS) + "\n").encode())
    cap.stdin.close()
    cap.stdout.close()
    scene, proj_err = proj.communicate(timeout=300)
    cap_err = cap.stderr.read()
    assert cap.wait(timeout=60) == 0 and proj.returncode == 0, (cap_err, proj_err)
    assert cap_err == b"" and proj_err == b""
    exported = tmp_path / "e.mtj"
    assert cli.main(["export", *PROMPTS, "--model", model_dir, "-o", str(exported)]) == 0
    _same_scene(scene, exported)


def test_export_dash_means_stdout(capsysbinary, model_dir):
    assert cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", "-"]) == 0
    out, err = capsysbinary.readouterr()
    assert out[:4] == b"MTRJ" and not Path("-").exists() and err == b""


def test_export_dash_prompt_reads_stdin(monkeypatch, tmp_path, model_dir):
    _stdin(monkeypatch, PROMPTS[1] + "\n")
    out = tmp_path / "s.mtj"
    assert cli.main(["export", "-", "--model", model_dir, "-o", str(out)]) == 0
    assert statefile.load_scene(out)["runs"][0]["prompt"] == PROMPTS[1]


def test_project_takes_policy_from_flags_config_and_set(tmp_path, model_dir):
    traj = tmp_path / "t.mtj"
    cli.main(["capture", PROMPTS[0], "--model", model_dir, "-o", str(traj)])
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"grid_size": 24}))
    out = tmp_path / "s.mtj"
    assert cli.main(["project", str(traj), "--config", str(conf), "--bootstrap", "0",
                     "--set", "smooth_sigma=0.5", "-o", str(out)]) == 0
    rec = statefile.load_scene(out)["analysis"]["config"]
    assert (rec["grid_size"], rec["density_bootstrap"], rec["smooth_sigma"]) == (24, 0, 0.5)
    # capture-stage settings come from the trajectory, not from defaults
    assert rec["model"] == model_dir
    assert statefile.load_scene(out)["terrain"]["z"].shape == (24, 24)


# ----------------------------------------------------- inspect / validate
def test_inspect_json_and_ndjson(tmp_path, capsys, model_dir):
    scene = tmp_path / "s.mtj"
    cli.main(["export", *PROMPTS, "--model", model_dir, "-o", str(scene)])
    assert cli.main(["inspect", str(scene), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary[0]["kind"] == "scene" and len(summary[0]["runs"]) == 2
    assert summary[0]["schema"] == "mottled-scene/1"

    assert cli.main(["inspect", str(scene), "--ndjson"]) == 0
    lines = capsys.readouterr().out.splitlines()
    recs = [json.loads(line) for line in lines]          # every line is JSON
    L = summary[0]["runs"][0]["layers"]
    assert len(recs) == L * sum(len(p.split()) for p in PROMPTS)
    assert {"run", "layer", "token", "text", "entropy", "quality", "x", "y", "z"} <= set(recs[0])

    assert cli.main(["inspect", str(scene)]) == 0
    assert "2 run(s)" in capsys.readouterr().out


def test_inspect_ndjson_of_a_trajectory_stream(tmp_path, capsys, model_dir):
    traj = tmp_path / "t.mtj"
    cli.main(["capture", *PROMPTS, "--model", model_dir, "-o", str(traj)])
    assert cli.main(["inspect", str(traj), "--ndjson"]) == 0
    recs = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert {r["container"] for r in recs} == {0, 1}
    assert all(r["norm"] is not None for r in recs)


def test_validate_is_silent_when_valid_and_loud_when_not(tmp_path, capsys, model_dir):
    scene = tmp_path / "s.mtj"
    cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", str(scene)])
    assert cli.main(["validate", str(scene)]) == 0
    assert capsys.readouterr() == ("", "")
    bad = tmp_path / "bad.mtj"
    bad.write_bytes(b"MTRJ" + scene.read_bytes()[4:12] + b"{")
    assert cli.main(["validate", str(scene), str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.startswith(f"{bad}: error:") and str(scene) not in out


def test_validate_json_report(tmp_path, capsys):
    sample = ROOT / "viewer" / "samples" / "scene-abc.mtj"
    assert cli.main(["validate", "--json", str(sample)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report[0]["valid"] is True
    # an old sample has no schema id: a warning now, an error under --strict
    assert any("schema id" in p["message"] for p in report[0]["problems"])
    assert cli.main(["validate", "--strict", str(sample)]) == 1


# ------------------------------------------------------- CLI conventions
def test_version_goes_to_stdout(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("mottled ") and out.strip() != "mottled unknown"


def test_help_lists_each_command_on_one_line_with_examples(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("capture", "project", "inspect", "validate", "export"):
        assert any(line.split()[:1] == [cmd] and len(line.split()) > 2
                   for line in out.splitlines()), cmd
    assert "examples:" in out and "| mottled project" in out


@pytest.mark.parametrize("argv, code, needle", [
    (["capture", ""], 2, "empty prompt"),
    (["capture", "  "], 2, "empty prompt"),
    (["project", "--set", "nonsense=1", "/dev/null"], 1, ""),   # file read first
    (["inspect", "no/such/file.mtj"], 1, "no such file"),
    (["validate", "no/such/file.mtj"], 1, "no such file"),
    (["export-manifest", "no/such/file.mtj"], 1, "no such file"),
])
def test_errors_are_one_line_with_the_right_exit_code(argv, code, needle, capsys):
    assert cli.main(argv) == code
    err = capsys.readouterr().err
    assert err.startswith("mottled: error:") and "Traceback" not in err
    assert needle in err


def test_usage_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["no-such-command"])
    assert e.value.code == 2
    assert cli.main(["project", "--set", "nonsense=1", str(ROOT / "viewer/samples/single.mtj")]) == 1


def test_empty_stdin_is_a_usage_error(monkeypatch, capsys):
    _stdin(monkeypatch, "\n\n")
    assert cli.main(["capture"]) == 2
    assert "stdin was empty" in capsys.readouterr().err


def test_unknown_model_fails_before_any_work(tmp_path, capsys):
    out = tmp_path / "x.mtj"
    assert cli.main(["capture", "hello", "--model", str(tmp_path / "nope"),
                     "-o", str(out)]) == 1
    assert "unknown model" in capsys.readouterr().err and not out.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["capture", "hello", "--model", str(empty), "-o", str(out)]) == 1
    assert "no config.json" in capsys.readouterr().err


def test_bad_input_is_named_in_one_line(tmp_path, capsys):
    junk = tmp_path / "junk.mtj"
    junk.write_text("hello")
    assert cli.main(["project", str(junk), "-o", str(tmp_path / "s.mtj")]) == 1
    err = capsys.readouterr().err
    assert err == f"mottled: error: {junk}: not a .mtj file (bad magic)\n"
    scene = ROOT / "viewer" / "samples" / "single.mtj"
    assert cli.main(["project", str(scene), "-o", str(tmp_path / "s.mtj")]) == 1
    assert "is a scene, not a trajectory" in capsys.readouterr().err


def test_tracebacks_only_when_debugging(tmp_path, monkeypatch, capsys):
    import pipeline

    def boom(*a, **k):
        raise RuntimeError("deliberate\nsecond line")
    monkeypatch.setattr(pipeline, "project_trajectories", boom)
    scene = tmp_path / "t.mtj"
    statefile.save(tiny.capture("the cat sat"), scene)
    assert cli.main(["project", str(scene), "-o", str(tmp_path / "s.mtj")]) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: deliberate" in err and "second line" not in err
    assert "Traceback" not in err
    with pytest.raises(RuntimeError):
        cli.main(["--debug", "project", str(scene), "-o", str(tmp_path / "s2.mtj")])
    monkeypatch.setenv("MOTTLED_DEBUG", "1")
    with pytest.raises(RuntimeError):
        cli.main(["project", str(scene), "-o", str(tmp_path / "s3.mtj")])


def test_refuses_to_overwrite_without_force(tmp_path, capsys, model_dir):
    out = tmp_path / "s.mtj"
    out.write_bytes(b"precious")
    assert cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", str(out)]) == 1
    assert "exists; pass -f/--force" in capsys.readouterr().err
    assert out.read_bytes() == b"precious"
    assert cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", str(out), "-f"]) == 0
    assert out.read_bytes()[:4] == b"MTRJ"


def test_refuses_binary_to_a_terminal(monkeypatch, capsys, model_dir):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli.main(["capture", "hello", "--model", model_dir]) == 2
    assert "refusing to write binary" in capsys.readouterr().err
    _stdin(monkeypatch, b"")
    assert cli.main(["project", "-o", "-", str(ROOT / "viewer/samples/single.mtj")]) == 2


def test_quiet_by_default_verbose_on_request(tmp_path, capsys, model_dir):
    out = tmp_path / "s.mtj"
    assert cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    assert cli.main(["export", PROMPTS[0], "--model", model_dir, "-o", str(out),
                     "-f", "-v"]) == 0
    got = capsys.readouterr()
    assert got.out == "" and f"wrote {out}" in got.err
    # -v works before the command as well as after it
    assert cli.main(["-v", "export", PROMPTS[0], "--model", model_dir, "-o", str(out),
                     "-f"]) == 0
    assert "wrote" in capsys.readouterr().err


def test_quiet_hides_warnings(tmp_path, capsys):
    sample = str(ROOT / "viewer" / "samples" / "scene-abc.mtj")
    assert cli.main(["validate", sample]) == 0
    assert "warning" in capsys.readouterr().out
    assert cli.main(["-q", "validate", sample]) == 0
    assert capsys.readouterr().out == ""


def test_no_color(monkeypatch, capsys):
    class TTY(io.StringIO):
        def isatty(self):
            return True
    fake = TTY()
    monkeypatch.setattr(sys, "stderr", fake)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    cli.main(["inspect", "no/such/file"])
    assert "\033[31m" in fake.getvalue()
    fake.seek(0), fake.truncate()
    monkeypatch.setenv("NO_COLOR", "1")
    cli.main(["inspect", "no/such/file"])
    assert "\033[" not in fake.getvalue() and "mottled: error:" in fake.getvalue()


def test_broken_pipe_is_quiet(tmp_path):
    """`mottled inspect --ndjson big.mtj | head -1` must not spray a traceback."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    sample = ROOT / "viewer" / "samples" / "gpt2-capitals.mtj"
    shell = (f"{sys.executable} {ROOT / 'cli.py'} inspect --ndjson {sample} | head -1")
    r = subprocess.run(["bash", "-c", shell], capture_output=True, env=env, timeout=120)
    assert json.loads(r.stdout.decode().splitlines()[0])["layer"] == 0
    assert b"Traceback" not in r.stderr and b"BrokenPipe" not in r.stderr
