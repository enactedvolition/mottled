"""mottled: capture, project and inspect transformer state trajectories.

The pipeline is a set of stages that compose through .mtj files and pipes.
Each stage reads stdin when given no file (or `-`), writes stdout when
piped (or given `-o -`), and keeps stdout for data: status goes to stderr,
and nothing is said on success unless you ask with -v.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path

EPILOG = """\
examples:
  mottled capture "The cat sat on the mat" | mottled project > scene.mtj
  printf 'the capital of france is\\nthe capital of germany is\\n' \\
      | mottled capture --model gpt2 | mottled project -o scene.mtj
  mottled capture "a prompt" -o run.mtj && mottled project run.mtj --projection umap > s.mtj
  mottled inspect scene.mtj                     # human summary
  mottled inspect scene.mtj --json | jq '.[0].runs[0]'
  mottled inspect scene.mtj --ndjson | jq -c 'select(.layer == 12)'
  mottled validate viewer/samples/*.mtj         # silent when valid; exit 1 if not
  mottled export "a prompt" -o scene.mtj        # capture | project in one step

exit status: 0 success, 1 runtime error or invalid input, 2 usage error.
environment: MOTTLED_DEBUG=1 (tracebacks), MOTTLED_CACHE_DIR (cache root),
NO_COLOR (plain stderr), HF_TOKEN (gated models).
"""

# MarbleConfig fields that belong to the capture stage. `mottled capture`
# records them in each trajectory's analysis record and `mottled project`
# takes them from there, so a scene's record states what actually ran.
CAPTURE_FIELDS = ("model", "device", "dtype", "keep_logits", "capture_components",
                  "capture_attention", "generate_tokens", "generate_temperature",
                  "top_k", "seed")


# ------------------------------------------------------------- plumbing
class UsageError(Exception):
    """Bad invocation: exit 2."""


class CLIError(Exception):
    """The command could not do its job: exit 1."""


class Ctx:
    """Verbosity and error style for one invocation."""

    def __init__(self, verbosity: int = 0, debug: bool = False):
        self.verbosity = verbosity
        self.debug = debug

    def _say(self, label: str, colour: str, msg: str) -> None:
        tag = f"{label}:"
        if _colour_ok(sys.stderr):
            tag = f"\033[{colour}m{tag}\033[0m"
        print(f"mottled: {tag} {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        self._say("error", "31", msg)

    def warn(self, msg: str) -> None:
        if self.verbosity >= 0:
            self._say("warning", "33", msg)

    def info(self, msg: str) -> None:
        """Status: shown with -v only (the Rule of Silence)."""
        if self.verbosity >= 1:
            print(f"mottled: {msg}", file=sys.stderr)


def _colour_ok(stream) -> bool:
    """Colour only on a terminal, and never when NO_COLOR is set
    (https://no-color.org) or TERM=dumb."""
    return (hasattr(stream, "isatty") and stream.isatty()
            and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb")


def _quiet_libraries(ctx: Ctx) -> None:
    """Keep library progress bars and advisories off stderr unless -vv.
    Set before transformers is imported, so its own defaults pick them up."""
    if ctx.verbosity < 2:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if ctx.verbosity < 0:
        import warnings

        warnings.simplefilter("ignore")


def _silence_transformers(ctx: Ctx) -> None:
    if ctx.verbosity >= 2:
        return
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # transformers missing or reorganised: nothing to quiet
        pass


class Sink:
    """Where a command's data goes: a file (written atomically, never
    clobbered without --force) or stdout (never binary to a terminal).

    Everything is checked in the constructor, before any expensive work,
    so a refused output costs nothing."""

    def __init__(self, path: str | None, force: bool, binary: bool = True,
                 what: str = ".mtj"):
        self.to_stdout = path in (None, "-")
        self.path = None if self.to_stdout else Path(path)
        self.binary = binary
        if self.to_stdout:
            if binary and sys.stdout.isatty() and not force:
                raise UsageError(
                    f"refusing to write binary {what} to a terminal; redirect it "
                    "(> FILE), pipe it into another command, or pass -o FILE")
        else:
            if self.path.is_dir():
                raise CLIError(f"{self.path}: is a directory")
            if self.path.exists() and not force:
                raise CLIError(f"{self.path} exists; pass -f/--force to overwrite")
            if not self.path.parent.exists():
                raise CLIError(f"{self.path.parent}: no such directory")
        self._buf = io.BytesIO()

    @property
    def name(self) -> str:
        return "stdout" if self.to_stdout else str(self.path)

    def write(self, data: bytes) -> None:
        if self.to_stdout:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        else:
            self._buf.write(data)

    def close(self) -> None:
        if self.to_stdout:
            return
        tmp = self.path.with_name(f".{self.path.name}.partial")
        tmp.write_bytes(self._buf.getvalue())
        tmp.replace(self.path)


def _read_inputs(paths: list[str], what: str = "a .mtj file") -> list[tuple[str, bytes]]:
    """(name, bytes) per input. No paths: stdin, unless stdin is a terminal.
    Every named file is checked before any is read."""
    if not paths:
        if sys.stdin.isatty():
            raise UsageError(f"no input: name {what} or pipe one in on stdin")
        paths = ["-"]
    for p in paths:
        if p != "-" and not Path(p).exists():
            raise CLIError(f"{p}: no such file")
        if p != "-" and Path(p).is_dir():
            raise CLIError(f"{p}: is a directory")
    out = []
    for p in paths:
        out.append(("<stdin>", sys.stdin.buffer.read()) if p == "-"
                   else (p, Path(p).read_bytes()))
    return out


def _prompts(given: list[str]) -> list[str]:
    """Prompts from arguments, with `-` (or no arguments and a piped stdin)
    meaning one prompt per line of stdin. Blank stdin lines are skipped; an
    explicitly empty argument is an error."""
    if not given:
        if sys.stdin.isatty():
            raise UsageError("no prompts: give PROMPT arguments, or pipe one "
                             "prompt per line on stdin")
        given = ["-"]
    out: list[str] = []
    for p in given:
        if p == "-":
            out += [line for line in sys.stdin.read().splitlines() if line.strip()]
        elif not p.strip():
            raise UsageError("empty prompt: a prompt needs at least one token")
        else:
            out.append(p)
    if not out:
        raise UsageError("no prompts: stdin was empty")
    return out


def _check_model(name: str) -> None:
    """Fail before loading weights when `name` cannot be a model: a local
    directory needs a config.json; anything else must resolve on the hub
    (fetching only its config — kilobytes)."""
    path = Path(name).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise CLIError(f"{name}: not a model directory (no config.json)")
        return
    try:
        import transformers
    except ImportError:
        raise CLIError('capturing needs torch and transformers: pip install "mottled[models]"')
    try:
        transformers.AutoConfig.from_pretrained(name)
    except (OSError, ValueError) as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise CLIError(f"unknown model {name!r}: not a local model directory or a "
                       f"Hugging Face model id the hub will serve ({first})")


def _set_field(cfg, assignment: str):
    """Apply one --set KEY=VALUE to a MarbleConfig, typed by the field."""
    from config import MarbleConfig

    key, sep, raw = assignment.partition("=")
    known = {f.name: f for f in fields(MarbleConfig)}
    if not sep or key not in known:
        raise UsageError(f"--set {assignment!r}: expected KEY=VALUE with KEY one of "
                         f"{', '.join(sorted(known))}")
    current = getattr(MarbleConfig(), key)
    try:
        if isinstance(current, bool):
            if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError("expected true or false")
            value = raw.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif current is None:
            value = None if raw.lower() in ("", "none", "null") else raw
        else:
            value = raw
    except ValueError as exc:
        raise UsageError(f"--set {key}: {exc}")
    return replace(cfg, **{key: value})


def _config_file(cfg, path: str):
    from config import MarbleConfig

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CLIError(f"{path}: no such file")
    except ValueError as exc:
        raise CLIError(f"{path}: not JSON ({exc})")
    if not isinstance(data, dict):
        raise CLIError(f"{path}: must be a JSON object of MarbleConfig fields")
    data = data.get("config", data)       # an analysis record works as-is
    known = {f.name for f in fields(MarbleConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise CLIError(f"{path}: unknown config field(s) {', '.join(unknown)}")
    return replace(cfg, **data)


# ------------------------------------------------------------- commands
def cmd_capture(args, ctx: Ctx) -> int:
    prompts = _prompts(args.prompts)
    sink = Sink(args.output, args.force)
    _quiet_libraries(ctx)
    _check_model(args.model)
    _silence_transformers(ctx)

    import pipeline
    import provenance
    import statefile
    from capture import load_model
    from config import MarbleConfig

    cfg = MarbleConfig(model=args.model, device=args.device, dtype=args.dtype,
                       generate_tokens=args.generate,
                       generate_temperature=args.temperature, seed=args.seed,
                       keep_logits=not args.no_logits, use_cache=False)
    model, tokenizer = load_model(cfg.model, device=cfg.device, dtype=cfg.dtype)
    for prompt in prompts:
        traj = pipeline._capture_with(cfg, prompt, model=model, tokenizer=tokenizer)
        inspector = pipeline.inspector_entry(traj)
        record = provenance.record(cfg, prompts=[prompt], trajs=[traj])
        if not args.full:
            # the (V, D) embedding matrix and vocabulary dwarf everything
            # else; the inspector layers computed from them travel instead
            traj = replace(traj, embedding_matrix=None, vocab=None)
        buf = io.BytesIO()
        statefile.save(traj, buf, analysis=record, inspector=inspector)
        sink.write(buf.getvalue())
        ctx.info(f"captured {prompt!r}: {traj.n_layers} layers x {traj.n_tokens} tokens")
    sink.close()
    ctx.info(f"wrote {len(prompts)} trajector{'y' if len(prompts) == 1 else 'ies'} "
             f"to {sink.name}")
    return 0


def _load_trajectories(inputs):
    import statefile

    trajs, inspectors, manifests = [], [], []
    for name, raw in inputs:
        try:
            for traj, insp, manifest in statefile.load_stream(io.BytesIO(raw)):
                trajs.append(traj)
                inspectors.append(insp)
                manifests.append(manifest)
        except ValueError as exc:
            msg = str(exc)
            if "expected kind 'trajectory'" in msg:
                msg = ("is a scene, not a trajectory; `mottled project` takes the "
                       "output of `mottled capture`")
            raise CLIError(f"{name}: {msg}")
        except (KeyError, struct_error()) as exc:
            raise CLIError(f"{name}: malformed .mtj ({type(exc).__name__}: {exc}); "
                           f"`mottled validate {name}` says where")
    return trajs, inspectors, manifests


def struct_error():
    import struct

    return struct.error


def cmd_project(args, ctx: Ctx) -> int:
    inputs = _read_inputs(args.inputs, "a trajectory .mtj")
    sink = Sink(args.output, args.force)

    import pipeline
    import statefile
    from config import MarbleConfig

    trajs, inspectors, manifests = _load_trajectories(inputs)
    if not trajs:
        raise CLIError("no trajectories in the input")
    # start from what the capture stage recorded, so the record is truthful
    captured = ((manifests[0].get("analysis") or {}).get("config") or {})
    cfg = MarbleConfig(**{k: captured[k] for k in CAPTURE_FIELDS if k in captured})
    models = {(m.get("meta") or {}).get("model") for m in manifests}
    if len(models) > 1:
        ctx.warn(f"inputs come from {len(models)} different models; the analysis "
                 "record lists the first one's capture settings")
    if args.config:
        cfg = _config_file(cfg, args.config)
    for flag, key in (("projection", "projection"), ("density", "density"),
                      ("seed", "seed"), ("grid_size", "grid_size"),
                      ("bootstrap", "density_bootstrap")):
        if getattr(args, flag) is not None:
            cfg = replace(cfg, **{key: getattr(args, flag)})
    for assignment in args.set or []:
        cfg = _set_field(cfg, assignment)
    cfg = replace(cfg, use_cache=False)

    result = pipeline.project_trajectories(cfg, trajs, inspectors)
    buf = io.BytesIO()
    statefile.save_scene(result, buf)
    sink.write(buf.getvalue())
    sink.close()
    ctx.info(f"projected {len(trajs)} run(s) with {cfg.projection}/{cfg.density} "
             f"to {sink.name}")
    return 0


def cmd_export(args, ctx: Ctx) -> int:
    prompts = _prompts(args.prompts)
    sink = Sink(args.output, args.force)
    if args.manifest and Path(args.manifest).exists() and not args.force:
        raise CLIError(f"{args.manifest} exists; pass -f/--force to overwrite")
    _quiet_libraries(ctx)

    import pipeline
    import statefile
    from config import MarbleConfig

    names = ([m.strip() for m in args.models.split(",") if m.strip()]
             if args.models else [args.model])
    for name in names:
        _check_model(name)
    _silence_transformers(ctx)
    cfg = MarbleConfig(model=args.model, use_cache=False,
                       generate_tokens=args.generate,
                       generate_temperature=args.temperature)

    if args.models:
        if len(prompts) > 1:
            ctx.warn(f"--models compares models on ONE prompt; using the first of "
                     f"{len(prompts)} and ignoring the rest")
        result = pipeline.run_model_scene(cfg, prompts[0], names)
        pipeline.attach_manifest(pipeline.attach_inspector(result), cfg)
        # the comparison is the result of this mode, not status: always shown
        print(f"mottled: {len(names)} models on {result['shared_vocab']} shared "
              "vocabulary entries", file=sys.stderr)
        for name, cmp in zip(names[1:], result["model_comparisons"]):
            print(f"mottled:   {names[0]} vs {name}: final JS "
                  f"{cmp.final_divergence:.4f}, top-1 {cmp.top_a!r} vs {cmp.top_b!r}",
                  file=sys.stderr)
    else:
        # exactly `mottled capture | mottled project`, without the pipe
        trajs = pipeline.capture_trajectories(cfg, prompts)
        result = pipeline.project_trajectories(cfg, trajs)

    buf = io.BytesIO()
    statefile.save_scene(result, buf)
    sink.write(buf.getvalue())
    sink.close()
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(statefile.finite_json(result["analysis"]), indent=2,
                       ensure_ascii=False, allow_nan=False) + "\n")
    ctx.info(f"wrote {sink.name}" + (f" and {args.manifest}" if args.manifest else ""))
    return 0


def cmd_export_blast(args, ctx: Ctx) -> int:
    inputs = _read_inputs([args.items] if args.items else [], "a JSONL items file")
    sink = Sink(args.output, args.force)
    items = []
    for name, raw in inputs:
        for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except ValueError as exc:
                    raise CLIError(f"{name}:{n}: not a JSON object ({exc})")
    if not items:
        raise CLIError("no items: the JSONL input is empty")
    _quiet_libraries(ctx)
    _check_model(args.model)
    _silence_transformers(ctx)

    import statefile
    from blast import BlastConfig
    from pipeline import run_blast

    cfg = BlastConfig(model=args.model, chat=args.chat, seed=args.seed,
                      methods=tuple(m.strip() for m in args.layouts.split(",")
                                    if m.strip()))
    result = run_blast(items, args.model, cfg=cfg)
    buf = io.BytesIO()
    statefile.save_scene(result, buf)
    sink.write(buf.getvalue())
    sink.close()
    names = [b.name for b in result["blast"]["layouts"]]
    ctx.info(f"wrote {sink.name}: {len(items)} pellets; layouts: {', '.join(names)}")
    for why in result["blast"]["skipped"]:
        ctx.warn(f"no monitor for {why}")
    if result["blast"]["spread"][0] > 1e-6:
        # no muzzle: the prompts already differ before any block runs, by
        # the read token or, with learned absolute positions (GPT-2), by
        # length, which every layer can then carry to the monitor
        ctx.warn(f"layer 0 is not one point (spread {result['blast']['spread'][0]:.3f}); "
                 "the read token or prompt length differs across items, and a "
                 "monitor can read it")
    return 0


def _inspect_summary(name: str, index: int, manifest: dict, arrays: dict) -> dict:
    import numpy as np

    def mean(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        return round(float(a.mean()), 6) if a.size else None

    analysis = manifest.get("analysis") or {}
    out = {"file": name, "container": index, "kind": manifest.get("kind"),
           "schema": manifest.get("schema"), "version": manifest.get("version"),
           "model": (manifest.get("meta") or {}).get("model"),
           "mottled": analysis.get("mottled"), "created": analysis.get("created")}
    if manifest.get("kind") == "scene":
        z = arrays.get(manifest.get("terrain", {}).get("z"))
        out["terrain"] = list(z.shape) if z is not None else None
        runs = []
        for run in manifest.get("runs", []):
            ent = arrays.get(run.get("entropy"))
            q = arrays.get(run.get("quality"))
            pts = arrays.get(run.get("points"))
            runs.append({
                "label": run.get("label"), "model": run.get("model"),
                "prompt": run.get("prompt"),
                "layers": int(ent.shape[0]) if ent is not None else None,
                "tokens": len(run.get("tokens", [])),
                "trajectories": int(pts.shape[0]) if pts is not None else None,
                "quality_mean": mean(q) if q is not None else None,
                "entropy_final_mean": mean(ent[-1]) if ent is not None else None,
                "layers_present": sorted(k for k in ("entropy", "quality", "attention",
                                                     "topk", "inspector", "features",
                                                     "generation") if k in run),
            })
        out["runs"] = runs
        out["comparisons"] = manifest.get("comparisons", [])
    elif manifest.get("kind") == "trajectory":
        h = arrays.get("hidden")
        out.update({"prompt": (manifest.get("meta") or {}).get("prompt"),
                    "layers": int(h.shape[0]) if h is not None else None,
                    "tokens": len(manifest.get("tokens", [])),
                    "dim": int(h.shape[2]) if h is not None else None})
    out["arrays"] = {k: {"dtype": v["dtype"], "shape": v["shape"]}
                     for k, v in manifest.get("arrays", {}).items()}
    return out


def _inspect_states(name: str, index: int, manifest: dict, arrays: dict):
    """One record per (run, layer, token): the scene's per-state data as
    a text stream."""
    import numpy as np

    def num(a, *ix):
        if a is None:
            return None
        try:
            v = float(a[ix])
        except (IndexError, TypeError):
            return None
        return v if np.isfinite(v) else None

    def top(topk, l, t):
        try:
            tok, p = topk[l][t][0]
            return tok, (p if p is None or np.isfinite(p) else None)
        except (IndexError, TypeError, ValueError):
            return None, None

    base = {"file": name, "container": index}
    if manifest.get("kind") == "trajectory":
        h = arrays.get("hidden")
        ent = arrays.get("entropy")
        norms = np.linalg.norm(h.astype(np.float64), axis=-1) if h is not None else None
        tokens = manifest.get("tokens", [])
        for l in range(h.shape[0] if h is not None else 0):
            for t, text in enumerate(tokens):
                tok, p = top(manifest.get("topk"), l, t)
                yield {**base, "layer": l, "token": t, "text": text,
                       "entropy": num(ent, l, t), "norm": num(norms, l, t),
                       "top1": tok, "top1_p": p}
        return
    for run in manifest.get("runs", []):
        tokens = run.get("tokens", [])
        ent = arrays.get(run.get("entropy"))
        q = arrays.get(run.get("quality"))
        pts = arrays.get(run.get("points"))
        feats = run.get("features") or {}
        fid, fact = arrays.get(feats.get("top_id")), arrays.get(feats.get("top_act"))
        L = ent.shape[0] if ent is not None else (pts.shape[1] if pts is not None else 0)
        per_token = pts is not None and pts.ndim == 3 and pts.shape[0] == len(tokens)
        for l in range(L):
            for t, text in enumerate(tokens):
                tok, p = top(run.get("topk"), l, t)
                rec = {**base, "run": run.get("label"), "layer": l, "token": t,
                       "text": text, "entropy": num(ent, l, t),
                       "quality": num(q, l, t), "top1": tok, "top1_p": p}
                if per_token and l < pts.shape[1]:
                    rec.update(x=num(pts, t, l, 0), y=num(pts, t, l, 1),
                               z=num(pts, t, l, 2))
                if fid is not None:
                    rec["feature"] = int(fid[l, t]) if l < fid.shape[0] else None
                    rec["feature_act"] = num(fact, l, t)
                yield rec


def _dumps(obj, **kw) -> str:
    import statefile

    return json.dumps(statefile.finite_json(obj), ensure_ascii=False, allow_nan=False, **kw)


def _format_summary(s: dict) -> str:
    head = f"{s['file']}"
    if s["container"]:
        head += f" [{s['container']}]"
    head += f": {s['kind']} ({s['schema'] or 'no schema id'})"
    lines = []
    if s["kind"] == "scene":
        head += f", {len(s['runs'])} run(s)"
        if s.get("terrain"):
            head += f", terrain {'x'.join(map(str, s['terrain']))}"
        lines.append(head)
        for r in s["runs"]:
            q = f"  preservation {r['quality_mean']:.3f}" if r["quality_mean"] is not None else ""
            lines.append(f"  {r['label']}  {r['model'] or '?'}  {r['layers']} layers x "
                         f"{r['tokens']} tokens{q}  {json.dumps(r['prompt'])}")
    else:
        lines.append(f"{head}, {s.get('model') or '?'}, {s.get('layers')} layers x "
                     f"{s.get('tokens')} tokens x {s.get('dim')} dims  "
                     f"{json.dumps(s.get('prompt'))}")
    return "\n".join(lines)


def cmd_inspect(args, ctx: Ctx) -> int:
    import statefile

    inputs = _read_inputs(args.inputs)
    summaries = []
    out = sys.stdout
    for name, raw in inputs:
        try:
            containers = list(statefile.iter_containers(raw))
        except (ValueError, KeyError, struct_error()) as exc:
            raise CLIError(f"{name}: {exc}")
        for index, (manifest, arrays) in enumerate(containers):
            if args.ndjson:
                for rec in _inspect_states(name, index, manifest, arrays):
                    out.write(_dumps(rec) + "\n")
            else:
                summaries.append(_inspect_summary(name, index, manifest, arrays))
    if args.json:
        out.write(_dumps(summaries, indent=2) + "\n")
    elif not args.ndjson:
        out.write("\n".join(_format_summary(s) for s in summaries) + "\n")
    out.flush()
    return 0


def cmd_validate(args, ctx: Ctx) -> int:
    import mtjschema

    inputs = _read_inputs(args.inputs)
    report, any_error = [], False
    for name, raw in inputs:
        problems = mtjschema.validate_bytes(raw, strict=args.strict)
        errors = [p for p in problems if p.level == "error"]
        any_error |= bool(errors)
        report.append({"file": name, "valid": not errors,
                       "problems": [{"level": p.level, "where": p.where,
                                     "message": p.message} for p in problems]})
        if args.json:
            continue
        for p in problems:
            if p.level == "error" or ctx.verbosity >= 0:
                print(f"{name}: {p.level}: {p}")
        if not errors:
            ctx.info(f"{name}: ok")
    if args.json:
        print(_dumps(report, indent=2))
    return 1 if any_error else 0


def cmd_export_manifest(args, ctx: Ctx) -> int:
    import statefile

    (name, raw), = _read_inputs([args.scene] if args.scene else [])
    sink = Sink(args.output, args.force, binary=False) if args.output not in (None, "-") else None
    try:
        manifest, _ = statefile.read_container(io.BytesIO(raw))
    except (ValueError, KeyError, struct_error()) as exc:
        raise CLIError(f"{name}: {exc}")
    record = manifest.get("analysis")
    if record is None:
        raise CLIError(f"{name}: no analysis record — written either before "
                       "provenance export existed or by another producer")
    text = _dumps(record, indent=2) + "\n"
    if sink is None:
        sys.stdout.write(text)
    else:
        sink.write(text.encode("utf-8"))
        sink.close()
        ctx.info(f"wrote {sink.name}")
    return 0


def cmd_smoke(args, ctx: Ctx) -> int:
    import smoke

    return smoke.main()


def cmd_parity(args, ctx: Ctx) -> int:
    _quiet_libraries(ctx)
    import parity

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else None)
    report = parity.run(models, args.prompt or parity.DEFAULT_PROMPT)
    table = parity.format_report(report)
    print(table)
    if args.output:
        Path(args.output).write_text(report.to_json() + "\n")
    if args.markdown:
        Path(args.markdown).write_text(table + "\n")
    # a non-zero exit so this can gate a release: a deviation above
    # tolerance is a claim the README should not be making
    return 0 if report.passed else 1


def cmd_export_weights(args, ctx: Ctx) -> int:
    out = Path(args.output or f"{args.model.split('/')[-1]}.mwt")
    if out.exists() and not args.force:
        raise CLIError(f"{out} exists; pass -f/--force to overwrite")
    _quiet_libraries(ctx)
    _check_model(args.model)
    _silence_transformers(ctx)
    import mweights
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    tok = None
    if not args.no_tokenizer:
        tok = transformers.AutoTokenizer.from_pretrained(args.model)
    header = mweights.export_model(model, model.config, out,
                                   quant=args.quant, tokenizer=tok)
    cfg = header["config"]
    ctx.info(f"{out}  {out.stat().st_size / 1e6:.1f} MB  ({args.quant}); "
             f"{cfg['numLayers']} layers x {cfg['hiddenSize']}, vocab {cfg['vocabSize']}"
             f"{', tied embeddings' if cfg['tiedEmbeddings'] else ''}")
    return 0


def cmd_serve(args, ctx: Ctx) -> int:
    from serve import run_server

    run_server(port=args.port, model=args.model)
    return 0


def _app_path() -> str:
    import ui

    return str(Path(ui.__file__).resolve())


def cmd_app(args, ctx: Ctx) -> int:
    from streamlit.web import cli as st_cli

    sys.argv = ["streamlit", "run", _app_path()]
    return st_cli.main()


# --------------------------------------------------------------- parser
def _version_string() -> str:
    import provenance

    return f"mottled {provenance._version() or 'unknown'}"


def build_parser() -> argparse.ArgumentParser:
    # Global flags work before or after the command (`mottled -v capture` and
    # `mottled capture -v`): the subcommand copy defaults to SUPPRESS so it
    # never overwrites a value given before the command.
    def common(suppress: bool) -> argparse.ArgumentParser:
        d = argparse.SUPPRESS if suppress else None
        p = argparse.ArgumentParser(add_help=False)
        g = p.add_argument_group("output control")
        g.add_argument("-v", "--verbose", action="count",
                       default=d if suppress else 0,
                       help="status on stderr (-vv: library progress bars too)")
        g.add_argument("-q", "--quiet", action="store_true",
                       default=d if suppress else False,
                       help="errors only (no warnings)")
        g.add_argument("--debug", action="store_true",
                       default=d if suppress else False,
                       help="show tracebacks (also MOTTLED_DEBUG=1)")
        return p

    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="mottled", description=__doc__, epilog=EPILOG,
                                     formatter_class=fmt, parents=[common(False)])
    parser.add_argument("--version", action="version", version=_version_string())
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")
    sp = [common(True)]

    def add(name, help_, desc=None, epilog=None):
        return sub.add_parser(name, help=help_, description=desc or help_,
                              epilog=epilog, parents=sp, formatter_class=fmt)

    def out_args(p, default=None, what="the .mtj"):
        p.add_argument("-o", "--output", default=default, metavar="FILE",
                       help=f"write {what} here ('-' = stdout; default: "
                            f"{default or 'stdout when piped'})")
        p.add_argument("-f", "--force", action="store_true",
                       help="overwrite FILE if it exists; allow binary to a terminal")

    p = add("capture", "prompts -> trajectory .mtj, one container per prompt",
            epilog="examples:\n  mottled capture \"The cat sat\" -o run.mtj\n"
                   "  cat prompts.txt | mottled capture --model gpt2 | mottled project > s.mtj")
    p.add_argument("prompts", nargs="*", metavar="PROMPT",
                   help="prompts; '-' or none (with piped stdin) = one per stdin line")
    p.add_argument("--model", default="gpt2", help="HF model id or local directory (default: gpt2)")
    p.add_argument("--generate", type=int, default=0, metavar="N",
                   help="decode N tokens per prompt before capturing (default 0)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="decode temperature (0 = greedy; above 0 sampled with --seed)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16")
    p.add_argument("--no-logits", action="store_true",
                   help="omit logit-lens logits (smaller; scenes lose readout-change onsets)")
    p.add_argument("--full", action="store_true",
                   help="also keep the (V, D) embedding matrix and vocabulary")
    out_args(p)
    p.set_defaults(func=cmd_capture)

    p = add("project", "trajectory .mtj stream -> scene .mtj",
            epilog="examples:\n  mottled capture \"a\" \"b\" | mottled project > scene.mtj\n"
                   "  mottled project runs.mtj --projection umap --set grid_size=96 -o s.mtj")
    p.add_argument("inputs", nargs="*", metavar="FILE",
                   help="trajectory .mtj files or streams ('-' or none = stdin)")
    p.add_argument("--projection", choices=["pca", "umap"], default=None)
    p.add_argument("--density", choices=["kde", "knn"], default=None)
    p.add_argument("--seed", type=int, default=None, help="projection/density seed")
    p.add_argument("--grid-size", type=int, default=None, metavar="N")
    p.add_argument("--bootstrap", type=int, default=None, metavar="N",
                   help="density bootstrap resamples (0 = no uncertainty)")
    p.add_argument("--config", default=None, metavar="FILE.json",
                   help="MarbleConfig fields as JSON (an analysis record works too)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="set any MarbleConfig field; repeatable")
    out_args(p)
    p.set_defaults(func=cmd_project)

    p = add("inspect", "summarise a .mtj; --json, or --ndjson per state")
    p.add_argument("inputs", nargs="*", metavar="FILE", help="'-' or none = stdin")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true", help="one JSON array of summaries")
    g.add_argument("--ndjson", action="store_true",
                   help="one JSON object per (run, layer, token) state")
    p.set_defaults(func=cmd_inspect)

    p = add("validate", "check .mtj files against the schema (silent if valid)")
    p.add_argument("inputs", nargs="*", metavar="FILE", help="'-' or none = stdin")
    p.add_argument("--strict", action="store_true",
                   help="treat warnings (e.g. a missing schema id) as errors")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.set_defaults(func=cmd_validate)

    p = add("export", "prompts -> scene .mtj (= capture | project)")
    p.add_argument("prompts", nargs="*", metavar="PROMPT",
                   help="prompts; '-' or none (with piped stdin) = one per stdin line")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--generate", type=int, default=0, metavar="N",
                   help="decode N tokens per prompt before capturing (default 0)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="decode temperature (0 = greedy, seeded above 0)")
    p.add_argument("--manifest", default=None, metavar="PATH",
                   help="also write the analysis record as standalone JSON")
    p.add_argument("--models", default=None, metavar="A,B",
                   help="compare several models on ONE prompt, in readout space")
    out_args(p, default="scene.mtj")
    p.set_defaults(func=cmd_export)

    p = add("export-blast", "JSONL items -> blast scene .mtj, one pellet per prompt")
    p.add_argument("items", nargs="?", default=None,
                   help='JSONL, one item per line: {"id", "text" or "messages", '
                        '"labels": {name: 0/1/null}, optional "group"} (\'-\' or none = stdin)')
    p.add_argument("--model", default="gpt2")
    p.add_argument("--chat", action="store_true",
                   help="send text items through the model's chat template")
    p.add_argument("--layouts", default="monitor,open", metavar="A,B",
                   help="monitor (one per usable label) and/or open")
    p.add_argument("--seed", type=int, default=0, help="seed of the monitor's shuffle null")
    out_args(p, default="blast.mtj")
    p.set_defaults(func=cmd_export_blast)

    p = add("export-manifest", "print the analysis record a .mtj carries, as JSON")
    p.add_argument("scene", nargs="?", default=None, help="a .mtj file ('-' or none = stdin)")
    p.add_argument("-o", "--output", default=None, metavar="FILE", help="default: stdout")
    p.add_argument("-f", "--force", action="store_true", help="overwrite FILE")
    p.set_defaults(func=cmd_export_manifest)

    p = add("parity", "check captures against HF/TL/NNsight (exit 1 if off)")
    p.add_argument("--models", default=None, metavar="A,B")
    p.add_argument("--prompt", default=None)
    p.add_argument("-o", "--output", default=None, metavar="PATH",
                   help="write the machine-readable JSON report")
    p.add_argument("--markdown", default=None, metavar="PATH")
    p.set_defaults(func=cmd_parity)

    p = add("smoke", "check that this install works (exit 1 if not)")
    p.set_defaults(func=cmd_smoke)

    p = add("export-weights", "write a model as .mwt for in-browser inference")
    p.add_argument("model", help="HuggingFace model id or local path")
    p.add_argument("-o", "--output", default=None, help="default: <model name>.mwt")
    p.add_argument("-f", "--force", action="store_true", help="overwrite the output")
    p.add_argument("--quant", default="q8", choices=["q8", "f16", "f32"])
    p.add_argument("--no-tokenizer", action="store_true")
    p.set_defaults(func=cmd_export_weights)

    p = add("serve", "serve the web viewer + capture API on 127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="gpt2", help="model for /api captures")
    p.set_defaults(func=cmd_serve)

    p = add("app", "Streamlit explorer (what bare `mottled` runs)")
    p.set_defaults(func=cmd_app)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx = Ctx(verbosity=-1 if args.quiet else int(args.verbose or 0),
              debug=bool(args.debug or os.environ.get("MOTTLED_DEBUG")))
    func = getattr(args, "func", cmd_app)
    try:
        return func(args, ctx)
    except UsageError as exc:
        ctx.error(str(exc))
        print(f"usage: see `mottled {args.command or ''} --help`".replace("  ", " "),
              file=sys.stderr)
        return 2
    except CLIError as exc:
        ctx.error(str(exc))
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # the reader went away (`| head`): not an error worth a word
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return 141
    except Exception as exc:  # anything else: one line, unless debugging
        if ctx.debug:
            raise
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        ctx.error(f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__)
        print("mottled: (rerun with --debug or MOTTLED_DEBUG=1 for the traceback)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
