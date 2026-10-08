"""Console entry points: `mottled` launches the explorer.

    mottled                    # Streamlit explorer (default)
    mottled serve              # stdlib web server: viewer + capture API
    mottled export PROMPT ...  # capture prompts -> scene.mtj on stdout/file
    mottled export-blast ITEMS # one pellet per JSONL item -> blast scene .mtj
    mottled export-manifest S  # print the analysis record a .mtj carries
    mottled parity             # compare captures against the reference libraries
    mottled smoke              # does this install actually work?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _app_path() -> str:
    import ui

    return str(Path(ui.__file__).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mottled", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("app", help="launch the Streamlit explorer (default)")

    p_serve = sub.add_parser("serve", help="serve the web viewer + capture API")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", default="gpt2",
                         help="model for /api captures (default: gpt2)")

    p_export = sub.add_parser("export", help="capture prompts and write a .mtj scene")
    p_export.add_argument("prompts", nargs="+", help="one or more prompts")
    p_export.add_argument("-o", "--output", default="scene.mtj")
    p_export.add_argument("--model", default="gpt2")
    p_export.add_argument("--generate", type=int, default=0, metavar="N",
                          help="decode N tokens per prompt before capturing "
                               "(the decode axis; default 0 = prompt only)")
    p_export.add_argument("--temperature", type=float, default=0.0,
                          help="decode temperature (0 = greedy, seeded above 0)")
    p_export.add_argument("--manifest", default=None, metavar="PATH",
                          help="also write the analysis record as standalone "
                               "JSON, for a methods section or a "
                               "pre-registration (the scene always embeds it)")
    p_export.add_argument("--models", default=None, metavar="A,B",
                          help="compare several models on ONE prompt instead of "
                               "several prompts on one model. The scene is built "
                               "in readout space (the vocabulary the models share), "
                               "since they share no hidden space.")

    p_blast = sub.add_parser(
        "export-blast",
        help="capture one pellet per prompt and write a blast scene (.mtj)")
    p_blast.add_argument("items", help='JSONL, one item per line: {"id", "text" or '
                         '"messages", "labels": {name: 0/1/null}, optional '
                         '"group" (held out together, e.g. a contrast pair)}')
    p_blast.add_argument("-o", "--output", default="blast.mtj")
    p_blast.add_argument("--model", default="gpt2")
    p_blast.add_argument("--chat", action="store_true",
                         help="send text items through the model's chat template "
                              "(the pellet is then the state before it writes)")
    p_blast.add_argument("--layouts", default="monitor,open", metavar="A,B",
                         help="monitor (one per usable label) and/or open")
    p_blast.add_argument("--seed", type=int, default=0,
                         help="seed of the monitor's shuffle null")

    p_manifest = sub.add_parser(
        "export-manifest",
        help="print the analysis record embedded in a .mtj (config, "
             "environment, model and SAE identity)")
    p_manifest.add_argument("scene", help="a .mtj file")
    p_manifest.add_argument("-o", "--output", default=None,
                            help="default: stdout")

    p_parity = sub.add_parser(
        "parity",
        help="check Mottled's capture against HuggingFace / TransformerLens / "
             "NNsight on a model matrix and print the deviations")
    p_parity.add_argument("--models", default=None, metavar="A,B",
                          help="models to check (default: one per resolved "
                               "layout family)")
    p_parity.add_argument("--prompt", default=None)
    p_parity.add_argument("-o", "--output", default=None, metavar="PATH",
                          help="write the machine-readable JSON report")
    p_parity.add_argument("--markdown", default=None, metavar="PATH",
                          help="write the table as markdown")

    sub.add_parser("smoke",
                   help="check that this install works: flat API, viewer "
                        "assets, .mtj round-trip, analysis record")

    p_weights = sub.add_parser(
        "export-weights",
        help="write a model as .mwt so the web viewer can run it in-browser")
    p_weights.add_argument("model", help="HuggingFace model id or local path")
    p_weights.add_argument("-o", "--output", default=None,
                           help="default: <model name>.mwt")
    p_weights.add_argument("--quant", default="q8", choices=["q8", "f16", "f32"],
                           help="q8 = per-output-row int8 (default, smallest); "
                                "norms and the embedding table stay f32 either way")
    p_weights.add_argument("--no-tokenizer", action="store_true",
                           help="omit the id->piece table used to label states")

    args = parser.parse_args(argv)

    if args.command == "export-manifest":
        import json

        import statefile

        manifest, _ = statefile.read_container(args.scene)
        record = manifest.get("analysis")
        if record is None:
            print(f"{args.scene}: no analysis record — written either before "
                  "provenance export existed or by another producer",
                  file=sys.stderr)
            return 1
        text = json.dumps(record, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text + "\n")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(text)
        return 0

    if args.command == "smoke":
        import smoke

        return smoke.main()

    if args.command == "parity":
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

    if args.command == "export-weights":
        import mweights

        try:
            import transformers
        except ImportError:
            print("export-weights needs transformers: "
                  'pip install "mottled[models]"', file=sys.stderr)
            return 1

        out = Path(args.output or f"{args.model.split('/')[-1]}.mwt")
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
        tok = None
        if not args.no_tokenizer:
            tok = transformers.AutoTokenizer.from_pretrained(args.model)

        header = mweights.export_model(model, model.config, out,
                                       quant=args.quant, tokenizer=tok)
        size = out.stat().st_size
        cfg = header["config"]
        print(f"{out}  {size / 1e6:.1f} MB  ({args.quant})", file=sys.stderr)
        print(f"  {cfg['numLayers']} layers x {cfg['hiddenSize']}, "
              f"vocab {cfg['vocabSize']}"
              f"{', tied embeddings' if cfg['tiedEmbeddings'] else ''}",
              file=sys.stderr)
        return 0

    if args.command == "serve":
        from serve import run_server

        run_server(port=args.port, model=args.model)
        return 0

    if args.command == "export-blast":
        import json

        import statefile
        from blast import BlastConfig
        from pipeline import run_blast

        items = [json.loads(line) for line in
                 Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
        cfg = BlastConfig(model=args.model, chat=args.chat, seed=args.seed,
                          methods=tuple(m.strip() for m in args.layouts.split(",")
                                        if m.strip()))
        result = run_blast(items, args.model, cfg=cfg)
        statefile.save_scene(result, args.output)
        names = [b.name for b in result["blast"]["layouts"]]
        print(f"wrote {args.output}: {len(items)} pellets; layouts: {', '.join(names)}")
        for why in result["blast"]["skipped"]:
            print(f"  no monitor for {why}")
        if result["blast"]["spread"][0] > 1e-6:
            # no muzzle: the prompts already differ before any block runs, by
            # the read token or, with learned absolute positions (GPT-2), by
            # length, which every layer can then carry to the monitor
            print(f"  note: layer 0 is not one point (spread "
                  f"{result['blast']['spread'][0]:.3f}); the read token or prompt "
                  "length differs across items, and a monitor can read it")
        return 0

    if args.command == "export":
        import statefile
        from config import MarbleConfig
        from pipeline import attach_inspector, attach_manifest
        from ui import run_scene

        cfg = MarbleConfig(model=args.model, use_cache=False,
                           generate_tokens=args.generate,
                           generate_temperature=args.temperature)

        def _write(result):
            """Every exported scene carries its own analysis record."""
            attach_manifest(attach_inspector(result), cfg)
            statefile.save_scene(result, args.output)
            if args.manifest:
                import json

                Path(args.manifest).write_text(
                    json.dumps(result["analysis"], indent=2,
                               ensure_ascii=False) + "\n")

        if args.models:
            from pipeline import run_model_scene

            names = [m.strip() for m in args.models.split(",") if m.strip()]
            result = run_model_scene(cfg, args.prompts[0], names)
            _write(result)
            print(f"wrote {args.output}: {len(names)} models on "
                  f"{result['shared_vocab']} shared vocabulary entries")
            for name, cmp in zip(names[1:], result["model_comparisons"]):
                print(f"  {names[0]} vs {name}: final JS "
                      f"{cmp.final_divergence:.4f}, top-1 {cmp.top_a!r} vs {cmp.top_b!r}")
            if args.manifest:
                print(f"wrote {args.manifest}")
            return 0
        _write(run_scene(cfg, args.prompts))
        print(f"wrote {args.output}"
              + (f" and {args.manifest}" if args.manifest else ""))
        return 0

    from streamlit.web import cli as st_cli

    sys.argv = ["streamlit", "run", _app_path()]
    return st_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
