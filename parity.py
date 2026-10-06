"""Does Mottled see what the reference implementations see?

Mottled's own tests prove its pipeline is self-consistent. That is not the
question a reader of a paper built on it has. Theirs is whether the residual
stream in the picture is the one the model actually computed, and until now
the only answer was the README's word for it.

`mottled parity` answers it out loud: run the same prompt through Mottled and
through the implementations a reviewer already trusts, and report the largest
disagreement, per model, as a number they can read.

Two comparisons carry the claim, and they are different claims:

* **States** — Mottled's per-block capture against HuggingFace's own
  ``output_hidden_states``. These are the same tensors read two ways, so
  anything above float noise is a bug, not a tolerance.
* **Readout** — Mottled's deepest logit-lens output against the model's actual
  ``logits``. This one is not circular: if the lens at the last layer does not
  reproduce what the model itself predicts, every shallower readout in the
  explorer is measuring something other than what it claims to.

A third, `transformer_lens`, is comparable only against
``from_pretrained_no_processing``: TransformerLens by default folds layer-norms
and centers writing weights, which changes the residual stream on purpose. A
"disagreement" against a processed model would be a real difference reported as
an error, so this harness refuses to make that comparison rather than publish a
number that means nothing.

Nothing here runs in CI — the matrix downloads real weights. It is a command a
reader runs, and `tests/test_parity.py` pins the machinery offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

# The deviation above which a comparison is a finding rather than float noise.
# Both sides run the same kernels on the same weights, so the honest
# expectation is exact equality; this leaves room only for the reduction-order
# differences a chunked logit lens can introduce.
TOLERANCE = 1e-4

DEFAULT_PROMPT = "The capital of France is"
# One per layout family that capture.py resolves, smallest useful member of
# each: the matrix exists to show the *structural* resolution holds, not to
# rank models. Gated weights (Llama, Gemma) are named but not defaulted —
# they need an accepted licence, so a default run would fail for most readers.
DEFAULT_MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B-Instruct", "EleutherAI/pythia-160m"]


@dataclass
class Comparison:
    """One reference's verdict on one model, in numbers a reader can check."""

    model: str
    reference: str
    status: str = "ok"                  # "ok" | "skipped" | "error"
    detail: str = ""
    n_layers: int = 0
    n_tokens: int = 0
    state_max_abs: float | None = None   # largest |Mottled - reference| state
    state_max_rel: float | None = None   # ... over the reference state's norm
    logit_max_abs: float | None = None   # deepest lens vs the model's logits
    top1_agreement: float | None = None  # fraction of positions predicting alike

    @property
    def passed(self) -> bool:
        """True when nothing exceeded tolerance. A skip is not a pass."""
        if self.status != "ok":
            return False
        return all(v is None or v <= TOLERANCE
                   for v in (self.state_max_rel, self.logit_max_abs))


@dataclass
class Report:
    created: str
    prompt: str
    tolerance: float
    environment: dict
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        ran = [c for c in self.comparisons if c.status != "skipped"]
        return bool(ran) and all(c.passed for c in ran)

    def to_json(self) -> str:
        return json.dumps(
            {"created": self.created, "prompt": self.prompt,
             "tolerance": self.tolerance, "environment": self.environment,
             "comparisons": [asdict(c) for c in self.comparisons]},
            indent=2)


# ------------------------------------------------------------------ measures
def _deviation(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """(max absolute, max per-state relative) deviation between two stacks.

    Relative is per state rather than global because residual norms grow with
    depth: one global scale would hide a late-layer disagreement behind an
    early-layer magnitude.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        raise ValueError(f"shape mismatch: {got.shape} vs {want.shape}")
    abs_err = np.abs(got - want)
    norms = np.linalg.norm(want, axis=-1, keepdims=True)
    rel = abs_err / np.maximum(norms, 1e-12)
    return float(abs_err.max()), float(rel.max())


def against_transformers(model, tokenizer, prompt: str = DEFAULT_PROMPT,
                         name: str = "") -> Comparison:
    """Mottled's capture against HuggingFace's own `output_hidden_states`.

    Takes a loaded model so a caller can compare one they already have, and so
    this is testable without the hub.
    """
    import torch

    from capture import capture, logit_lens
    from models.families import resolve_family

    out = Comparison(model=name or _model_name(model), reference="transformers")
    traj = capture(model, prompt, tokenizer=tokenizer)
    adapter = resolve_family(model)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    ids = ids.to(next(model.parameters()).device)
    with torch.no_grad():
        ref = model(ids, output_hidden_states=True)

    hidden = torch.from_numpy(traj.hidden)
    n_blocks = len(adapter.blocks)
    reference = torch.stack([h[0] for h in ref.hidden_states]).float().cpu()
    # HF's tuple is (embeddings, block_1 … block_{N-1}, final_norm(block_N)):
    # every entry but the last is pre-norm and must match the capture as-is,
    # and the last matches only once the final norm Mottled defers is applied.
    with torch.no_grad():
        final = adapter.final_norm(hidden[-1]) if adapter.final_norm is not None \
            else hidden[-1]
    got = torch.cat([hidden[:n_blocks], final[None]]).numpy()
    out.state_max_abs, out.state_max_rel = _deviation(got, reference.numpy())

    # The readout claim. Recomputed in float32 rather than read off
    # traj.logits, which capture stores as float16 — that is a size choice
    # about the file, and charging its rounding to the lens would misattribute.
    with torch.no_grad():
        lens = logit_lens(hidden[-1:], adapter)[0].numpy()
        want = ref.logits[0].float().cpu().numpy()
    out.logit_max_abs = float(np.abs(lens.astype(np.float64)
                                     - want.astype(np.float64)).max())
    out.top1_agreement = float((lens.argmax(-1) == want.argmax(-1)).mean())
    out.n_layers, out.n_tokens = traj.n_layers, traj.n_tokens
    return out


def against_transformer_lens(name: str, prompt: str = DEFAULT_PROMPT) -> Comparison:
    """Mottled's capture against a TransformerLens run of the same weights.

    Uses ``from_pretrained_no_processing``: TL's default folds layer-norms and
    centers writing weights, which moves the residual stream deliberately. The
    comparison is only meaningful on the unprocessed model — see the module
    docstring.
    """
    out = Comparison(model=name, reference="transformer_lens")
    try:
        import transformer_lens as tl
    except ImportError:
        out.status, out.detail = "skipped", "transformer-lens is not installed"
        return out

    import torch

    from capture import capture, load_model
    from models.hooked import from_hooked_transformer

    # No BOS: TL prepends one for GPT-2 by default, HF does not, and parity
    # needs both sides to run the same tokens.
    ht = tl.HookedTransformer.from_pretrained_no_processing(
        name, default_prepend_bos=False)
    model, tokenizer = load_model(name)
    with torch.no_grad():
        theirs = from_hooked_transformer(ht, prompt)
    ours = capture(model, prompt, tokenizer=tokenizer)
    if ours.hidden.shape != theirs.hidden.shape:
        out.status = "error"
        out.detail = f"shape {ours.hidden.shape} vs {theirs.hidden.shape}"
        return out
    out.state_max_abs, out.state_max_rel = _deviation(ours.hidden, theirs.hidden)
    out.n_layers, out.n_tokens = ours.n_layers, ours.n_tokens
    return out


def against_nnsight(name: str, prompt: str = DEFAULT_PROMPT) -> Comparison:
    """Mottled's capture against states saved from an NNsight trace.

    NNsight wraps the same HuggingFace module, so this checks the *ingestion*
    path end to end: what `.save()` hands back, run through
    `models.external.from_hidden_states`, must be the states `capture()` read
    with its own hooks.
    """
    out = Comparison(model=name, reference="nnsight")
    try:
        import nnsight
    except ImportError:
        out.status, out.detail = "skipped", "nnsight is not installed"
        return out

    from capture import capture, load_model

    model, tokenizer = load_model(name)
    ours = capture(model, prompt, tokenizer=tokenizer)
    theirs = trace_with_nnsight(model, tokenizer, prompt, tokens=ours.tokens)
    if ours.hidden.shape != theirs.hidden.shape:
        out.status = "error"
        out.detail = f"shape {ours.hidden.shape} vs {theirs.hidden.shape}"
        return out
    out.state_max_abs, out.state_max_rel = _deviation(theirs.hidden, ours.hidden)
    out.n_layers, out.n_tokens = ours.n_layers, ours.n_tokens
    return out


def trace_with_nnsight(model, tokenizer, prompt: str, tokens=None):
    """One NNsight trace of `model`, ingested as a StateTrajectory.

    Mirrors Mottled's layer convention exactly: layer 0 is block 0's *input*,
    the tensor entering the residual stream, not the embedding module's output
    — those are the same for Llama-likes and different for GPT-2, which adds
    positional embeddings in between. Taking the block input is what
    `capture.HookCapture` does, and is family-agnostic.

    The states are appended in a loop rather than built by a comprehension:
    NNsight re-executes the trace body, and a name bound inside a comprehension
    there does not survive the block.
    """
    import nnsight

    from models.external import from_hidden_states
    from models.families import resolve_paths

    block_path, _ = resolve_paths(model)
    n_blocks = len(_walk(model, block_path))
    if tokenizer.pad_token is None:      # NNsight pads; GPT-2 has no pad token
        tokenizer.pad_token = tokenizer.eos_token
    lm = nnsight.LanguageModel(model, tokenizer=tokenizer)
    envoy = _walk(lm, block_path)

    saved = []
    with lm.trace(prompt):
        saved.append(envoy[0].input.save())
        for i in range(n_blocks):
            saved.append(envoy[i].output[0].save())

    if tokens is None:
        tokens = tokenizer(prompt)["input_ids"]
    return from_hidden_states(saved, tokens, model, tokenizer, prompt=prompt,
                              backend="nnsight")


def _walk(root, path: str):
    for part in path.split("."):
        root = getattr(root, part)
    return root


# -------------------------------------------------------------------- runner
def run(models: list[str] | None = None, prompt: str = DEFAULT_PROMPT,
        references: list[str] | None = None) -> Report:
    """Run the matrix. Each cell is isolated: one model's failure to download
    or one library's absence must not cost the reader the rest of the table."""
    import provenance

    models = list(models or DEFAULT_MODELS)
    references = list(references or ["transformers", "transformer_lens", "nnsight"])
    report = Report(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds")
                                          .replace("+00:00", "Z"),
        prompt=prompt, tolerance=TOLERANCE,
        environment=provenance.environment(),
    )
    for name in models:
        for reference in references:
            report.comparisons.append(_cell(name, reference, prompt))
    return report


def _cell(name: str, reference: str, prompt: str) -> Comparison:
    try:
        if reference == "transformers":
            from capture import load_model

            model, tokenizer = load_model(name)
            return against_transformers(model, tokenizer, prompt, name=name)
        if reference == "transformer_lens":
            return against_transformer_lens(name, prompt)
        if reference == "nnsight":
            return against_nnsight(name, prompt)
        return Comparison(model=name, reference=reference, status="error",
                          detail=f"unknown reference {reference!r}")
    except Exception as exc:  # a gated model, a missing weight, an OOM
        return Comparison(model=name, reference=reference, status="error",
                          detail=f"{type(exc).__name__}: {exc}")


def format_report(report: Report) -> str:
    """The table a reader inspects. Numbers, not a verdict badge."""
    lines = [
        "# Mottled parity report",
        "",
        f"- Generated: `{report.created}`",
        f"- Prompt: `{report.prompt}`",
        f"- Tolerance: `{report.tolerance}` (both sides run the same kernels on "
        "the same weights, so the honest expectation is exact equality)",
        f"- Python {report.environment.get('python')} on "
        f"{report.environment.get('platform')}",
        "",
        "| Model | Reference | States max abs | States max rel | Logits max abs "
        "| Top-1 agree | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report.comparisons:
        if c.status == "skipped":
            verdict = f"skipped — {c.detail}"
        elif c.status == "error":
            verdict = f"error — {c.detail}"
        else:
            verdict = "pass" if c.passed else "**OVER TOLERANCE**"
        lines.append(
            f"| `{c.model}` | {c.reference} | {_num(c.state_max_abs)} | "
            f"{_num(c.state_max_rel)} | {_num(c.logit_max_abs)} | "
            f"{_pct(c.top1_agreement)} | {verdict} |")
    lines += [
        "",
        "`states` compares Mottled's per-block capture with the reference's own "
        "residual stream; `logits` compares Mottled's deepest logit-lens readout "
        "with the model's actual output logits, which is the check that the lens "
        "measures what it claims to. A skipped row is not a passed row.",
    ]
    return "\n".join(lines)


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.2e}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _model_name(model) -> str:
    return getattr(getattr(model, "config", None), "name_or_path", "") \
        or type(model).__name__
