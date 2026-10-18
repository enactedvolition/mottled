"""Can a model learn to report a concept written into its own residual stream?

The design is pre-registered in experiments/README.md, which is the authority
on it; this file implements that plan. A concept vector is added to the
residual stream over a passage, and the model is asked which concept it was
(identification), which part of the passage it was in (localization), or —
the priming control — which listed word appears in the passage. The model is
scored from its own next-token logits over the answer labels, before and
after LoRA training on train concepts, and evaluated on held-out concepts
and held-out passages.

    python -m experiments.introspection --out results.json --summary summary.json

The experimenter writes the injected vector, so the ground truth is known by
construction and never comes from the model's own text. An injection shows
sufficiency under these conditions, not mechanism (docs/validity.md): a model
that names the injected concept is reporting something about its residual
stream, which is not by itself evidence about its unperturbed states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from capture import HookCapture

SCHEMA = "mottled-introspection/1"
SUMMARY_SCHEMA = "mottled-introspection-summary/1"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Hub commit of `MODEL` on 2026-10-18; a run refuses any other resolution.
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
HERE = Path(__file__).resolve().parent
CORPUS = HERE / "sotu_corpus.jsonl"
CORPUS_SHA256 = "0e00dc3a326d362094e8c142418edbf6c1a5a42421ed9ecb6b3a108605b6d9a2"

# Concrete nouns, chosen to be single ordinary words a reader can picture.
# The list is a design choice, not data: no label below comes from it, because
# the ground truth of every trial is the vector the experimenter wrote.
CONCEPTS = (
    "apple", "guitar", "volcano", "penguin", "umbrella", "candle", "dragon",
    "bicycle", "octopus", "chocolate", "telescope", "cactus", "violin",
    "glacier", "butterfly", "lighthouse", "pumpkin", "tornado", "banana",
    "castle", "rainbow", "spider", "piano", "desert", "mushroom", "rocket",
    "whale", "diamond", "snowman", "coffee", "camera", "island", "honey",
    "lemon", "feather", "kangaroo", "pyramid", "waterfall", "zebra",
    "crystal", "saxophone", "jungle", "cookie", "moon", "pirate", "robot",
    "garden", "thunder", "origami", "tiger",
)
N_TRAIN_CONCEPTS = 30
N_PARTS = 4
WORDS_PER_PART = 25
STRENGTHS = (0.25, 0.5, 1.0, 2.0, 4.0)   # multiples of the median state norm
CALIBRATION_TOPK = 10
CALIBRATION_HIT_RATE = 0.5

INTRO = f"Read this passage. It is split into {N_PARTS} numbered parts."
QUESTIONS = {
    "identify": ("While you read that passage, a concept may have been injected "
                 "directly into your internal activations. Which concept was it? "
                 "Answer with the letter only."),
    "localize": ("While you read that passage, a concept may have been injected "
                 "directly into your internal activations, in one of the "
                 f"{N_PARTS} parts. Which part was it injected into? "
                 "Answer with the part number only."),
    # No option is in the passage (items where one is are dropped), so a pick
    # of the injected concept here is priming, not reading the text.
    "prime": "Which of these words appears in the passage? Answer with the letter only.",
}
LETTERS = ("A", "B", "C", "D")
NUMBERS = ("1", "2", "3", "4")


def _h(*parts) -> int:
    """A seed from strings, so selection never depends on an RNG implementation."""
    key = "mottled-introspection-v1:" + ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def concept_split(concepts=CONCEPTS, n_train=N_TRAIN_CONCEPTS):
    """(train, test) by hash rank: fixed before any model has been run."""
    ranked = sorted(concepts, key=lambda c: _h("concept", c))
    return tuple(ranked[:n_train]), tuple(ranked[n_train:])


def load_corpus(path=CORPUS, check=True):
    data = Path(path).read_bytes()
    if check and hashlib.sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError(f"{path} is not the registered corpus")
    docs = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    docs.sort(key=lambda d: d["id"])
    return docs


def doc_split(docs):
    """Even rank -> train, odd -> test, as lens_patch_predictiveness splits."""
    return docs[0::2], docs[1::2]


def passage_parts(doc) -> tuple[str, ...]:
    words = doc["text"].split()
    need = N_PARTS * WORDS_PER_PART
    if len(words) < need:
        raise ValueError(f"doc {doc['id']} has {len(words)} words, needs {need}")
    return tuple(" ".join(words[i * WORDS_PER_PART:(i + 1) * WORDS_PER_PART])
                 for i in range(N_PARTS))


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}", text, re.IGNORECASE) is not None


@dataclass(frozen=True)
class Item:
    task: str                  # identify | localize | prime
    doc: str
    concept: str               # the concept whose vector is (or would be) written
    parts: tuple[str, ...]
    options: tuple[str, ...]   # identify/prime: four concepts; localize: ()
    target: int                # index of the correct label
    inject_parts: tuple[int, ...]


def make_items(task, docs, concepts):
    """Every (doc, concept) cell of one task; cells the passage would give
    away are skipped and counted, never re-drawn."""
    items, skipped = [], 0
    for doc in docs:
        parts = passage_parts(doc)
        text = " ".join(parts)
        for c in concepts:
            if _mentions(text, c):
                skipped += 1
                continue
            if task == "localize":
                t = _h("part", doc["id"], c) % N_PARTS
                items.append(Item(task, doc["id"], c, parts, (), t, (t,)))
                continue
            pool = sorted((o for o in concepts if o != c and not _mentions(text, o)),
                          key=lambda o: _h("distractor", task, doc["id"], c, o))
            if len(pool) < 3:
                skipped += 1
                continue
            t = _h("slot", task, doc["id"], c) % 4
            opts = list(pool[:3])
            opts.insert(t, c)
            items.append(Item(task, doc["id"], c, parts, tuple(opts), t,
                              tuple(range(N_PARTS))))
    return items, skipped


# --------------------------------------------------------------------------
# Rendering: exact token positions of every part
# --------------------------------------------------------------------------
def _render(tokenizer, user: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
    return user + "\n"


def render(tokenizer, item: Item, labels):
    """(input_ids (T,), part_masks (N_PARTS, T) bool) for one item.

    Masks come from character offsets in the rendered text, so a part is
    exactly the tokens that overlap its characters: re-tokenizing a part on
    its own can split differently at its edges and would mark other tokens.
    """
    lines = [f"[{i + 1}] {p}" for i, p in enumerate(item.parts)]
    user = INTRO + "\n\n" + "\n".join(lines) + "\n\n" + QUESTIONS[item.task]
    if item.options:
        user += "\n" + "\n".join(f"{labels[i]}) {o}" for i, o in enumerate(item.options))
    text = _render(tokenizer, user)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    offsets = enc["offset_mapping"]
    masks = torch.zeros((N_PARTS, len(ids)), dtype=torch.bool)
    cursor = 0
    for i, line in enumerate(lines):
        start = text.index(line, cursor) + len(f"[{i + 1}] ")
        end = start + len(item.parts[i])
        cursor = end
        for t, (a, b) in enumerate(offsets):
            if a < end and b > start:
                masks[i, t] = True
    if not masks.any(dim=1).all():
        raise RuntimeError("a passage part maps to no tokens")
    return ids, masks


def label_ids(tokenizer, labels) -> list[int]:
    out = []
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer label {lab!r} is {len(ids)} tokens")
        out.append(ids[0])
    if len(set(out)) != len(out):
        raise ValueError("answer labels share a token id")
    return out


# --------------------------------------------------------------------------
# The forward pass with a written vector
# --------------------------------------------------------------------------
def answer_logits(model, ids, mask, vector, layer, answer_ids, device):
    """Logits of the answer labels at the next position, with `vector` added
    to the state at `layer` over the positions in `mask` (no edit if None).

    The edit is a write-hook on the residual stream (capture.HookCapture), the
    same mechanism every Mottled intervention uses; gradients flow through it.
    """
    ids = ids.to(device)[None]
    edits = {}
    if vector is not None:
        m = mask.to(device)[None, :, None]
        v = vector.to(device)

        def edit(h):
            return h + m.to(h.dtype) * v.to(h.dtype)
        edits[layer] = edit
    with HookCapture(model, state_edits=edits):
        out = model(input_ids=ids, logits_to_keep=1)
    return out.logits[0, -1, answer_ids].float()


def injection_layer(model) -> int:
    """State index about two-thirds deep (Steering Awareness used ~2/3)."""
    from models.families import resolve_family
    return round(2 * resolve_family(model).n_layers / 3)


@torch.no_grad()
def state_at(model, ids, layer, device):
    with HookCapture(model) as hc:
        model(input_ids=ids.to(device)[None], logits_to_keep=1)
    return hc.states[layer][0].float().cpu()


@torch.no_grad()
def concept_directions(model, tokenizer, concepts, train_concepts, layer, device):
    """Unit vectors: the state at the last prompt position of "Tell me about
    {c}." minus the mean of that state over train concepts only, so no
    held-out concept shapes any training vector."""
    raw = {}
    for c in concepts:
        text = _render(tokenizer, f"Tell me about {c}.")
        ids = torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"])
        raw[c] = state_at(model, ids, layer, device)[-1]
    mean = torch.stack([raw[c] for c in train_concepts]).mean(0)
    diffs = {c: raw[c] - mean for c in concepts}
    flat = [c for c, d in diffs.items() if not float(d.norm()) > 0]
    if flat:
        # a zero difference normalises to NaN and would inject NaN silently
        raise ValueError(f"no direction for {flat}: the prompt does not distinguish them")
    return {c: d / d.norm() for c, d in diffs.items()}


@torch.no_grad()
def median_norm(model, tokenizer, items, layer, labels, device):
    """Median residual norm at `layer` over passage tokens of `items`.
    Passage tokens only: the first position of a sequence carries an outsized
    norm in many models and would set the scale for every injection."""
    norms = []
    for item in items:
        ids, masks = render(tokenizer, item, labels)
        h = state_at(model, ids, layer, device)
        norms.append(h[masks.any(0)].norm(dim=-1))
    return float(torch.cat(norms).median())


def _word_token_ids(tokenizer, word):
    out = set()
    for w in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            out.add(ids[0])
    return out


@torch.no_grad()
def calibrate_strength(model, tokenizer, directions, train_concepts, layer, scale, device):
    """The smallest strength in STRENGTHS at which writing a train concept over
    "Say one word." puts that word's first token in the model's top-10 next
    tokens for at least half the train concepts; the largest if none does.
    Uses train concepts only and a prompt outside the tasks."""
    user = "Say one word."
    text = _render(tokenizer, user)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor(enc["input_ids"])
    start = text.index(user)
    mask = torch.tensor([a < start + len(user) and b > start for a, b in enc["offset_mapping"]])
    table = {}
    for s in STRENGTHS:
        hits = 0
        for c in train_concepts:
            v = directions[c] * (s * scale)
            edits = {layer: lambda h, v=v: h + mask.to(h.device)[None, :, None].to(h.dtype) * v.to(h.device, h.dtype)}
            with HookCapture(model, state_edits=edits):
                logits = model(input_ids=ids.to(device)[None], logits_to_keep=1).logits[0, -1]
            top = set(torch.topk(logits.float(), CALIBRATION_TOPK).indices.tolist())
            hits += bool(top & _word_token_ids(tokenizer, c))
        table[s] = hits / len(train_concepts)
    chosen = next((s for s in STRENGTHS if table[s] >= CALIBRATION_HIT_RATE), STRENGTHS[-1])
    return chosen, table


# --------------------------------------------------------------------------
# Conditions, evaluation, training
# --------------------------------------------------------------------------
def condition_vector(cond, item, directions, magnitude):
    """None (no write), the item's concept, or a norm-matched random direction."""
    if cond == "none":
        return None
    if cond == "concept":
        return directions[item.concept] * magnitude
    if cond == "random":
        g = torch.Generator().manual_seed(_h("random", item.doc, item.concept) % 2**63)
        r = torch.randn(directions[item.concept].shape, generator=g)
        return r / r.norm() * magnitude
    raise ValueError(cond)


def _labels_for(task, letters, numbers):
    return numbers if task == "localize" else letters


@torch.no_grad()
def evaluate(model, tokenizer, items, cond, directions, magnitude, layer, device,
             letters=LETTERS, numbers=NUMBERS):
    model.eval()
    records = []
    for item in items:
        labels = _labels_for(item.task, letters, numbers)
        ids, masks = render(tokenizer, item, labels)
        mask = masks[list(item.inject_parts)].any(0)
        vec = condition_vector(cond, item, directions, magnitude)
        logits = answer_logits(model, ids, mask, vec, layer,
                               label_ids(tokenizer, labels), device)
        p = torch.softmax(logits, -1)
        records.append({"task": item.task, "cond": cond, "doc": item.doc,
                        "concept": item.concept, "target": item.target,
                        "pred": int(p.argmax()),
                        "p": [round(float(x), 6) for x in p]})
    return records


def train_lora(model, tokenizer, items, directions, magnitude, layer, device,
               steps_per_update=8, lr=1e-4, rank=16, seed=0,
               letters=LETTERS, numbers=NUMBERS, log=None):
    """LoRA on every attention and MLP projection, 4-way cross-entropy over
    the answer labels, one pass over `items` in a seeded order. Returns the
    model with adapters injected in place and the per-update mean losses."""
    from peft import LoraConfig, get_peft_model

    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in
                      {"q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"}})
    peft_model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
                                                  lora_dropout=0.0,
                                                  target_modules=targets))
    base = peft_model.get_base_model()   # same modules, adapters in place
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    order = np.random.default_rng(seed).permutation(len(items))
    base.train()
    losses, acc = [], 0.0
    opt.zero_grad()
    for step, i in enumerate(order, 1):
        item = items[i]
        labels = _labels_for(item.task, letters, numbers)
        ids, masks = render(tokenizer, item, labels)
        mask = masks[list(item.inject_parts)].any(0)
        logits = answer_logits(base, ids, mask, directions[item.concept] * magnitude,
                               layer, label_ids(tokenizer, labels), device)
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([item.target], device=logits.device))
        (loss / steps_per_update).backward()
        acc += float(loss.detach())
        if step % steps_per_update == 0 or step == len(order):
            opt.step()
            opt.zero_grad()
            n = steps_per_update if step % steps_per_update == 0 else step % steps_per_update
            losses.append(acc / n)
            acc = 0.0
            if log and len(losses) % 25 == 0:
                log(f"  update {len(losses)}: loss {losses[-1]:.4f}")
    base.eval()
    return base, losses


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def cell_matrix(records, concepts, docs, value):
    """(len(concepts), len(docs)) array of value(record); NaN where no trial."""
    ci = {c: i for i, c in enumerate(concepts)}
    di = {d: j for j, d in enumerate(docs)}
    m = np.full((len(concepts), len(docs)), np.nan)
    for r in records:
        m[ci[r["concept"]], di[r["doc"]]] = value(r)
    return m


def crossed_bootstrap(mats: dict, B=2000, seed=0):
    """Resample concepts and documents independently with replacement; every
    statistic uses the same draws, so differences between them are paired."""
    rng = np.random.default_rng(seed)
    nc, nd = next(iter(mats.values())).shape
    out = {k: np.empty(B) for k in mats}
    for b in range(B):
        wc = np.bincount(rng.integers(0, nc, nc), minlength=nc)
        wd = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        w = np.outer(wc, wd)
        for k, m in mats.items():
            ok = ~np.isnan(m)
            out[k][b] = (w * np.where(ok, m, 0)).sum() / max((w * ok).sum(), 1)
    return out


def _point(m):
    return float(np.nanmean(m))


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize(records, concepts, docs, B=2000):
    """Accuracies per stage/task/condition and the registered contrasts."""
    def mat(stage, task, cond, value):
        rs = [r for r in records if r["stage"] == stage and r["task"] == task and r["cond"] == cond]
        return cell_matrix(rs, concepts, docs, value) if rs else None

    correct = lambda r: float(r["pred"] == r["target"])
    mats, out = {}, {"accuracy": {}, "contrasts": {}}
    for stage in ("untrained", "trained"):
        for task, cond in (("identify", "concept"), ("identify", "none"),
                           ("localize", "concept"), ("localize", "random"),
                           ("localize", "none")):
            m = mat(stage, task, cond, correct)
            if m is not None:
                mats[f"{stage}/{task}/{cond}"] = m
        for cond in ("concept", "none"):
            # rate at which the injected item's concept is picked as "in the passage"
            m = mat(stage, "prime", cond, correct)
            if m is not None:
                mats[f"{stage}/prime/{cond}"] = m
    boot = crossed_bootstrap(mats, B=B)
    for k, m in mats.items():
        out["accuracy"][k] = {"point": _point(m), "ci95": _ci(boot[k]),
                              "n": int((~np.isnan(m)).sum())}

    def contrast(name, a, b):
        if a in mats and b in mats:
            d = boot[a] - boot[b]
            out["contrasts"][name] = {"point": _point(mats[a]) - _point(mats[b]),
                                      "ci95": _ci(d), "above_zero": bool(_ci(d)[0] > 0)}

    for stage in ("untrained", "trained"):
        contrast(f"{stage}: identify concept - identify none", f"{stage}/identify/concept", f"{stage}/identify/none")
        contrast(f"{stage}: identify concept - prime concept", f"{stage}/identify/concept", f"{stage}/prime/concept")
        contrast(f"{stage}: localize concept - localize random", f"{stage}/localize/concept", f"{stage}/localize/random")
    if "trained/identify/concept" in mats and "untrained/identify/concept" in mats:
        d = (boot["trained/identify/concept"] - boot["trained/prime/concept"]) - \
            (boot["untrained/identify/concept"] - boot["untrained/prime/concept"])
        out["contrasts"]["training gain beyond priming"] = {
            "point": (_point(mats["trained/identify/concept"]) - _point(mats["trained/prime/concept"]))
            - (_point(mats["untrained/identify/concept"]) - _point(mats["untrained/prime/concept"])),
            "ci95": _ci(d), "above_zero": bool(_ci(d)[0] > 0)}
    a = out["contrasts"].get("trained: identify concept - identify none")
    b = out["contrasts"].get("trained: identify concept - prime concept")
    out["H1"] = ("supported" if a and b and a["above_zero"] and b["above_zero"]
                 else "not supported")
    return out


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def load_model(name, revision, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision,
                                                 dtype=getattr(torch, dtype))
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is None and revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        # transformers 5.20 no longer sets _commit_hash; a full commit SHA is
        # immutable on the hub, so files fetched by it are that commit's
        resolved = revision
        model.config._commit_hash = revision
    if revision is not None and resolved != revision:
        raise ValueError(f"{name} at revision {revision} resolved to commit {resolved}")
    return model.to(device).eval(), tok


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _round(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def main(argv=None, loaded=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--docs", type=int, default=None, help="cap documents per split (tests)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    t0 = time.time()
    torch.manual_seed(0)

    letters, numbers = LETTERS, NUMBERS
    if loaded is None:
        model, tok = load_model(args.model, args.revision, args.device, args.dtype)
    else:
        model, tok, letters, numbers = loaded
        model.to(args.device)
    device = args.device

    docs = load_corpus(args.corpus, check=args.corpus == str(CORPUS))
    train_docs, test_docs = doc_split(docs)
    if args.docs:
        train_docs, test_docs = train_docs[:args.docs], test_docs[:args.docs]
    train_c, test_c = concept_split()
    layer = injection_layer(model)

    directions = concept_directions(model, tok, CONCEPTS, train_c, layer, device)
    # localize items need no distractors, so one concept gives one item per doc
    scale_items, _ = make_items("localize", train_docs[:10], train_c[:1])
    scale = median_norm(model, tok, scale_items, layer, numbers, device)
    strength, cal_table = calibrate_strength(model, tok, directions, train_c, layer, scale, device)
    magnitude = strength * scale
    log(f"layer {layer}, median norm {scale:.2f}, strength {strength} -> |v| = {magnitude:.2f}; calibration {cal_table}")

    test_items, skipped = {}, {}
    for task in ("identify", "localize", "prime"):
        test_items[task], skipped[f"test/{task}"] = make_items(task, test_docs, test_c)
    plan = [("identify", "concept"), ("identify", "none"), ("localize", "concept"),
            ("localize", "random"), ("localize", "none"), ("prime", "concept"), ("prime", "none")]

    records = []

    def run_eval(stage, m):
        for task, cond in plan:
            rs = evaluate(m, tok, test_items[task], cond, directions, magnitude, layer,
                          device, letters, numbers)
            for r in rs:
                r["stage"] = stage
            records.extend(rs)
            acc = np.mean([r["pred"] == r["target"] for r in rs]) if rs else float("nan")
            log(f"  {stage} {task}/{cond}: {acc:.3f} over {len(rs)}")

    log("untrained evaluation")
    run_eval("untrained", model)

    train_items = []
    for task in ("identify", "localize"):
        its, skipped[f"train/{task}"] = make_items(task, train_docs, train_c)
        train_items += its
    log(f"training on {len(train_items)} items")
    t_train = time.time()
    model, losses = train_lora(model, tok, train_items, directions, magnitude, layer, device,
                               letters=letters, numbers=numbers, log=log)
    t_train = time.time() - t_train

    log("trained evaluation")
    run_eval("trained", model)

    summary = summarize(records, list(test_c), [d["id"] for d in test_docs], B=args.bootstrap)
    import transformers
    provenance = {
        "git_commit": _git_commit(), "model": args.model,
        "model_revision_requested": args.revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "dtype": args.dtype, "device": device,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
    }
    config = {"layer": layer, "median_norm": scale, "strength": strength,
              "magnitude": magnitude, "calibration": {str(k): v for k, v in cal_table.items()},
              "train_concepts": list(train_c), "test_concepts": list(test_c),
              "n_train_docs": len(train_docs), "n_test_docs": len(test_docs),
              "n_train_items": len(train_items), "skipped": skipped,
              "lora": {"rank": 16, "alpha": 32, "lr": 1e-4, "steps_per_update": 8, "seed": 0}}
    summary.update({"schema": SUMMARY_SCHEMA, "config": config, "provenance": provenance,
                    "train_loss_first_last": [losses[0], losses[-1]] if losses else None})
    results = {"schema": SCHEMA, "config": config, "provenance": provenance,
               "train_losses": losses, "records": records,
               "timing": {"total_s": time.time() - t0, "train_s": t_train}}
    Path(args.out).write_text(json.dumps(_round(results)))
    Path(args.summary).write_text(json.dumps(_round(summary), sort_keys=True, indent=1))
    log(f"H1: {summary['H1']}")
    for name, c in summary["contrasts"].items():
        log(f"  {name}: {c['point']:+.3f} {c['ci95']}")
    for p in (args.out, args.summary):
        log(f"{p} sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()}")
    return summary


if __name__ == "__main__":
    main()
