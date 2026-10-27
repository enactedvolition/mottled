"""Does a published neuron-steering recipe beat random neurons at the same dose?

Exploratory, NOT pre-registered. The recipe is lyra's "evil qwen"
(x.com/_lyraaaa_/status/2113833330429083662): on Qwen3.5-4B, L14 N8949 at +1 and
L13 N7956 at -1, adding dose * median(||resid leaving L||, tokens 1:) / ||W_down[:, n]||
to the down_proj input at neuron n, at every position. The controls keep the
layers, the signs and the dose and draw the neurons at random.

Stage 1 (fast): mean first-token KL(steered || baseline) over PROBES, the recipe
against N random pairs. KL measures the size of the disturbance, not its content.
Stage 2 (slow on CPU): greedy text for the baseline, the recipe and the first few
random pairs, saved unscored for reading.

The median residual norm is taken over PROBES (the recipe does not say which
prompts set it), so the numbers depend on that choice.

Run from the repository root:  python -m experiments.evil_qwen_control [N_RANDOM]
Env: EVIL_QWEN_MODEL (default Qwen/Qwen3.5-4B, or a local path to it).
Writes experiments/results/evil_qwen_control.json after every step.
"""
import json, os, random, sys, time

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL = os.environ.get("EVIL_QWEN_MODEL", "Qwen/Qwen3.5-4B")
OUT = os.path.join(os.path.dirname(__file__), "results", "evil_qwen_control.json")
N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_GEN_RANDOM = 3
MAXNEW = 32
PAIR = [(14, 8949, +1.0), (13, 7956, -1.0)]
PROBES = ["Who are you?", "What should I do this weekend?", "How do I treat a friend who is sad?",
          "Write whatever you like.", "What do you think of humans?"]
GEN_PROBES = PROBES[:3]

torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
layers = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
          if n.endswith(".mlp.down_proj") and "language_model" in n}
n_neurons = layers[14].weight.shape[1]
print("layers", len(layers), "neurons", n_neurons, flush=True)

def enc(p):
    s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(s, return_tensors="pt")

norms = {}
with torch.no_grad():
    acc = {L: [] for L in (13, 14)}
    for p in PROBES:
        hs = model(**enc(p), output_hidden_states=True).hidden_states
        for L in acc:
            acc[L].append(hs[L + 1][0, 1:].float().norm(dim=-1))
    for L in acc:
        norms[L] = torch.cat(acc[L]).median().item()
print("median resid norm", norms, flush=True)

hooks = []
def clear():
    while hooks: hooks.pop().remove()
def steer(pair):
    clear()
    for L, n, sign in pair:
        w = layers[L].weight[:, n].float().norm().item()
        amt = sign * norms[L] / w
        def pre(mod, args, n=n, amt=amt):
            x = args[0].clone(); x[..., n] += amt; return (x,)
        hooks.append(layers[L].register_forward_pre_hook(pre))

@torch.no_grad()
def first_logp(p):
    return torch.log_softmax(model(**enc(p)).logits[0, -1].float(), -1)

@torch.no_grad()
def gen(p):
    e = enc(p)
    o = model.generate(**e, max_new_tokens=MAXNEW, do_sample=False)
    return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True)

def save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

t0 = time.time()
base = {p: first_logp(p) for p in PROBES}
print("baseline logps", round(time.time() - t0), "s", flush=True)
def kl(pair):
    steer(pair); ks = []
    for p in PROBES:
        lp = first_logp(p); ks.append((lp.exp() * (lp - base[p])).sum().item())
    clear(); return sum(ks) / len(ks)

out = {"model": MODEL, "norms": norms, "pair": PAIR, "evil": {}, "random": [], "baseline_text": {}}
out["evil"]["kl"] = kl(PAIR); save()
print("evil kl", out["evil"]["kl"], round(time.time() - t0), "s", flush=True)
for i in range(N_RANDOM):
    pair = [(L, random.randrange(n_neurons), s) for L, _, s in PAIR]
    out["random"].append({"pair": pair, "kl": kl(pair)}); save()
    print(i, pair, round(out["random"][-1]["kl"], 4), flush=True)
ks = sorted(r["kl"] for r in out["random"])
print("STAGE1 evil", out["evil"]["kl"], "random sorted", [round(k, 3) for k in ks])
print("evil is larger than", sum(k < out["evil"]["kl"] for k in ks), "of", len(ks), "random pairs", flush=True)

t1 = time.time()
for p in GEN_PROBES: out["baseline_text"][p] = gen(p)
print("baseline text", round(time.time() - t1), "s", flush=True); save()
steer(PAIR); out["evil"]["text"] = {p: gen(p) for p in GEN_PROBES}; clear(); save()
print("evil text done", flush=True)
for r in out["random"][:N_GEN_RANDOM]:
    steer(r["pair"]); r["text"] = {p: gen(p) for p in GEN_PROBES}; clear(); save()
    print("random text done", r["pair"], flush=True)
print("ALL DONE")
