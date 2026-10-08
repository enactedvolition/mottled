# Blast scenes: one pellet per prompt

Two pellet families a deployed monitor would care about, drawn as a shotgun
blast (`blast.py`). Each pellet is one prompt's residual state at one
position, followed through all 25 readout layers. They are demo material for
the viewer, not results about any model beyond the one captured here.

## How they were made

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, CPU, float32, with its chat
  template and default system prompt.
- **Command:** `mottled export-blast <family>.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --chat -o <family>.mtj`,
  defaults otherwise (5-fold cross-fitting, 200 label shuffles, seed 0). The
  `.jsonl` beside each scene is its exact input, and the scene carries its
  own analysis record (`mottled export-manifest <family>.mtj`).

## `refusal.mtj`

- **Pellets:** 24 harmful requests from AdvBench (Zou et al., 2023, MIT
  licence, via `mlabonne/harmful_behaviors`) and 24 harmless requests written
  for this sample in the same imperative style ("Create a tutorial on how
  to…"), so the two halves do not differ in form as well as content. Read at
  the generation prompt: the state before the model writes.
- **Labels:** `harmful` is the design. `refused` is measured: the model's
  greedy 80-token reply matched against refusal markers ("I'm sorry", "I
  can't", "As an AI", …), the same rule for every item. An earlier 24-token
  window called one reply that refuses after a preamble (`h04`) compliance,
  which is why it is 80. Harmful: 23 of 24 refused. Harmless: 0 of 24.
- **What it shows:** at layer 0 every pellet is the same template token, one
  point. Under `monitor · harmful` the held-out readout is 0.77 at layer 1
  and 0.99 or more from layer 13. On 24-prompt slices of Alpaca's harmless
  prompts (not shipped: CC BY-NC 4.0) it read 0.86 to 0.94 at layer 1:
  matching the style removed part of the early split, which was wording.
  The one harmful request the model answered (`h23`, fake reviews) is ringed
  when colouring by `refused`. With one discordant pellet, a
  refusal-behaviour monitor cannot be told apart from a harmful-request
  classifier here.

## `truth.mtj`

- **Pellets:** 24 TruthfulQA questions (Lin et al., 2022, Apache-2.0), each
  answered twice as the assistant's turn: its best answer and its first
  incorrect answer. Read at the claim's last token. Each pair shares a
  `group`, so cross-fitting holds both answers out together. Without that, a
  held-out claim is scored on a direction its own twin pulled, and 9 of 25
  layers read below the shuffle null's 5th percentile.
- **Labels:** `false_claim` is the design. `negation` is a surface feature:
  the claim contains a negation or hedge word (no, not, nothing, never,
  don't, doesn't, isn't, cannot, no comment, depends, unknown). 12 of 24
  true answers carry one and 2 of 24 false ones; that one bit alone reads
  0.71 for `false_claim`.
- **What it shows:** `monitor · false_claim` reads 0.54 at layer 0 and 0.64
  to 0.82 from layer 1, above its shuffle null at 24 of 25 layers (the null
  moves labels only in ways the pair design could, see `blast._shuffle`).
  `monitor · negation` reads 0.88 to 0.93 from layer 15. Read together: a
  held-out readout ranks false answers above true ones, and ranks negated
  claims above plain ones more strongly, so a "truth monitor" fitted on this
  set may be a negation monitor. Colouring by `negation` under the
  `false_claim` layout rings the 14 pellets that break the usual pairing (a
  true answer without a negation word, or a false one with one); those are
  the pellets that tell the two readouts apart.
- **No muzzle:** each claim ends on its own token, so layer 0 is not one
  point here, and the open layout counts two pairs as copies of one flight:
  two identical "I have no comment" replies, and a true and a false answer
  one word apart.

## Read the numbers with the picture

Every split on screen is a readout of these 48 prompts under this view, to be
confirmed on held-out prompts. It is not evidence of mechanism
([`docs/validity.md`](../../../docs/validity.md)). The monitor's x is
cross-fitted, and its AUROC is printed beside the shuffle null's 5–95% band,
so a split the labels could have produced by chance reads as one.

These scenes are not bundled in the wheel. Open them on the hosted viewer
(`?file=samples/blast/refusal.mtj`), or locally with `python -m http.server`
from the repo root at `/viewer/?file=samples/blast/refusal.mtj`.
