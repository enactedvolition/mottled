# Bundled sample scenes

Real captures, shipped so the viewer has something to open and so the offline
test suite (`tests/test_bundled_samples.py`) can run the analysis steps on real
model output with no download. No model weights are bundled; to capture new
scenes you bring your own model.

## Provenance

| Scenes | Source model | Model licence |
|---|---|---|
| `gpt2-capitals`, `gpt2-decode`, `gpt2-features`, `scene-abc`, `self-portrait`, `single` | `gpt2` (OpenAI) | MIT |
| `models-gpt2-distilgpt2` | `gpt2` and `distilgpt2` (Hugging Face) | MIT / Apache-2.0 |
| `models-qwen-gpt2`, `qwen-capitals` | `Qwen/Qwen2.5-1.5B-Instruct` and `gpt2` | Apache-2.0 / MIT |
| `blast/truth`, `blast/refusal`, `music-qa-honesty/*` | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 |

Prompt sources and labelling for the blast and music-qa-honesty sets are in
their own READMEs (`blast/` includes AdvBench prompts, MIT licence).
Each scene's `meta` records the model it was captured from.

Scenes store projected coordinates, entropies, attention and per-state
preservation, not raw hidden states. Tests that need hidden states still use
the tiny locally built models in `tests/tiny.py`.
