"""Global configuration for Mottled.

A single dataclass carries every knob the pipeline understands so that
capture -> project -> density -> terrain -> render can be driven from one
object (and hashed for caching).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

DEFAULT_PROMPT = "The capital of France is"

# Model families the transformers backend has been exercised against share
# the Llama-style module layout (Qwen / Llama / Mistral / Gemma).
# Verified end-to-end (capture + logit lens + attention + exact residual
# decomposition). Qwen2.5-1.5B is the default *modern* reference: GQA, RoPE,
# SwiGLU, RMSNorm, 29 layers x 1536 — nothing like GPT-2's 2019 design, and
# the decomposition still reconciles to 0.0000. Llama-3.2 and Gemma-2 are
# license-gated on the Hub, so they need an accepted licence + HF_TOKEN and
# cannot back the bundled samples or CI. GPT-2 stays because the public
# trained-SAE ecosystem (SAELens, Neuronpedia) is overwhelmingly GPT-2-small,
# which is what M2's real-feature work depends on — not because it is the
# ceiling.
MODEL_CHOICES = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B",      # gated: needs an accepted licence
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-2-2b",            # gated: needs an accepted licence
]

PROJECTION_CHOICES = ["pca", "umap"]
DENSITY_CHOICES = ["kde", "knn"]
TRAJECTORY_MODES = ["all_tokens", "token", "mean", "cls"]


@dataclass
class MarbleConfig:
    # model / capture
    model: str = "gpt2"
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"
    keep_logits: bool = True      # keep full logit-lens logits (float16)
    capture_components: bool = True  # record attn/MLP residual writes; set
    # False for layouts without a single attn/mlp module per block (e.g. OPT)
    capture_attention: bool = True   # record head-averaged attention patterns
    # (forces the eager attention path on torch models)

    # generation (the decode axis): 0 keeps plain prompt capture; N > 0
    # decodes N tokens first and captures prompt + continuation, with the
    # per-step record in traj.meta["generation"].  temperature 0 = greedy;
    # sampling above 0 is seeded by `seed` for reproducible scenes.
    generate_tokens: int = 0
    generate_temperature: float = 0.0

    # SAE feature overlay (demo dictionary; see sae.py for real weights)
    sae_features: int = 256

    # projection
    projection: str = "pca"
    n_components: int = 2
    seed: int = 0

    # density / terrain
    density: str = "kde"
    density_bootstrap: int = 24   # bootstrap resamples behind Landscape.density_se
    # (the terrain confidence overlay); 0 disables uncertainty estimation
    grid_size: int = 64
    grid_padding: float = 0.2
    smooth_sigma: float = 1.5
    height_scale: float = 1.0
    invert_terrain: bool = False  # True: dense regions become valleys
    marble_lift: float = 0.04     # trajectory height offset above terrain

    # neighbors / predictions
    top_k: int = 5
    n_neighbors: int = 5
    neighbor_backend: str = "auto"  # "auto" | "faiss" | "numpy"

    # animation
    frames_per_layer: int = 4
    frame_ms: int = 120

    # trajectory extraction
    trajectory_mode: str = "all_tokens"
    trajectory_token: int = -1

    # caching
    cache_dir: str | None = None  # None: cache.default_dir("pipeline") —
    # $MOTTLED_CACHE_DIR, else the per-user cache home (never the cwd)
    use_cache: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = MarbleConfig()
