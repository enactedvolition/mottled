"""The Mottled pipeline: capture -> project -> density -> terrain -> paths.

Pure functions over a `MarbleConfig`, with no Streamlit and no Plotly, so a
scene can be built from a notebook, a script, the CLI, or the capture server
exactly as the explorer builds it. `run_pipeline` is the single-prompt path;
`run_scene` joint-projects several prompts into one shared space so their
trajectories are comparable; `run_intervention` runs the counterfactual pair.

Split out of `ui.py` (which re-exports everything here, so
`from ui import run_pipeline` keeps working) — the module had grown to carry
the pipeline, two renderers and the app shell at once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import cache as cache_mod
import compare as compare_mod
import density as density_mod
import projection as projection_mod
import provenance as provenance_mod
import sae as sae_mod
import terrain as terrain_mod
import trajectory as trajectory_mod
from config import MarbleConfig
from trajectory import StateTrajectory

def capture(*args, **kwargs):
    """`capture.capture`, imported on first use: torch is a capture-time
    dependency, so a projection-only consumer (`mottled project`) never pays
    for importing it."""
    import capture as capture_mod

    return capture_mod.capture(*args, **kwargs)


def generate_and_capture(*args, **kwargs):
    """`capture.generate_and_capture`, imported on first use (see `capture`)."""
    import capture as capture_mod

    return capture_mod.generate_and_capture(*args, **kwargs)


# Config fields that cannot change a computed artifact: where the cache lives,
# whether it is used, and display-only knobs. Everything else in MarbleConfig
# is in the cache key *by construction* — a new field is keyed the day it is
# added, instead of the day someone notices a stale scene.
_UNKEYED_FIELDS = frozenset({"cache_dir", "use_cache", "frame_ms"})


def cache_key(tag: str, cfg: MarbleConfig, prompts, model=None, **extra) -> str:
    """The disk-cache key for a pipeline artifact.

    Complete by construction: every `MarbleConfig` field except
    `_UNKEYED_FIELDS` (so dtype, device, keep_logits, grid_padding,
    marble_lift … are all in it), plus the Mottled version and the model
    revision the weights resolve to, since either can move every number.
    """
    knobs = {k: v for k, v in cfg.to_dict().items() if k not in _UNKEYED_FIELDS}
    return cache_mod.make_key(tag, prompts, knobs,
                              mottled=provenance_mod._version(),
                              revision=model_revision(cfg.model, model), **extra)


def model_revision(name: str, model=None) -> str | None:
    """Best-effort identity of the weights `name` resolves to, without
    loading them: the hub commit (from an in-memory model's config, or the
    local hub cache's snapshot directory), or for a local checkpoint
    directory a fingerprint of its files' sizes and mtimes. None when it
    cannot be known offline (a hub model not yet downloaded)."""
    import hashlib
    from pathlib import Path

    if model is not None:
        commit = getattr(getattr(model, "config", None), "_commit_hash", None)
        if commit:
            return str(commit)
    path = Path(str(name)).expanduser()
    if path.is_dir():
        h = hashlib.sha256()
        for f in sorted(path.iterdir()):
            if f.is_file():
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
        return "local:" + h.hexdigest()[:16]
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(str(name), "config.json")
    except Exception:  # not installed, or not a repo id
        return None
    if isinstance(hit, str):
        parts = Path(hit).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return None


# Pipeline: capture -> project -> compute_density -> mesh -> trajectory
# --------------------------------------------------------------------------
def run_pipeline(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None,
                 input_ids=None) -> dict:
    """Execute the full Mottled pipeline and return every artifact.

    `model`/`tokenizer` may be pre-loaded objects (the UI caches them); when
    omitted, `cfg.model` is loaded by name. `input_ids`, when the prompt's
    exact token ids are known (a chat template's own tokenization), are
    captured as given; `prompt` then only labels the run.
    """
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    # the same text can stand for different ids, so ids are part of the key;
    # passed only when given, so every other key is what it was
    ids = {} if input_ids is None else {"input_ids": [int(i) for i in input_ids]}
    key = cache_key("pipeline-v7", cfg, prompt, model=model, **ids)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    traj = _capture_with(cfg, prompt, model=model, tokenizer=tokenizer,
                         input_ids=input_ids)

    coords, projector = projection_mod.project(
        traj.hidden, method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed
    )
    quality = projection_mod.projection_quality(traj.hidden, coords, projector)
    landscape = density_mod.compute_density(
        coords, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding,
        bootstrap=cfg.density_bootstrap, seed=cfg.seed,
    )
    surface = terrain_mod.mesh(
        landscape, smooth_sigma=cfg.smooth_sigma,
        height_scale=cfg.height_scale, invert=cfg.invert_terrain,
    )

    flat = trajectory_mod.extract(coords, traj.tokens, mode=cfg.trajectory_mode,
                                  token=cfg.trajectory_token)
    trajectories = [
        replace(t, points=terrain_mod.drape(surface, t.points, lift=cfg.marble_lift))
        for t in flat
    ]
    fine_paths = [trajectory_mod.densify(t.points, cfg.frames_per_layer) for t in trajectories]

    result = {
        "prompt": prompt,
        "traj": traj,
        "coords": coords,
        "projector": projector,
        "quality": quality,
        "landscape": landscape,
        "mesh": surface,
        "trajectories": trajectories,
        "fine_paths": fine_paths,
    }
    if disk is not None:
        disk.put(key, result)
    return result


def _capture_with(cfg: MarbleConfig, prompt: str, model=None, tokenizer=None,
                  input_ids=None) -> StateTrajectory:
    """One validated capture under the config's capture knobs.

    With `cfg.generate_tokens > 0` the capture decodes that many tokens
    first (greedy, or seeded sampling at `cfg.generate_temperature`) and
    returns the trajectory of prompt + continuation, decode record in
    `meta["generation"]`."""
    kwargs = dict(
        tokenizer=tokenizer,
        top_k=cfg.top_k,
        device=cfg.device,
        dtype=cfg.dtype,
        keep_logits=cfg.keep_logits,
        capture_components=cfg.capture_components,
        capture_attention=cfg.capture_attention,
        input_ids=input_ids,
    )
    target = model if model is not None else cfg.model
    if cfg.generate_tokens > 0:
        traj = generate_and_capture(target, prompt,
                                    max_new_tokens=cfg.generate_tokens,
                                    temperature=cfg.generate_temperature,
                                    seed=cfg.seed, **kwargs)
    else:
        traj = capture(target, prompt, **kwargs)
    traj.validate()
    return traj


def chat_input(tokenizer, messages: list[dict]) -> tuple[str, list[int]]:
    """A conversation as the model is sent it, ready for its reply: the text,
    and the token ids to capture.

    With a chat template the ids are the tokenizer's own chat tokenization,
    to be captured as they are. Tokenizing the rendered text again would add
    the tokenizer's special tokens to ones the template already wrote (a
    doubled BOS is the common case): a different question to the model that
    nothing downstream would notice. Without a template (GPT-2) the
    conversation is a plain transcript, tokenized as any prompt is.
    """
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True)
        if hasattr(ids, "keys"):            # transformers 5 returns an encoding
            ids = ids["input_ids"]
        return text, [int(i) for i in ids]
    turns = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    text = "\n".join(turns + ["Assistant:"])
    return text, tokenizer(text)["input_ids"]


def chat_reply(tokenizer, traj: StateTrajectory) -> str:
    """The reply a chat capture generated, decoded from the token ids the
    decode chose (not re-joined from their display strings)."""
    steps = (traj.meta.get("generation") or {}).get("steps", [])
    return tokenizer.decode([s["id"] for s in steps], skip_special_tokens=True).strip()


def _assemble_scene(cfg: MarbleConfig, trajs: list[StateTrajectory]) -> dict:
    """Shared multi-run assembly: joint projection, one terrain from the
    union of all runs' states, draped trajectories, comparisons vs run 0."""
    coords_list, projector = projection_mod.project_joint(
        [t.hidden for t in trajs],
        method=cfg.projection, n_components=cfg.n_components, seed=cfg.seed,
    )
    quality_list = [
        projection_mod.projection_quality(t.hidden, c, projector)
        for t, c in zip(trajs, coords_list)
    ]
    union = np.concatenate([c.reshape(-1, cfg.n_components) for c in coords_list])
    landscape = density_mod.compute_density(
        union, method=cfg.density, grid_size=cfg.grid_size, padding=cfg.grid_padding,
        bootstrap=cfg.density_bootstrap, seed=cfg.seed,
    )
    surface = terrain_mod.mesh(
        landscape, smooth_sigma=cfg.smooth_sigma,
        height_scale=cfg.height_scale, invert=cfg.invert_terrain,
    )

    trajectories_list, fine_paths_list = [], []
    for traj, coords in zip(trajs, coords_list):
        flat = trajectory_mod.extract(coords, traj.tokens, mode=cfg.trajectory_mode,
                                      token=cfg.trajectory_token)
        trajectories = [
            replace(t, points=terrain_mod.drape(surface, t.points, lift=cfg.marble_lift))
            for t in flat
        ]
        trajectories_list.append(trajectories)
        fine_paths_list.append(
            [trajectory_mod.densify(t.points, cfg.frames_per_layer) for t in trajectories])

    # `compare` is a layer-for-layer measurement, so it is only defined when
    # the runs have the same depth. Prompts through one model always do;
    # different *models* generally do not (GPT-2's 13 layers vs
    # DistilGPT-2's 7), and there the depth-normalised
    # `crossmodel.compare_models` is the right instrument instead — so the
    # pairwise table is simply absent rather than fabricated.
    comparisons = []
    for t, c in zip(trajs[1:], coords_list[1:]):
        if t.n_layers == trajs[0].n_layers and t.dim == trajs[0].dim:
            comparisons.append(compare_mod.compare(trajs[0], t, coords_list[0], c))

    result = {
        "trajs": trajs,
        "coords_list": coords_list,
        "projector": projector,
        "quality_list": quality_list,
        "landscape": landscape,
        "mesh": surface,
        "trajectories_list": trajectories_list,
        "fine_paths_list": fine_paths_list,
        "comparisons": comparisons,
        # run-0 view (the run_pipeline keys)
        "traj": trajs[0],
        "coords": coords_list[0],
        "quality": quality_list[0],
        "trajectories": trajectories_list[0],
        "fine_paths": fine_paths_list[0],
    }
    if len(trajs) == 2:  # the run_compare aliases
        result.update({
            "traj_b": trajs[1],
            "coords_b": coords_list[1],
            "trajectories_b": trajectories_list[1],
            "fine_paths_b": fine_paths_list[1],
        })
        if comparisons:  # absent when the two runs' depths differ (see above)
            result["comparison"] = comparisons[0]
    return result


def run_scene(cfg: MarbleConfig, prompts: list[str], model=None, tokenizer=None) -> dict:
    """Multi-prompt scene: capture every prompt, joint-project into ONE
    shared space, build one terrain from the union of all states, and
    compare each run against the first.

    Returns per-run lists ("trajs", "coords_list", "trajectories_list",
    "fine_paths_list", "comparisons") plus the `run_pipeline` keys for run 0
    and, for exactly two prompts, the `run_compare` aliases.
    """
    if not prompts:
        raise ValueError("run_scene needs at least one prompt")
    disk = cache_mod.DiskCache(cfg.cache_dir) if cfg.use_cache else None
    key = cache_key("scene-v5", cfg, list(prompts), model=model)
    if disk is not None and (hit := disk.get(key)) is not None:
        return hit

    trajs = [_capture_with(cfg, p, model=model, tokenizer=tokenizer) for p in prompts]
    result = {"prompts": list(prompts), "prompt": prompts[0], **_assemble_scene(cfg, trajs)}
    if len(prompts) == 2:
        result["prompt_b"] = prompts[1]
    if disk is not None:
        disk.put(key, result)
    return result


def capture_trajectories(cfg: MarbleConfig, prompts: list[str], model=None,
                         tokenizer=None) -> list[StateTrajectory]:
    """The capture stage alone: one validated trajectory per prompt.

    `mottled capture` writes these; `project_trajectories` turns them into a
    scene. `run_scene` is the two composed (plus the cache)."""
    if not prompts:
        raise ValueError("capture needs at least one prompt")
    if model is None:
        from capture import load_model

        model, tokenizer = load_model(cfg.model, device=cfg.device, dtype=cfg.dtype)
    return [_capture_with(cfg, p, model=model, tokenizer=tokenizer) for p in prompts]


def project_trajectories(cfg: MarbleConfig, trajs: list[StateTrajectory],
                         inspectors: list | None = None,
                         prompts: list[str] | None = None) -> dict:
    """The project stage alone: trajectories -> an exportable scene result.

    Joint projection, terrain and comparisons (`_assemble_scene`), the
    inspector layers (precomputed ones from `inspectors` are used as given;
    the rest are computed from the trajectory's embedding matrix), and the
    analysis record. `mottled capture | mottled project` and `mottled
    export` both end here, which is what makes them the same scene.
    """
    if not trajs:
        raise ValueError("project needs at least one trajectory")
    if prompts is None:
        prompts = [str(t.meta.get("prompt", "")) for t in trajs]
    result = {"prompts": list(prompts), "prompt": prompts[0], **_assemble_scene(cfg, trajs)}
    if len(prompts) == 2:
        result["prompt_b"] = prompts[1]
    given = list(inspectors or [])
    result["inspector_list"] = [
        given[i] if i < len(given) and given[i] is not None else inspector_entry(t)
        for i, t in enumerate(trajs)
    ]
    return attach_manifest(result, cfg)


def degraded_note(meta: dict) -> str | None:
    """The banner a producer that could not see inside the model must carry.

    Returns None for a full capture. For a degraded one (e.g. the API-logprob
    producer) it names the backend, lists what that backend cannot observe,
    says which axis is actually animated, and flags a truncated — therefore
    under-counted — entropy. Pure, so the wording is testable without
    driving the app.
    """
    if not meta.get("degraded"):
        return None
    absent = ", ".join(meta.get("absent") or []) or "internal state"
    note = (f"**Degraded capture** — this run came from `{meta.get('backend')}`, "
            f"which cannot observe: {absent}. The animated axis is "
            f"**{meta.get('axis', 'decode')}**, not depth, and the geometry is "
            f"the model's *output distribution*, not its residual stream.")
    if meta.get("entropy_is_lower_bound"):
        note += (" Reported entropy is a **lower bound** — the API truncates "
                 "its distribution, so the true spread is at least this wide.")
    return note


def attach_features(result: dict, sae, source: str | None = None,
                    hook: str | None = None) -> dict:
    """Attach an SAE feature layer to a pipeline/scene result, for export.

    Per run: the dominant feature per state (top-1 id + activation) and the
    dictionary's **measured fit** (`sae.fit_report`), so a viewer never shows
    a feature without its calibration attached. `source`/`hook` say where the
    dictionary came from. Mutates and returns `result` (adds
    "features_list", aligned with the runs); `statefile.save_scene` carries
    it into the `.mtj` additively.
    """
    trajs = result.get("trajs") or [result["traj"]]
    feats = []
    for traj in trajs:
        acts = sae.encode(traj.hidden)                      # (L, T, F)
        fit = sae_mod.fit_report(sae, traj)
        feats.append({
            "source": source, "hook": hook,
            "best_layer": fit.best_layer,
            "recon_error": fit.recon_error,
            "top_id": acts.argmax(axis=-1).astype(np.int32),
            "top_act": acts.max(axis=-1).astype(np.float32),
        })
    result["features_list"] = feats
    return result


def attach_inspector(result: dict, n_neighbors: int = 5) -> dict:
    """Precompute the inspector layers a scene file cannot recover on its own.

    The explorer can show representation-space neighbors and the attention/MLP split
    because it still holds the embedding matrix (V x D) and the residual
    components (2 x (L-1) x T x D) in memory. A scene file carries neither —
    they are orders of magnitude larger than everything else in it — so the
    web viewer has been the lesser surface. Both reduce to something tiny if
    they are resolved *before* export:

    * neighbors  -> the k nearest vocabulary tokens per state, as indices into
                    a compact table of just the strings that actually appear
    * components -> the attention/MLP share per state, which is two numbers,
                    not two D-dimensional vectors

    Mutates and returns `result` (adds "inspector_list", aligned with the
    runs); `statefile.save_scene` carries it into the `.mtj` additively.
    Runs lacking the source data are skipped individually rather than
    failing the export.
    """
    trajs = result.get("trajs") or [result["traj"]]
    result["inspector_list"] = [inspector_entry(t, n_neighbors) for t in trajs]
    return result


def inspector_entry(traj: StateTrajectory, n_neighbors: int = 5) -> dict:
    """One run's inspector layers (see `attach_inspector`): nearest
    vocabulary tokens per state and the attn/MLP share per state, each
    present only when the trajectory carries its source data."""
    from neighbors import TokenNeighbors

    entry: dict = {}

    if traj.embedding_matrix is not None and traj.vocab:
        tn = TokenNeighbors(traj.embedding_matrix, traj.vocab)
        flat = traj.hidden.reshape(-1, traj.dim)
        table: dict[str, int] = {}
        idx = np.zeros((len(flat), n_neighbors), dtype=np.int32)
        sim = np.zeros((len(flat), n_neighbors), dtype=np.float32)
        for i, vec in enumerate(flat):
            for j, (tok, s) in enumerate(tn.nearest(vec, k=n_neighbors)):
                idx[i, j] = table.setdefault(tok, len(table))
                sim[i, j] = s
        shape = (traj.n_layers, traj.n_tokens, n_neighbors)
        entry["neighbor_tokens"] = list(table)
        entry["neighbor_idx"] = idx.reshape(shape)
        entry["neighbor_sim"] = sim.reshape(shape)

    comps = traj.components
    if comps is not None and {"attn", "mlp"} <= set(comps):
        a = np.linalg.norm(comps["attn"].astype(np.float64), axis=-1)
        m = np.linalg.norm(comps["mlp"].astype(np.float64), axis=-1)
        total = np.maximum(a + m, 1e-12)
        entry["component_shares"] = np.stack(
            [a / total, m / total], axis=-1).astype(np.float32)

    return entry


def attach_manifest(result: dict, cfg: MarbleConfig, sae=None,
                    sae_source: str | None = None,
                    sae_hook: str | None = None) -> dict:
    """Attach the analysis record — the scene's own methods section.

    `docs/validity.md` asks a user publishing on Mottled output to version-lock
    the model, tokenizer, library versions, precision, seeds and SAE artifact
    hashes. This puts all of it in the file instead of in the user's notes:
    `provenance.record` over the config that drove the run and the metas of the
    trajectories it produced. Mutates and returns `result` (adds "analysis");
    `statefile.save_scene` carries it into the `.mtj` additively.
    """
    result["analysis"] = provenance_mod.record(
        # a producer that ran under a different config than it was handed
        # (run_intervention: the prompt pass only) puts it on the result, and
        # the record states what ran, not what the session asked for
        result.get("config", cfg),
        prompts=result.get("prompts") or [result.get("prompt", "")],
        trajs=result.get("trajs") or [result["traj"]],
        sae=sae, sae_source=sae_source, sae_hook=sae_hook,
    )
    return result


def run_model_scene(cfg: MarbleConfig, prompt: str, models: list,
                    loaded: dict | None = None) -> dict:
    """One prompt, several **models**, on one terrain.

    Different models share no hidden space — different widths, different
    depths, different tokenizers — so the scene is assembled in *readout
    space* (`crossmodel.readout_space`): every state becomes the next-token
    distribution it predicts over the vocabulary all the models share, plus a
    visible bucket for the mass spent outside it. That is a real shared
    coordinate system, so joint projection, the terrain, and every viewer
    work unchanged.

    `models` are hub names; `loaded` optionally maps a name to a
    preloaded (model, tokenizer) pair. The result carries the usual scene
    keys plus `model_names`, `shared_vocab`, and `model_comparisons` (each
    model measured against the first).

    Readout space says what the models *do*. For how they *represent* —
    which layer of B matches layer l of A — use `crossmodel.layer_similarity`,
    which needs no shared space but does need matching tokenization.
    """
    import crossmodel

    if not models:
        raise ValueError("run_model_scene needs at least one model")
    loaded = loaded or {}
    raw = []
    for name in models:
        model, tokenizer = loaded.get(name, (None, None))
        raw.append(_capture_with(replace(cfg, model=name), prompt,
                                 model=model, tokenizer=tokenizer))

    trajs, vocab = crossmodel.readout_space(raw)
    for traj, name in zip(trajs, models):
        traj.meta["model"] = name
    result = {"prompts": [f"{m}: {prompt}" for m in models], "prompt": prompt,
              "model_names": list(models), "shared_vocab": len(vocab),
              "source_trajs": raw, "space": "readout",
              **_assemble_scene(cfg, trajs)}
    result["model_comparisons"] = [crossmodel.compare_models(raw[0], other)
                                   for other in raw[1:]]
    return result


def run_compare(cfg: MarbleConfig, prompt_a: str, prompt_b: str,
                model=None, tokenizer=None) -> dict:
    """A/B pipeline: a two-prompt `run_scene` (kept as the pairwise API)."""
    return run_scene(cfg, [prompt_a, prompt_b], model=model, tokenizer=tokenizer)


def run_intervention(cfg: MarbleConfig, prompt: str, interventions: list,
                     model, tokenizer, target_id: int | None = None) -> dict:
    """Interactive patching: baseline capture vs perturb-and-replay branch.

    Runs the same prompt twice — once untouched, once under `interventions`
    (intervene.Intervention edits) — then assembles the two counterfactual
    runs as a scene: shared projection, one terrain, comparison, plus an
    `intervene.divergence` readout.  Requires a torch model; results are not
    cached (the edit space is unbounded).

    When `target_id` is given for a single directional steer, also attaches a
    `Faithfulness` readout: the run measures a norm-matched *random* control so
    the UI can say how much of the effect is the steering direction rather than
    the perturbation's raw size. When the baseline additionally carries the
    residual decomposition (`cfg.capture_components`), a `PersistenceProfile`
    is attached too — the same faithfulness effect read at every layer from
    the edit down, paired with the attention/MLP write shares.
    """
    from intervene import (divergence, intervene, persistence_profile,
                           score_against_control)

    # The edit replays the prompt pass only, so the baseline has to be that
    # same pass: with generation on it would span prompt + continuation and
    # could not be compared state-for-state with the branch. The temperature
    # goes too: nothing was sampled, and this is the config the record states.
    prompt_cfg = replace(cfg, generate_tokens=0, generate_temperature=0.0)
    baseline = _capture_with(prompt_cfg, prompt, model=model, tokenizer=tokenizer)
    # Attention capture moves a pass onto the eager kernel, so every pass
    # measured against the baseline has to share it: across kernels, rounding
    # alone reads as a separation, even for an edit that changed nothing.
    # Components too: the config records them for the scene, not one run.
    branch = intervene(model, prompt, interventions, tokenizer=tokenizer,
                       top_k=cfg.top_k, device=cfg.device, dtype=cfg.dtype,
                       keep_logits=cfg.keep_logits,
                       capture_attention=cfg.capture_attention,
                       capture_components=cfg.capture_components)
    branch.validate()

    result = {"prompts": [prompt, prompt], "prompt": prompt,
              "prompt_b": "patched: " + ", ".join(iv.describe() for iv in interventions),
              # attach_manifest records this in place of the caller's config,
              # which may ask for a decode that neither run did
              "config": prompt_cfg,
              **_assemble_scene(cfg, [baseline, branch])}
    result["divergence"] = divergence(baseline, branch)

    if (target_id is not None and baseline.logits is not None
            and len(interventions) == 1 and interventions[0].kind == "perturb"
            and interventions[0].vector is not None):
        iv = interventions[0]
        tok = iv.token if iv.token is not None else -1
        # steered everywhere -> measured everywhere (token=None), not at -1
        result["faithfulness"] = score_against_control(
            model, prompt, baseline, branch, iv.vector, iv.layer, int(target_id),
            token=iv.token, tokenizer=tokenizer, seed=cfg.seed, device=cfg.device,
            dtype=cfg.dtype, top_k=cfg.top_k,
            capture_attention=cfg.capture_attention)
        if (baseline.components is not None
                and {"attn", "mlp"} <= set(baseline.components)):
            result["persistence"] = persistence_profile(
                model, prompt, iv.vector, iv.layer, int(target_id),
                tokenizer=tokenizer, token=tok,
                scale=float(np.linalg.norm(iv.vector)),
                baseline=baseline, branch=branch, seed=cfg.seed,
                device=cfg.device, dtype=cfg.dtype, top_k=cfg.top_k,
                capture_attention=cfg.capture_attention)
    return result



def _blast_input(tokenizer, item: dict, chat: bool) -> tuple[str, list[int] | None]:
    """An item's text and, when a chat template is involved, its exact ids.

    Ending on a user turn, the ids end on the generation prompt: the state the
    model is in before it writes. Ending on an assistant turn, they end on the
    reply's last token (the template continues that turn instead of opening
    a new one), which is where a claim the model has made can be read."""
    msgs = item.get("messages")
    if msgs is None:
        if not chat:
            return item["text"], None
        msgs = [{"role": "user", "content": item["text"]}]
    text = item.get("text") or msgs[-1]["content"]
    if msgs[-1]["role"] != "assistant":
        return text, chat_input(tokenizer, msgs)[1]
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"item {item.get('id')!r} ends on an assistant turn, which "
                         "needs the tokenizer's chat template to continue it")
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                        continue_final_message=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return text, [int(i) for i in ids]


def run_blast(items: list[dict], model, tokenizer=None, cfg=None) -> dict:
    """One pellet per item, every layout its labels support, ready for
    `statefile.save_scene`.

    Each item is {"id", "text" or "messages", "labels": {name: 0/1/null}}
    and optionally "group": items sharing one (a contrast pair) are held out
    together by the monitor's cross-fitting.
    Every item is captured at `cfg.position` only (`capture(positions=...)`),
    stacked into a pellet family (`blast.pellets`), and laid out
    (`blast.layouts`); the result carries its analysis record.
    """
    import blast as blast_mod
    from capture import load_model

    cfg = cfg or blast_mod.BlastConfig(model=model if isinstance(model, str) else "")
    # every check that can fail on the file, before a capture that can take
    # an hour on a large model
    ids = [str(it.get("id", "")) for it in items]
    if len(items) < 3:
        raise ValueError(f"a blast needs at least 3 items; got {len(items)}")
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("every item needs an id, and ids must be unique")
    if any(it.get("text") is None and not it.get("messages") for it in items):
        raise ValueError("every item needs 'text' or 'messages'")
    bad = set(cfg.methods) - {"monitor", "open"}
    if bad:
        raise ValueError(f"unknown layout {sorted(bad)}; choose from monitor, open")
    if any(it.get("messages") for it in items):
        # messages always go through the template, whatever --chat said
        from dataclasses import replace as _replace
        cfg = _replace(cfg, chat=True)
    if isinstance(model, str):
        model, tokenizer = load_model(model, device=cfg.device, dtype=cfg.dtype)
    trajs, texts, sent = [], [], []
    for item in items:
        text, input_ids = _blast_input(tokenizer, item, cfg.chat)
        trajs.append(capture(model, text, tokenizer=tokenizer, top_k=cfg.top_k,
                             device=cfg.device, dtype=cfg.dtype, keep_logits=False,
                             input_ids=input_ids, positions=[cfg.position]))
        texts.append(text)
        # what the model was given, template and system prompt included: the
        # display text alone does not reproduce a chat capture
        sent.append(text if input_ids is None else tokenizer.decode(input_ids))
    fam = blast_mod.pellets(trajs, ids,
                            labels=[it.get("labels") or {} for it in items],
                            texts=texts, position=0,
                            groups=[it.get("group") for it in items])
    fam.meta["position"] = cfg.position
    built, skipped = blast_mod.layouts(fam, methods=cfg.methods, seed=cfg.seed,
                                       folds=cfg.folds, null_draws=cfg.null_draws,
                                       copy_tol=cfg.copy_tol)
    result = blast_mod.scene(fam, built, skipped)
    result["analysis"] = provenance_mod.record(cfg, prompts=sent, trajs=[fam])
    return result
