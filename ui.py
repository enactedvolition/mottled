"""Mottled UI: the Streamlit explorer, and the project's flat public API.

Run with:  streamlit run ui.py

The pipeline and the renderers live in their own modules now — `pipeline.py`
(capture -> project -> density -> terrain -> paths) and `render.py` (the
Plotly scene and the SAE feature field) — leaving this file the Streamlit
shell that drives them. Everything they export is re-exported here, so the
documented flat API (`from ui import run_pipeline, render, run_scene, ...`)
is unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import attractor as attractor_mod
import cache as cache_mod
import compare as compare_mod
import metrics as metrics_mod
import projection as projection_mod
import sae as sae_mod
import statefile as statefile_mod
from config import (
    DEFAULT_PROMPT,
    DENSITY_CHOICES,
    MODEL_CHOICES,
    PROJECTION_CHOICES,
    TRAJECTORY_MODES,
    MarbleConfig,
)
from neighbors import TokenNeighbors
from trajectory import StateTrajectory
from tuned_lens import EARLY_LAYER_CAVEAT, readout_name

# The flat public API: these lived in this module before it was split, and
# every caller (README, tests, cli.py, serve.py, notebooks) still imports them
# from here. The underscored helpers come along because scripts outside the
# repo may reach for them; new code should import from `pipeline` / `render`.
from pipeline import (  # noqa: F401
    _assemble_scene,
    _capture_with,
    attach_features,
    attach_inspector,
    attach_manifest,
    chat_input,
    chat_reply,
    degraded_note,
    run_compare,
    run_intervention,
    run_model_scene,
    run_pipeline,
    run_scene,
)
from render import (  # noqa: F401
    _attention_trace,
    _continuation_text,
    _hover_text,
    field_rgb,
    render,
    render_feature_field,
    render_persistence,
)

__all__ = [
    "run_pipeline", "run_scene", "run_compare", "run_intervention",
    "run_model_scene", "attach_features", "attach_inspector", "attach_manifest",
    "degraded_note",
    "render", "render_feature_field", "render_persistence", "field_rgb",
    "main",
]


def _label_provenance(labels: dict) -> str:
    """The sentence that must accompany auto-interp feature names.

    These explanations are written by a language model reading a feature's
    top activations. They are frequently apt and sometimes wrong, and either
    way they are a *description of correlates*, not a definition of what the
    feature computes — so the surface says who wrote them rather than
    presenting them as the feature's meaning.
    """
    writers = sorted({l.explained_by for l in labels.values() if l.explained_by})
    who = ", ".join(writers) if writers else "an unnamed model"
    return (f"Feature names are **auto-generated explanations** (Neuronpedia, "
            f"written by {who}) — descriptions of what a feature correlates "
            f"with, not of what it computes. Treat them as leads, not labels.")


def chat_panel(st, cfg: MarbleConfig, load_model) -> None:
    """The chat column: the conversation so far, and a capture of its latest
    turn.

    A turn decodes the reply, then captures the conversation as the model was
    sent it, plus the reply: the forward pass that reply came from. The
    capture becomes the explorer's result, so the scene beside the chat is
    the latest turn's alone. Takes the Streamlit module as an argument, like
    `render_model_comparison`.
    """
    messages = st.session_state.setdefault("chat", [])
    if messages and st.session_state.get("chat_model") != cfg.model:
        # the replies so far are another model's, not this one's history
        messages.clear()
        st.session_state.pop("result", None)
        st.caption(f"The model changed to `{cfg.model}`, so the conversation "
                   "started over: the replies so far were another model's.")
    st.session_state["chat_model"] = cfg.model
    history = st.container(height=520)
    said = st.chat_input("Message the model", key="chat_input")
    if said:
        turn = messages + [{"role": "user", "content": said}]
        model, tokenizer = load_model(cfg.model)
        text, ids = chat_input(tokenizer, turn)
        with st.spinner("Replying, then capturing the forward pass…"):
            result = run_pipeline(cfg, text, model=model, tokenizer=tokenizer,
                                  input_ids=ids)
        # joined only once there is a reply: an exception above leaves the
        # conversation as it was, not holding a question never answered
        messages[:] = turn + [{"role": "assistant",
                               "content": chat_reply(tokenizer, result["traj"])}]
        st.session_state["result"], st.session_state["cfg"] = result, cfg
        st.session_state["chat_templated"] = bool(getattr(tokenizer, "chat_template", None))
    with history:
        for m in messages:
            # a reply of only special tokens decodes to nothing; an empty
            # bubble would read as a rendering fault
            st.chat_message(m["role"]).write(
                m["content"] or "*(no text: the reply was only special tokens "
                                "or whitespace)*")

    if not messages:
        return
    result = st.session_state.get("result") or {}
    with st.expander("Sent to the model", expanded=False):
        if not st.session_state.get("chat_templated", True):
            st.caption(f"`{cfg.model}` has no chat template, so the conversation "
                       "goes in as a plain User/Assistant transcript.")
        st.code(result.get("prompt", ""), language=None, wrap_lines=True)
        st.caption("This is the latest turn's input, and the scene is that turn "
                   "alone: the terrain is rebuilt every turn. Earlier replies are "
                   "in it as the chat template re-renders them from text, as any "
                   "chat re-sends its history, so their tokens can differ from "
                   "the ones generated at the time.")
        st.caption("Each turn decodes without a KV cache and captures the whole "
                   "conversation again, so turns slow down as it grows.")
    if st.button("Clear conversation", key="chat_clear"):
        messages.clear()
        st.session_state.pop("result", None)
        st.rerun()


def render_model_comparison(st, result: dict) -> None:
    """The cross-model panel: what the models do, how they represent, and
    where each measurement stops being valid.

    Takes the Streamlit module as an argument so the logic stays importable
    and testable without a browser (the rest of the app-only code lives in
    `main`).
    """
    import crossmodel

    names = result["model_names"]
    sources = result.get("source_trajs") or result["trajs"]
    base = sources[0]

    st.caption(
        f"Built in **readout space** over the **{result['shared_vocab']}** "
        f"vocabulary entries these models share — they have no common hidden "
        f"space, so the shared thing is what they predict, not how they "
        f"represent it. The `{crossmodel.UNSHARED_TOKEN}` component holds the "
        f"mass each model spends on tokens the others do not have.")

    for name, cmp in zip(names[1:], result.get("model_comparisons") or []):
        st.markdown(f"**{names[0]} vs {name}**")
        st.write(f"final-layer readout divergence: **{cmp.final_divergence:.4f}** "
                 f"(Jensen-Shannon, nats) · top-1 {cmp.top_a!r} vs {cmp.top_b!r}")
        if cmp.agree_from is not None:
            st.write(f"top-1 predictions agree from normalized layer "
                     f"**{cmp.agree_from}** onward")
        else:
            st.write("top-1 predictions never settle on the same token")
        st.line_chart({"readout divergence": cmp.divergence})
        st.caption("Depth is normalized before comparing: layer 6 of a "
                   "12-layer model is not layer 6 of a 32-layer one.")

    for name, other in zip(names[1:], sources[1:]):
        try:
            align = crossmodel.layer_similarity(base, other)
        except ValueError as err:
            st.caption(f"Layer alignment for **{name}** unavailable — {err}")
            continue
        st.markdown(f"**Layer alignment · {names[0]} → {name}**")
        st.write(align.summary())
        rows = {f"layer {l}": int(b) for l, b in enumerate(align.best)}
        st.bar_chart({"best-matching layer": list(rows.values())})
        weak = [l for l, ok in enumerate(align.identified()) if not ok]
        if weak:
            st.caption(f"Layers {weak} sit on flat rows: their best match "
                       f"barely beats the field, so read them as *unresolved*, "
                       f"not as a correspondence.")

    gens = [t for t in sources if isinstance(t.meta.get("generation"), dict)]
    if len(gens) >= 2:
        st.markdown("**Generation**")
        gcmp = crossmodel.compare_generations(gens[0], gens[1])
        st.write(gcmp.summary())
        st.dataframe(
            [{"step": i,
              names[0]: gcmp.tokens_a[i], "p(a)": round(float(gcmp.p_a[i]), 4),
              names[1]: gcmp.tokens_b[i], "p(b)": round(float(gcmp.p_b[i]), 4),
              "same": bool(gcmp.agree[i])}
             for i in range(len(gcmp.agree))],
            use_container_width=True, hide_index=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Mottled", page_icon="🔮", layout="wide")

    # Incision design language on top of the theme in .streamlit/config.toml:
    # 1px borders, near-sharp corners, mono for data values, tracked overlines.
    st.markdown("""<style>
      [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
      }
      [data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 11px; color: #818FB8;
      }
      [data-testid="stSidebar"] { border-right: 1px solid #1E2540; }
      div[data-testid="stExpander"] details {
        border: 1px solid #1E2540; border-radius: 2px;
      }
      .stButton button, .stDownloadButton button, div[data-baseweb="select"] > div,
      .stTextArea textarea, .stTextInput input {
        border-radius: 3px !important;
      }
      .stTextArea textarea, .stTextInput input { border: 1px solid #1E2540; }
      code { font-family: "JetBrains Mono", ui-monospace, monospace; }
      h1, h2, h3 { letter-spacing: -0.01em; }
    </style>""", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Loading model…")
    def load_model_cached(name: str):
        from capture import load_model

        return load_model(name)

    # ------------------------------------------------------------ left panel
    with st.sidebar:
        st.title("Mottled")
        st.caption("Latent trajectory explorer")
        chat_on = st.toggle(
            "Chat", key="chat_mode",
            help="Talk to the model on the left. The right shows the latest "
                 "turn: the conversation as the model was sent it, and its "
                 "reply as the decode axis. The terrain is rebuilt every turn.")
        prompt = prompt_b = extra_models = ""
        if not chat_on:
            prompt = st.text_area("Prompt", DEFAULT_PROMPT, key="prompt")
            prompt_b = st.text_area("Overlay prompts (one per line, optional)", "",
                                    key="prompt_b",
                                    help="Each line becomes another run drawn on the "
                                         "same terrain and compared against the prompt above.")
        model_name = st.selectbox("Model", MODEL_CHOICES, index=0, key="model")
        if not chat_on:
            extra_models = st.text_input(
                "Compare models (comma-separated, optional)", "", key="extra_models",
                help="Draw other models on the SAME terrain for this prompt. They "
                     "share no hidden space, so the scene is built in readout space "
                     "— the vocabulary the models have in common — with the mass "
                     "each spends outside it kept visible. Overrides the overlay "
                     "prompts above.")
        proj_name = st.selectbox("Projection", PROJECTION_CHOICES, key="projection")
        dens_name = st.selectbox("Density estimator", DENSITY_CHOICES, key="density")
        top_k = st.slider("Top-k", 1, 10, 5, key="top_k")
        lens = st.selectbox("Readout lens", ["logit", "tuned"], key="lens",
                            help="logit: the raw logit lens, biased at early layers. "
                                 "tuned: the published tuned lens for this model "
                                 "(GPT-2, Pythia-deduped, Llama-2/3, OPT; none yet "
                                 "for Qwen), which corrects most of that bias.")
        if chat_on:
            gen_tokens = st.slider("Reply tokens", 1, 32, 32, key="reply_tokens",
                                   help="The most tokens a reply may run to; it "
                                        "stops early at the model's end token. "
                                        "The cap is Generate tokens' own: each "
                                        "token is a full forward pass.")
        else:
            gen_tokens = st.slider("Generate tokens", 0, 32, 0, key="generate_tokens",
                                   help="Decode this many tokens before capturing: the "
                                        "scene then covers prompt + continuation — the "
                                        "decode axis. Exact: the captured states are "
                                        "the ones that existed at each decode step.")
        gen_temp = 0.0
        if gen_tokens:
            gen_temp = st.slider("Decode temperature", 0.0, 2.0, 0.0, 0.1,
                                 key="generate_temperature",
                                 help="0 decodes greedily; above 0 samples from the "
                                      "tempered distribution (seeded, reproducible).")
        mode = st.selectbox("Trajectory", TRAJECTORY_MODES, key="trajectory_mode")
        invert = st.checkbox("Dense regions as valleys", value=False, key="invert")
        sae_on = st.checkbox("SAE feature overlay", value=False, key="sae_overlay",
                             help="Color trajectory markers by one feature's activation. "
                                  "Uses an untrained demo dictionary; load real weights "
                                  "with sae.load_npz for interpretable features.")
        field_on = st.checkbox("SAE feature field (domain coloring)", value=False,
                               key="sae_field",
                               help="Wolfram-style domain coloring of the projection "
                                    "plane: every point is inverse-projected to hidden "
                                    "space and run through the SAE — hue = dominant "
                                    "feature, brightness = activation, rings = "
                                    "magnitude octaves. Exact for pca; umap's inverse "
                                    "is approximate and gets unreliable away from the "
                                    "captured states (the field's grid always pads "
                                    "20% past them), so treat those regions as "
                                    "illustrative, not measured.")
        attention_on = st.checkbox("Show attention flow", value=False, key="attention_flow",
                                   help="Draw head-averaged attention edges between token "
                                        "states at the selected layer.")
        run = False if chat_on else st.button("Run capture", type="primary",
                                              use_container_width=True, key="run")
        st.caption("Play, pause and scrub inside the figure animate the marble; "
                   "the layer slider below drives the inspector.")

    asked = MarbleConfig(model=model_name, projection=proj_name, density=dens_name,
                         top_k=top_k, trajectory_mode=mode, invert_terrain=invert,
                         generate_tokens=gen_tokens, generate_temperature=gen_temp,
                         lens=lens)
    if chat_on:
        col_chat, col_viz = st.columns([2, 3])
        with col_chat:
            # every turn is a new conversation, so a disk cache would only
            # ever write: a full-vocabulary capture, hundreds of MB, per turn
            chat_panel(st, replace(asked, use_cache=False), load_model_cached)

    if run and prompt.strip():
        cfg = asked
        # the chat and the explorer share one scene; a new one ends the chat
        # rather than sit beside a conversation it did not come from
        st.session_state.pop("chat", None)
        model, tokenizer = load_model_cached(model_name)
        overlays = [p.strip() for p in prompt_b.splitlines() if p.strip()]
        others = [m.strip() for m in extra_models.split(",") if m.strip()]
        with st.spinner("Capturing forward pass…"):
            if others:
                names = [model_name] + others
                loaded = {model_name: (model, tokenizer)} if model is not None else {}
                for name in others:
                    loaded[name] = load_model_cached(name)
                st.session_state["result"] = run_model_scene(cfg, prompt, names,
                                                             loaded=loaded)
            elif overlays:
                st.session_state["result"] = run_scene(cfg, [prompt] + overlays,
                                                       model=model, tokenizer=tokenizer)
            else:
                st.session_state["result"] = run_pipeline(cfg, prompt,
                                                          model=model, tokenizer=tokenizer)
            st.session_state["cfg"] = cfg

    result = st.session_state.get("result")
    if chat_on and not st.session_state.get("chat"):
        result = None               # a prompt capture is not this conversation
    if result is None:
        if chat_on:
            col_viz.info("Say something on the left: each reply's forward pass "
                         "is drawn here.")
        else:
            st.info("Enter a prompt and press **Run capture** to explore the "
                    "latent manifold.")
        return

    cfg: MarbleConfig = st.session_state["cfg"]
    traj: StateTrajectory = result["traj"]

    @st.cache_resource(show_spinner=False)
    def _demo_sae(dim: int, n_features: int):
        return sae_mod.demo_sae(dim, n_features)

    @st.cache_resource(show_spinner="Fetching trained SAE (~150 MB, cached)…")
    def _hub_sae(repo_id: str, subfolder: str):
        return sae_mod.fetch_from_hub(repo_id, subfolder)

    @st.cache_resource(show_spinner="Naming features…")
    def _feature_labels(source: tuple, indices: tuple):
        return sae_mod.fetch_labels(indices, source=source)

    # GPT-2 gets a real trained dictionary by default (fetched on first use);
    # everything else falls back to the demo dictionary, clearly labeled.
    _REAL_SAE = {768: ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre")}

    def _real_sae_source():
        """The trained dictionary for this capture, if one applies.

        Keyed on the hidden width, which only identifies a model's residual
        stream when the states *are* a residual stream. A readout-space
        trajectory's axes are vocabulary entries, not hidden dimensions, so a
        width collision there would offer a residual SAE for probability
        vectors — the fit report would call it out, but it should never be
        offered in the first place."""
        if traj.meta.get("space") == "readout":
            return None
        return _REAL_SAE.get(traj.dim)

    def _active_sae():
        source = _real_sae_source() if st.session_state.get("sae_real", True) else None
        if source is not None:
            try:
                return _hub_sae(*source), source
            except Exception as err:
                st.caption(f"trained SAE unavailable ({err}); using the demo dictionary")
        return _demo_sae(traj.dim, cfg.sae_features), None

    def _sae_fit_caption(active_sae, source):
        """Measured calibration, printed wherever features are shown: an SAE
        only means something on the activation distribution it was trained
        on, so measure the fit instead of trusting provenance."""
        fit = sae_mod.fit_report(active_sae, traj)
        name = f"`{source[0]}`" if source else "demo dictionary (untrained)"
        line = (f"SAE fit — {name}: best reconstruction at layer "
                f"{fit.best_layer} ({fit.best_error:.0%} of norm off, "
                f"{fit.active_frac[fit.best_layer]:.1%} of features firing).")
        if fit.best_error > 0.5:
            line += (" **This dictionary does not fit these activations** — "
                     "it was likely trained on differently-processed residuals "
                     "(e.g. TransformerLens folding/centering) or a different "
                     "model. Features shown are extrapolation, not measurement; "
                     "capture via `models.hooked.from_hooked_transformer` for "
                     "the calibrated pairing.")
        st.markdown(line)

    acts = overlay = None
    overlay_label = "feature"
    with st.sidebar:
        layer = st.slider("Layer", 0, traj.n_layers - 1, traj.n_layers - 1, key="layer")
        if (sae_on or field_on) and _real_sae_source() is not None:
            st.checkbox("Trained SAE (gpt2-small-res-jb, layer 8)", value=True,
                        key="sae_real",
                        help="Joseph Bloom's GPT-2-small residual-stream SAE, "
                             "fetched once from the HF hub (~150 MB) and cached. "
                             "It was trained at layer 8's resid_pre: activations "
                             "at other layers are extrapolation, not calibration. "
                             "Untick to use the untrained demo dictionary.")
        if sae_on:
            active_sae, sae_source = _active_sae()
            _sae_fit_caption(active_sae, sae_source)
            acts = sae_mod.feature_trajectory(traj, active_sae)
            choices = [int(f) for f in sae_mod.active_features(acts, k=25)]
            # Name the features that actually fire. A dictionary's features
            # are indices until something explains them, and an unnamed
            # overlay is a colour with no meaning. Only the active handful is
            # looked up (there is no bulk endpoint), cached on disk, and a
            # failed lookup simply leaves the bare index.
            if choices and sae_source is not None:
                fetched = _feature_labels(tuple(sae_source), tuple(choices))
                if fetched:
                    sae_mod.apply_labels(active_sae, fetched)
                    st.caption(_label_provenance(fetched))
            if choices and result.get("traj_b") is None:
                feat = st.selectbox(
                    "SAE feature", choices,
                    format_func=lambda f: f"{active_sae.feature_label(f)} · "
                                          f"peak {acts[..., f].max():.2f}",
                    key="feature",
                )
                overlay_label = active_sae.feature_label(feat)
                overlay = [
                    acts[:, t.token if isinstance(t.token, int) else traj.n_tokens - 1, feat]
                    for t in result["trajectories"]
                ]

        import io as _io

        # inspector layers (neighbors, attn/MLP share) so the shareable
        # viewer is not the lesser surface
        attach_inspector(result, n_neighbors=cfg.n_neighbors)
        sae_record = {}
        if acts is not None and sae_source is not None:
            # trained dictionary active: export its feature layer + measured
            # fit with the scene (demo features are decorative — not exported)
            attach_features(result, active_sae,
                            source=sae_source[0], hook=sae_source[1])
            sae_record = {"sae": active_sae, "sae_source": sae_source[0],
                          "sae_hook": sae_source[1]}
        # the analysis record: an exported scene states its own
        # parameterization rather than relying on the user's notes
        attach_manifest(result, cfg, **sae_record)
        _buf = _io.BytesIO()
        statefile_mod.save_scene(result, _buf)
        st.download_button("Export scene (.mtj)", data=_buf.getvalue(),
                           file_name="scene.mtj", mime="application/octet-stream",
                           use_container_width=True, key="export_scene",
                           help="Portable scene for the web viewer (viewer/) "
                                "or any .mtj consumer — see docs/mtj-format.md.")

        with st.expander("Intervention (perturb & replay)", expanded=False):
            iv_layer = st.slider("Edit layer", 0, traj.n_layers - 1,
                                 traj.n_layers - 1, key="iv_layer")
            iv_kind = st.selectbox("Edit", ["push toward token", "inject noise",
                                            "freeze block"], key="iv_kind")
            iv_target = ""
            if iv_kind == "push toward token":
                iv_target = st.text_input("Target token", "Berlin", key="iv_target")
            iv_scale = st.slider("Strength", 0.0, 100.0, 30.0, key="iv_scale")
            if st.button("Run intervention", key="iv_run"):
                from intervene import (FreezeLayer, InjectNoise, Perturb,
                                       direction_from_token)

                model, tokenizer = load_model_cached(cfg.model)
                target_id = None
                if iv_kind == "push toward token":
                    ids = tokenizer(iv_target, add_special_tokens=False)["input_ids"]
                    if not ids:
                        st.warning("target token is empty")
                        st.stop()
                    target_id = int(ids[0])
                    # data-derived: the target token's own embedding axis
                    direction = direction_from_token(traj, target_id)
                    edits = [Perturb(iv_layer, iv_scale * direction, token=-1)]
                elif iv_kind == "inject noise":
                    edits = [InjectNoise(iv_layer, iv_scale, token=-1)]
                else:
                    edits = [FreezeLayer(min(iv_layer, traj.n_layers - 2))]
                with st.spinner("Replaying under intervention…"):
                    st.session_state["result"] = run_intervention(
                        cfg, result["prompt"], edits, model, tokenizer,
                        target_id=target_id)
                st.rerun()

    @st.cache_resource(show_spinner=False)
    def _neighbor_cache(key: str):  # one TokenNeighbors per capture
        return {}

    def _token_neighbors(t: StateTrajectory) -> TokenNeighbors:
        holder = _neighbor_cache(cache_mod.make_key(t.meta.get("model"), t.meta.get("prompt")))
        if "tn" not in holder:
            holder["tn"] = TokenNeighbors(t.embedding_matrix, t.vocab)
        return holder["tn"]

    if chat_on:
        col_info = col_viz          # the inspector sits under the scene
    else:
        col_viz, col_info = st.columns([3, 1])

    extra_runs = None
    if result.get("trajs") is not None and len(result["trajs"]) > 2:
        extra_runs = [
            (result["trajs"][i], result["trajectories_list"][i], result["fine_paths_list"][i])
            for i in range(2, len(result["trajs"]))
        ]

    basin = attractor_mod.analyze(traj, result["coords"], result["landscape"])

    quality = result.get("quality")
    with col_viz:
        # the threshold drives the prose, the ✕ markers, and the viewer's panel
        low_fidelity = projection_mod.LOW_FIDELITY
        if (degraded := degraded_note(traj.meta)) is not None:
            st.warning(degraded)
        if quality is not None:
            ev = (f"keeps **{quality.explained_variance:.0%}** of the variance · "
                  if quality.explained_variance is not None else "")
            fidelity = projection_mod.fidelity_summary(quality.preservation)
            st.markdown(
                f"**Projection fidelity** — {ev}mean neighborhood preservation "
                f"**{fidelity['mean']:.2f}** (k={quality.k}). "
                f"**{fidelity['low_fraction']:.0%}** of states are low-fidelity — "
                f"flagged ✕ on the scene and drawn where the projection *could* "
                f"put them, not where they truly are.")
        gen0 = traj.meta.get("generation")
        if isinstance(gen0, dict):
            st.markdown(
                f"**Decode** — {gen0['mode']}"
                + (f" · T={gen0['temperature']:.1f}" if gen0["mode"] == "sample" else "")
                + f" · +{gen0['new_tokens']} tokens after the {gen0['prompt_tokens']}-token "
                  f"prompt: `{_continuation_text(traj.meta)}` — generated tokens are "
                  f"drawn with open-diamond markers and a `+` name prefix.")
        fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                     current_layer=layer, frames_per_layer=cfg.frames_per_layer,
                     frame_ms=cfg.frame_ms,
                     traj_b=result.get("traj_b"),
                     trajectories_b=result.get("trajectories_b"),
                     fine_paths_b=result.get("fine_paths_b"),
                     extra_runs=extra_runs,
                     overlay=overlay, overlay_label=overlay_label,
                     show_attention=attention_on,
                     basin=basin, quality=quality, low_fidelity=low_fidelity)
        st.plotly_chart(fig, use_container_width=True, key="scene")
        st.caption('The terrain is a density field over the projected states '
                   'themselves — height means "many states landed here", not an '
                   'external landscape. The pinned callout marks the basin; open '
                   '**Why this attractor** in the inspector for this run\'s numbers.')

        st.caption("Scenes generate hypotheses, not mechanisms — "
                   "`docs/validity.md` is the inferential contract.")
        with st.expander("What this is — and is not", expanded=False):
            st.markdown(
                "Mottled visualizes the **geometry of a run's latent "
                "trajectories** — where hidden states go and pile up — and "
                "measures how much of that geometry survives the projection. "
                "It summarizes representation-space behavior under the analysis "
                "choices you declared; it generates mechanistic hypotheses, it "
                "does **not** establish mechanisms (the full inferential "
                "contract is `docs/validity.md`).\n\n"
                "- A basin is a **state concentration region** under this "
                "projection and density estimator — states *accumulating*, not "
                "a circuit *computing*. "
                "Attention flow and the intervention divergence are **measurements** "
                "of what happened, not identified causes.\n"
                "- Logit-lens readouts are **probe diagnostics** — what the "
                "output head would say if pointed at an intermediate state — "
                "and the raw lens is **biased, most at early layers**; the "
                "*tuned* readout lens corrects much of that and is still a "
                "readout. Nearest-token lists are **representation-space** "
                "neighbors, not verified semantic ones.\n"
                "- An SAE overlay is only as interpretable as the SAE you load; the "
                "bundled `demo_sae` is a random dictionary (decorative). Load real "
                "weights with `sae.load_npz` / the `mottled-convert-sae` CLI.\n"
                "- For **verified causal claims** — circuit discovery, path patching, "
                "activation patching at scale — use a dedicated tool "
                "(TransformerLens, ACDC, EAP). Mottled is the honest map you read "
                "*before* and *alongside* them, not a substitute.")

        if field_on:
            projector = result.get("projector")
            if projector is None or not hasattr(projector, "inverse_transform"):
                st.caption("The feature field needs an invertible projection — "
                           "re-run with projection = pca.")
            else:
                if cfg.projection != "pca":
                    st.caption("Projection is umap: its inverse is approximate, so "
                               "away from the actual captured states (the grid pads "
                               "20% past them) the field can look confident while "
                               "showing extrapolation artifacts, not measurement.")
                field_sae, field_source = _active_sae()
                _sae_fit_caption(field_sae, field_source)
                if field_source is not None:
                    st.caption(f"Dictionary `{field_source[0]}` was trained at "
                               f"one hook point (`{field_source[1]}`); regions "
                               "far from the captured states, and states at "
                               "other layers, are extrapolation.")
                landscape = result["landscape"]
                try:
                    fld = sae_mod.feature_field(field_sae, projector,
                                                landscape.grid_x, landscape.grid_y)
                except Exception as err:  # e.g. umap's approximate inverse failing
                    st.caption(f"feature field unavailable: {err}")
                else:
                    relief = st.radio("Field view", ["plane", "relief"],
                                      horizontal=True, key="field_mode") == "relief"
                    tok_idx = cfg.trajectory_token % traj.n_tokens
                    st.plotly_chart(
                        render_feature_field(fld, field_sae,
                                             path=result["coords"][:, tok_idx, :2],
                                             relief=relief),
                        use_container_width=True, key="field")
                    st.caption('Domain coloring of the projection plane — the '
                               'complex-plane analogue for the latent manifold: every '
                               'point of the plane is inverse-projected to hidden '
                               'space and run through the SAE. Hue = which feature '
                               'decodes strongest there (the "phase"), brightness = '
                               'its activation (the "modulus"), rings = magnitude '
                               'octaves; the white path is this run\'s tracked-token '
                               'trajectory crossing feature domains. Demo dictionary — '
                               'load real weights with sae.load_npz for interpretable '
                               'domains.')

    # ----------------------------------------------------------- right panel
    with col_info:
        st.subheader("Inspector")
        token = st.selectbox(
            "Token", range(traj.n_tokens),
            index=traj.n_tokens - 1,
            format_func=lambda i: f"{i}: {traj.tokens[i]!r}",
            key="token",
        )
        state = traj.state(layer, token)
        a, b = st.columns(2)
        a.metric("Layer", layer)
        b.metric("Token", f"{state.text!r}")
        a.metric("Entropy", f"{state.entropy:.2f} nats")
        b.metric("Vector norm", f"{state.norm:.1f}")

        lens_name = readout_name(traj.meta)
        st.markdown(f"**Top predictions** *({lens_name})*")
        if lens_name == "logit lens":
            st.caption(EARLY_LAYER_CAVEAT[0].upper() + EARLY_LAYER_CAVEAT[1:] + ".")
        for tok, p in state.topk[: cfg.top_k]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{tok!r} — {p:.1%}")

        if traj.embedding_matrix is not None and traj.vocab is not None:
            st.markdown("**Nearest neighbors (representation space)**")
            tn = _token_neighbors(traj)
            for tok, sim in tn.nearest(state.vector, k=cfg.n_neighbors):
                st.write(f"`{tok}`  ·  cos {sim:.3f}")

        if acts is not None:
            st.markdown("**Top SAE features** *(demo dictionary)*")
            fired = sae_mod.top_features(acts, layer, token, k=cfg.top_k)
            for fid, act in fired:
                st.write(f"`f{fid}`  ·  act {act:.2f}")
            if not fired:
                st.caption("no features fire at this state")

        if traj.attention is not None and layer >= 1:
            st.markdown("**Attention** *(head-averaged, into this layer)*")
            weights = traj.attention[layer - 1, token]
            for src in np.argsort(-weights)[:5]:
                if weights[src] <= 0.01:
                    continue
                st.write(f"`{traj.tokens[src]}` ({src})  ·  {weights[src]:.2f}")

        if traj.components is not None:
            with st.expander("Residual decomposition", expanded=False):
                shares = metrics_mod.component_shares(traj, token=token)
                st.markdown("Share of each block's residual write")
                st.line_chart({"attention": shares[:, 0], "mlp": shares[:, 1]})

        with st.expander("Why this attractor", expanded=False):
            report = (basin if basin.token == token % traj.n_tokens else
                      attractor_mod.analyze(traj, result["coords"],
                                            result["landscape"], token=token))
            st.markdown(attractor_mod.explain(report, traj, quality=quality))
            if len(report.step):
                st.markdown("**Per-layer step of this token** *(hidden space)*")
                st.line_chart({"step": report.step})
            if report.entropy is not None:
                st.markdown("**Predictive entropy per layer** *(nats)*")
                st.line_chart({"entropy": report.entropy})

        with st.expander("Research metrics", expanded=False):
            summary = metrics_mod.summarize(traj, result["coords"], token=token)
            for name, value in summary.items():
                space = metrics_mod.SPACE.get(name, "unlabelled")
                st.write(f"{name.replace('_', ' ')} *({space})*: **{value:.3f}**")
            st.caption("*projection* metrics measure the picture: another projection "
                       "draws another path. *hidden space* metrics are measured on the "
                       "full-dimensional states, *readout* metrics on the lens's "
                       "output distributions.")

        gen_rec = traj.meta.get("generation")
        if isinstance(gen_rec, dict) and gen_rec.get("steps"):
            with st.expander("Decode steps", expanded=False):
                st.caption("The sampling distribution at each decode step — "
                           "p is the chosen token's probability, entropy is "
                           "the distribution's spread (nats).")
                st.dataframe(
                    [{"step": s_i + 1, "token": s["token"],
                      "p": round(float(s["p"]), 4),
                      "entropy": round(float(s["entropy"]), 3)}
                     for s_i, s in enumerate(gen_rec["steps"])],
                    use_container_width=True, hide_index=True)

        with st.expander("Uncertainty", expanded=False):
            q = result.get("quality")
            if q is not None:
                if q.explained_variance is not None:
                    st.write(f"projection keeps **{q.explained_variance:.0%}** "
                             "of hidden-space variance")
                st.write(f"neighborhood preservation (k={q.k}): "
                         f"mean **{q.preservation.mean():.2f}** · "
                         f"this state **{q.preservation[layer, token]:.2f}**")
                if q.residual is not None:
                    st.write(f"this state lies **{q.residual[layer, token]:.0%}** "
                             "off the projection plane")
                st.markdown("**Neighborhood preservation per layer** *(this token)*")
                st.line_chart({"preservation": q.preservation[:, token]})
            se = result["landscape"].density_se
            if se is not None:
                st.write(f"density bootstrap SE: mean **{se.mean():.3f}** · "
                         f"max **{se.max():.3f}** (normalized density units)")
            st.caption("The 2-D picture is a lossy view of hidden space and the "
                       "terrain is an estimate from finitely many states — treat "
                       "low-preservation states and high-SE regions as suggestive, "
                       "not measured. The web viewer's uncertainty toggle shows "
                       "the SE field on the terrain itself.")

        # A/B is a layer-for-layer measurement, so it is only defined when the
        # two runs have the same depth — true for prompts through one model,
        # generally false across models (which get their own panel below).
        if (result.get("traj_b") is not None
                and result["traj_b"].n_layers == traj.n_layers
                and result["traj_b"].dim == traj.dim):
            with st.expander("A/B comparison", expanded=True):
                st.caption(f"B = {result.get('prompt_b', '')}")
                cmp = compare_mod.compare(traj, result["traj_b"],
                                          result["coords"], result["coords_b"],
                                          token=token)
                st.write(f"shared prefix: **{cmp.shared_tokens} tokens**")
                st.write(f"Hausdorff distance: **{cmp.hausdorff:.3f}**")
                st.write(f"DTW distance (normalized): **{cmp.dtw.normalized:.3f}**")
                if cmp.onset_token is not None:
                    st.write(f"states separate from token **{cmp.onset_token}**, "
                             f"layer **{cmp.onset_layer}**")
                if cmp.readout_changed is not None:
                    st.write(f"top-1 prediction differs from layer **{cmp.readout_changed}**")
                st.markdown("**A–B distance per layer**")
                st.line_chart(cmp.profile)

        if result.get("model_names"):
            with st.expander("Model comparison", expanded=True):
                render_model_comparison(st, result)

        if (div := result.get("divergence")) is not None:
            with st.expander("Intervention divergence", expanded=True):
                st.write(f"separation onset: layer **{div.onset}**")
                if div.readout_changed is not None:
                    st.write(f"prediction flips at layer **{div.readout_changed}**")
                else:
                    st.write("top-1 prediction unchanged")
                st.line_chart(div.profile)
                if (fth := result.get("faithfulness")) is not None:
                    tt = fth.target_token or f"id {fth.target}"
                    hit = "yes" if fth.steer_hits_target else "no"
                    st.write(f"faithfulness toward `{tt}` — steer shifts its logit "
                             f"**{fth.steer_shift:+.2f}**, a norm-matched random "
                             f"delta **{fth.control_shift:+.2f}** "
                             f"(direction-specific effect **{fth.effect:+.2f}**; "
                             f"top-1 now the target: {hit})")
                    st.caption("A large *effect* means the steering direction, not "
                               "just the perturbation's size, moved the model. This "
                               "is a measured counterfactual, not an isolated circuit.")

        if (prof := result.get("persistence")) is not None:
            with st.expander("Injection persistence", expanded=True):
                st.plotly_chart(render_persistence(prof), use_container_width=True,
                                key="persistence")
                st.caption("The faithfulness effect (steer minus norm-matched "
                           "random control) read at every layer from the edit "
                           "down, with the baseline's MLP write-share overlaid. "
                           "A smooth decay or hold says the residual stream "
                           "carried the effect; a drop that returns alongside an "
                           "MLP spike says a later block rewrote it. Like the "
                           "divergence readout, this is a measurement of what "
                           "happened downstream — not a claimed cause.")

        if result.get("comparisons") and len(result["comparisons"]) > 1:
            with st.expander("Scene comparisons (vs A)", expanded=True):
                for i, cmp in enumerate(result["comparisons"], start=1):
                    st.write(f"**{chr(65 + i)}** · Hausdorff {cmp.hausdorff:.3f} "
                             f"· DTW {cmp.dtw.normalized:.3f} "
                             f"· shared prefix {cmp.shared_tokens}")


if __name__ == "__main__":
    main()
