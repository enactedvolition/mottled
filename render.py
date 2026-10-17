"""Plotly renderers: the 3-D scene, and the SAE feature field.

Pure functions returning `plotly.graph_objects.Figure` — no Streamlit — so a
scene renders identically in the explorer, a notebook, or a script. Styled to
the Incision design language: dark navy void, one precision-blue accent,
semantic data colors, mono for data values.

Split out of `ui.py` (which re-exports everything here, so
`from ui import render` keeps working).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import attractor as attractor_mod
import design_tokens as tokens
import projection as projection_mod
import sae as sae_mod
import terrain as terrain_mod
import trajectory as trajectory_mod
from trajectory import StateTrajectory

# --------------------------------------------------------------------------
# Design tokens come from the single source of truth (design_tokens.py); the
# Streamlit config and the web viewer mirror the same values, guarded by
# tests/test_tokens.py.
_MARBLE_COLORS = tokens.MARBLE_COLORS
_TERRAIN_COLORSCALE = tokens.TERRAIN_COLORSCALE
_FONT_SANS = tokens.FONT_SANS
_FONT_MONO = tokens.FONT_MONO


def _continuation_text(meta: dict) -> str:
    """Human-readable decoded continuation from meta["generation"]."""
    steps = meta.get("generation", {}).get("steps", [])
    toks = [str(s.get("token", "")) for s in steps]
    # Subword pieces carry their own leading space after _clean_token, so they
    # concatenate; word-level tokenizers emit bare words that need one put
    # back. Decided from the pieces themselves rather than from the backend
    # name: the distinction is the tokenizer's, and one backend can be driven
    # with either kind.
    bare = all(tok and not tok[:1].isspace() for tok in toks)
    joined = " ".join(toks) if bare and len(toks) > 1 else "".join(toks)
    return joined.strip()


def _hover_text(traj: StateTrajectory, t: trajectory_mod.Trajectory) -> list[str]:
    texts = []
    for layer in range(len(t.points)):
        token = t.token if isinstance(t.token, int) else traj.n_tokens - 1
        state = traj.state(layer, token)
        top = state.topk[0] if state.topk else ("?", 0.0)
        texts.append(
            f"<b>{t.label or t.token}</b><br>layer {layer}"
            f"<br>entropy {state.entropy:.2f} nats"
            f"<br>top: '{top[0]}' ({top[1]:.0%})"
            f"<br>|h| = {state.norm:.1f}"
        )
    return texts


_DASH_CYCLE = [None, "dash", "dot", "longdash", "dashdot"]


def _attention_trace(
    trajectories: list[trajectory_mod.Trajectory],
    attention: np.ndarray,
    layer: int,
    threshold: float = 0.1,
    top_k: int = 3,
) -> go.Scatter3d | None:
    """Attention-flow segments at one layer: destination states connected to
    the source states they read from (head-averaged weight ≥ threshold,
    self-attention omitted — it would be a zero-length segment)."""
    if layer < 1 or layer - 1 >= len(attention):
        return None
    weights = attention[layer - 1]
    xs, ys, zs = [], [], []
    for dst in range(weights.shape[0]):
        for src in np.argsort(-weights[dst])[:top_k]:
            if src == dst or weights[dst, src] < threshold:
                continue
            p, q = trajectories[src].points[layer], trajectories[dst].points[layer]
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line={"color": "rgba(129,143,184,0.55)", "width": 2},
        name="attention", hoverinfo="skip",
    )


def render(
    traj: StateTrajectory,
    surface: terrain_mod.TerrainMesh,
    trajectories: list[trajectory_mod.Trajectory],
    fine_paths: list[np.ndarray],
    current_layer: int = 0,
    frames_per_layer: int = 4,
    frame_ms: int = 120,
    traj_b: StateTrajectory | None = None,
    trajectories_b: list[trajectory_mod.Trajectory] | None = None,
    fine_paths_b: list[np.ndarray] | None = None,
    extra_runs: list[tuple] | None = None,
    overlay: list[np.ndarray] | None = None,
    overlay_label: str = "feature",
    show_attention: bool = False,
    basin: attractor_mod.BasinReport | None = None,
    quality: projection_mod.ProjectionQuality | None = None,
    low_fidelity: float = projection_mod.LOW_FIDELITY,
) -> go.Figure:
    """Build the 3-D scene: terrain surface, token trajectories, animated marbles.

    Passing `traj_b` / `trajectories_b` / `fine_paths_b` (from `run_compare`)
    overlays a second run on the same terrain; `extra_runs` — a list of
    (traj, trajectories, fine_paths) tuples from `run_scene` — overlays runs
    beyond the second.  Every overlaid run gets its own dash style and an
    A/B/C… label prefix.

    `overlay` colors the primary run's trajectory markers by a per-layer
    scalar (one (L,) array per trajectory, e.g. an SAE feature's activation),
    on a shared color scale with a colorbar.  `show_attention` draws the
    primary run's attention flow at `current_layer` (per-token trajectories
    with a captured attention pattern only).

    `basin` (an `attractor.analyze` report) marks the density peak and pins
    a measured explanation to it — membership count, layer range, settle
    layer, stabilized readout — so the scene says *why* the basin is there,
    not just that it is.
    """
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=surface.x, y=surface.y, z=surface.z,
        colorscale=_TERRAIN_COLORSCALE, opacity=0.94, showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "width": 2,
                        "highlightcolor": "#8FA7F7", "project": {"z": False}}},
        name="manifold", hoverinfo="skip",
    ))

    all_runs = [(traj, trajectories, fine_paths)]
    if trajectories_b:
        all_runs.append((traj_b, trajectories_b, fine_paths_b))
    if extra_runs:
        all_runs.extend(extra_runs)
    multi = len(all_runs) > 1
    runs = [
        (src, trajs, f"{chr(65 + r)} · " if multi else "",
         _DASH_CYCLE[r % len(_DASH_CYCLE)] if r else None)
        for r, (src, trajs, _) in enumerate(all_runs)
    ]
    vmin = vmax = 0.0
    if overlay is not None:
        vmin = float(min(v.min() for v in overlay))
        vmax = float(max(max(v.max() for v in overlay), vmin + 1e-6))
    i = 0
    for run_idx, (src, trajs, prefix, dash) in enumerate(runs):
        gen = src.meta.get("generation") if isinstance(src.meta, dict) else None
        boundary = gen.get("prompt_tokens") if isinstance(gen, dict) else None
        for j, t in enumerate(trajs):
            color = _MARBLE_COLORS[i % len(_MARBLE_COLORS)]
            i += 1
            line = {"color": color, "width": 5}
            if dash:
                line["dash"] = dash
            # generated tokens (the decode axis) read differently from the
            # prompt: "+"-prefixed name, open-diamond markers
            generated = (boundary is not None and isinstance(t.token, int)
                         and t.token >= boundary)
            marker = {"size": 3, "color": color}
            if generated:
                marker = {"size": 4, "color": color, "symbol": "diamond-open"}
            if overlay is not None and run_idx == 0 and j < len(overlay):
                marker = {"size": 5, "color": overlay[j], "colorscale": "Turbo",
                          "cmin": vmin, "cmax": vmax, "showscale": j == 0,
                          "colorbar": {"title": overlay_label, "len": 0.5,
                                       "x": 0.02, "thickness": 12}}
            pts = t.points
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines+markers",
                line=line,
                marker=marker,
                name=prefix + ("+" if generated else "") + str(t.label or t.token),
                text=_hover_text(src, t),
                hoverinfo="text",
            ))

    # Inline honesty: flag the primary run's states whose local neighborhood
    # did not survive the projection, right on the scene — so a low-fidelity
    # basin can't be read as solid structure. The full numbers live in the
    # Uncertainty panel; this is the unavoidable at-a-glance mark. It reads
    # within-layer preservation (projection.fidelity_basis): the pooled score
    # can rate a scrambled projection as highly as PCA.
    if quality is not None:
        basis = projection_mod.fidelity_basis(quality)
        cut = projection_mod.low_fidelity_cut(basis.chance, low_fidelity)
        pres = np.asarray(basis.values)
        lx, ly, lz, ltext = [], [], [], []
        for t in trajectories:
            tok = t.token if isinstance(t.token, int) else traj.n_tokens - 1
            series = pres[:, tok]
            for l in range(min(len(t.points), len(series))):
                if series[l] < cut:
                    lx.append(t.points[l, 0]); ly.append(t.points[l, 1])
                    lz.append(t.points[l, 2] + 0.02)
                    ltext.append(f"low fidelity · {series[l]:.0%} of {basis.metric} "
                                 f"k-NN kept<br>layer {l} · {t.label or t.token}")
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="markers",
                marker={"size": 7, "symbol": "x", "color": tokens.AMBER,
                        "line": {"color": tokens.AMBER, "width": 1}},
                name=f"low fidelity (<{cut:.0%} {basis.metric} k-NN kept)",
                hoverinfo="text", text=ltext))

    if show_attention and traj.attention is not None and len(trajectories) == traj.n_tokens:
        att = _attention_trace(trajectories, traj.attention, current_layer)
        if att is not None:
            fig.add_trace(att)

    basin_annotations = []
    if basin is not None:
        cx, cy = float(basin.center[0]), float(basin.center[1])
        cz = float(terrain_mod.surface_height(surface, np.array([[cx, cy]]))[0])
        span = (f"layers {basin.layer_range[0]}–{basin.layer_range[1]}"
                if basin.layer_range is not None else "no states above threshold")
        lines = [f"attractor basin · {basin.n_members}/{basin.n_states} states · {span}"]
        if basin.settle_layer is not None:
            lines.append(f"settles at layer {basin.settle_layer}")
        if basin.top_token is not None and basin.readout_stable_from is not None:
            lines.append(f"top-1 {basin.top_token!r} from layer {basin.readout_stable_from}")
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz + 0.03], mode="markers",
            marker={"size": 9, "symbol": "diamond-open", "color": "#C8D4FB",
                    "line": {"color": "#4B7CF3", "width": 3}},
            name="attractor", hoverinfo="text", text=["<br>".join(lines)],
        ))
        basin_annotations = [{
            "x": cx, "y": cy, "z": cz + 0.05,
            "text": "<br>".join(lines),
            "font": {"family": _FONT_MONO, "size": 10, "color": "#C8D4FB"},
            "bgcolor": "rgba(12,16,32,0.88)", "bordercolor": "#1E2540",
            "borderwidth": 1, "borderpad": 6,
            "arrowcolor": "#4B7CF3", "arrowwidth": 1.5,
            "ax": 60, "ay": -70, "xanchor": "left", "align": "left",
        }]

    # Marbles: one animated marker per trajectory, positioned along the
    # densified path; fine index f corresponds to layer f / frames_per_layer.
    fine_paths = [p for _, _, fp in all_runs for p in fp]
    # Span the DEEPEST run, not the shallowest. Runs of unequal depth are
    # normal now (two models on one terrain), and taking the minimum silently
    # made the deeper model's late layers unreachable — the scrubber simply
    # stopped early with no sign anything had been cut. A shorter run's
    # marble rests at its final state instead (marble_trace already clamps).
    n_frames = max(len(p) for p in fine_paths) if fine_paths else 0
    max_layers = max(src.n_layers for src, _, _ in all_runs)
    start = min(current_layer * frames_per_layer, max(n_frames - 1, 0))

    def marble_trace(f: int) -> go.Scatter3d:
        xyz = np.array([p[min(f, len(p) - 1)] for p in fine_paths])
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2] + 0.02,
            mode="markers",
            marker={"size": 9, "color": [_MARBLE_COLORS[i % len(_MARBLE_COLORS)]
                                         for i in range(len(fine_paths))],
                    "symbol": "circle", "line": {"color": "#EDF0FA", "width": 2}},
            name="marble", hoverinfo="skip", showlegend=False,
        )

    if n_frames:
        marble_idx = len(fig.data)
        fig.add_trace(marble_trace(start))
        fig.frames = [
            go.Frame(data=[marble_trace(f)], traces=[marble_idx], name=str(f))
            for f in range(n_frames)
        ]
        slider_steps = [
            {
                "args": [[str(layer * frames_per_layer)],
                         {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(layer),
                "method": "animate",
            }
            for layer in range(max_layers)
            if layer * frames_per_layer < n_frames
        ]
        fig.update_layout(
            updatemenus=[{
                "type": "buttons", "direction": "left",
                "x": 0.05, "y": 0.02, "xanchor": "left", "yanchor": "bottom",
                "font": {"color": "#EDF0FA"}, "bgcolor": "#11162A",
                "bordercolor": "#1E2540",
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "steps": slider_steps, "active": min(current_layer, len(slider_steps) - 1),
                "x": 0.05, "y": 0.08, "len": 0.9,
                "font": {"color": "#818FB8", "family": _FONT_MONO, "size": 11},
                "bgcolor": "#1E2540", "activebgcolor": "#4B7CF3",
                "bordercolor": "#1E2540", "tickcolor": "#283050",
                "currentvalue": {"prefix": "layer ", "font": {"size": 13}},
            }],
        )

    axis = {"showbackground": False, "gridcolor": "#1E2540",
            "zerolinecolor": "#283050", "color": "#818FB8"}
    # Real-model scenes span hundreds of units in x/y while terrain height is
    # normalized 0..1; pure "data" aspect flattens the relief into
    # invisibility. Keep data proportions but never let relief drop below
    # ~12% of the box (the web viewer applies the same rule).
    sx = float(surface.x.max() - surface.x.min()) or 1.0
    sy = float(surface.y.max() - surface.y.min()) or 1.0
    sz = float(surface.z.max() - surface.z.min()) or 1.0
    m = max(sx, sy)
    fig.update_layout(
        scene={
            "xaxis": {"title": "manifold x", **axis},
            "yaxis": {"title": "manifold y", **axis},
            "zaxis": {"title": "potential", **axis},
            "aspectmode": "manual",
            "aspectratio": {"x": sx / m, "y": sy / m, "z": max(sz / m, 0.12)},
            "annotations": basin_annotations,
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        height=680,
        paper_bgcolor="#080B18",
        font={"family": _FONT_SANS, "color": "#EDF0FA", "size": 12},
        legend={"x": 0.99, "y": 0.99, "xanchor": "right",
                "bgcolor": "rgba(12,16,32,0.88)", "bordercolor": "#1E2540",
                "borderwidth": 1, "font": {"color": "#818FB8", "size": 11}},
        hoverlabel={"bgcolor": "#161C33", "bordercolor": "#283050",
                    "font": {"family": _FONT_MONO, "color": "#EDF0FA", "size": 11}},
        title={"text": "Mottled — latent trajectory explorer",
               "font": {"size": 13, "color": "#818FB8"}},
    )
    return fig


# --------------------------------------------------------------------------
# Persistence profile — the layer-wise fate of an injected direction.
# --------------------------------------------------------------------------
def render_persistence(profile) -> go.Figure:
    """Line chart of an `intervene.PersistenceProfile`.

    `effect_size` by layer on the primary axis (steer minus norm-matched
    control, logit units), with the baseline pass's `mlp_share` overlaid on a
    secondary 0–1 axis — so a drop-and-return in the effect can be read
    against the residual write that might have rebuilt it. Same measured-not-
    mechanistic framing as the divergence readout.
    """
    layers = profile.layers
    tt = profile.target_token or f"id {profile.target}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=profile.effect_sizes, mode="lines+markers",
        name="effect size", line={"color": tokens.ACCENT, "width": 3},
        marker={"size": 5, "color": tokens.ACCENT},
        hovertemplate="layer %{x}<br>effect %{y:+.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=profile.mlp_shares, mode="lines",
        name="mlp share", yaxis="y2",
        line={"color": tokens.TEAL, "width": 2, "dash": "dot"},
        hovertemplate="layer %{x}<br>mlp share %{y:.0%}<extra></extra>",
    ))
    axis = {"gridcolor": tokens.BORDER, "zerolinecolor": tokens.BORDER_STRONG,
            "color": tokens.FG_2}
    fig.update_layout(
        xaxis={"title": "layer", **axis},
        yaxis={"title": f"effect toward {tt!r} (logits)", **axis},
        yaxis2={"title": "mlp write share", "overlaying": "y", "side": "right",
                "range": [0, 1], "showgrid": False,
                "color": tokens.TEAL, "tickformat": ".0%"},
        shapes=[{"type": "line", "x0": profile.inject_layer, "x1": profile.inject_layer,
                 "y0": 0, "y1": 1, "yref": "paper",
                 "line": {"color": tokens.FG_2, "width": 1, "dash": "dash"}}],
        annotations=[{"x": profile.inject_layer, "y": 1.0, "yref": "paper",
                      "text": "inject", "showarrow": False, "yanchor": "bottom",
                      "font": {"family": _FONT_MONO, "size": 10, "color": tokens.FG_2}}],
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        height=320,
        paper_bgcolor=tokens.BASE, plot_bgcolor=tokens.BASE,
        font={"family": _FONT_SANS, "color": tokens.FG_1, "size": 12},
        legend={"x": 0.99, "y": 0.99, "xanchor": "right",
                "bgcolor": "rgba(12,16,32,0.88)", "bordercolor": tokens.BORDER,
                "borderwidth": 1, "font": {"color": tokens.FG_2, "size": 11}},
        hoverlabel={"bgcolor": tokens.SURFACE_1, "bordercolor": tokens.BORDER_STRONG,
                    "font": {"family": _FONT_MONO, "color": tokens.FG_1, "size": 11}},
        title={"text": "Persistence of the injected direction — steer minus "
                       "norm-matched control, per layer",
               "font": {"size": 12, "color": tokens.FG_2}},
    )
    return fig


# --------------------------------------------------------------------------
# SAE feature field — domain coloring of the projection plane.
# --------------------------------------------------------------------------
_GOLDEN = 0.6180339887498949  # golden-ratio conjugate: well-spread hues


def _feature_hue(ids: np.ndarray) -> np.ndarray:
    """Stable, well-separated hue in [0, 1) per feature id."""
    return (np.asarray(ids, dtype=np.float64) * _GOLDEN) % 1.0


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB, all components in [0, 1]; returns (..., 3)."""
    h6 = (np.asarray(h, dtype=np.float64) % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    s, v = np.broadcast_to(s, i.shape), np.broadcast_to(v, i.shape)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def field_rgb(field: sae_mod.FeatureField) -> np.ndarray:
    """Domain-color a feature field: (H, W, 3) uint8.

    The classical complex-plane recipe, transplanted: hue encodes the
    dominant feature (the "phase" — which dictionary direction this point
    of the manifold decodes into), brightness encodes its activation (the
    "modulus"), and the sawtooth banding on log2(magnitude) draws the
    contour rings at magnitude octaves.  Points where no feature fires
    render as the void.
    """
    fires = field.dominant >= 0
    m = field.magnitude / max(float(field.magnitude.max()), 1e-12)
    hue = _feature_hue(np.maximum(field.dominant, 0))
    sat = np.where(fires, 0.62, 0.0)
    with np.errstate(divide="ignore"):
        octave = np.log2(np.maximum(m, 1e-9))
    band = 0.82 + 0.18 * (octave - np.floor(octave))
    val = np.where(fires, (0.18 + 0.82 * np.sqrt(m)) * band, 0.035)
    return (np.clip(_hsv_to_rgb(hue, sat, val), 0.0, 1.0) * 255).astype(np.uint8)


def render_feature_field(
    field: sae_mod.FeatureField,
    sae: sae_mod.SAE | None = None,
    path: np.ndarray | None = None,
    relief: bool = False,
    labels_k: int = 6,
) -> go.Figure:
    """The SAE feature field as a domain-colored figure.

    `path` (L, 2) overlays a projected trajectory so the run's route through
    feature territory is visible.  `relief=False` is the flat complex-plane
    view; `relief=True` lifts activation magnitude into z, leaving holes
    where no feature fires — the manifold sheets of the dictionary.

    When `sae` carries labels (see `sae.apply_labels`), the largest domains
    are **named on the plane**. Note what is deliberately *not* done: hue
    still comes from the feature id, not from the label's meaning. Golden-
    angle hue encodes *identity* — adjacent ids look maximally different on
    purpose — and recolouring by semantics would turn colour proximity into
    an implied semantic metric it cannot support, while destroying the
    separation the palette exists for. Names are text; colour is identity.
    """
    fig = go.Figure()
    if relief:
        m = field.magnitude / max(float(field.magnitude.max()), 1e-12)
        z = np.where(field.dominant >= 0, m, np.nan)
        hue_scale = [[i / 12, "rgb({:.0f},{:.0f},{:.0f})".format(
            *(255 * _hsv_to_rgb(np.array(i / 12), 0.62, 0.9)))] for i in range(13)]
        fig.add_trace(go.Surface(
            x=field.grid_x, y=field.grid_y, z=z,
            surfacecolor=_feature_hue(np.maximum(field.dominant, 0)),
            colorscale=hue_scale, cmin=0.0, cmax=1.0, showscale=False,
            opacity=0.96, name="feature field", hoverinfo="skip",
        ))
        if path is not None:
            p = np.asarray(path, dtype=np.float64)[:, :2]
            top = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=np.full(len(p), top + 0.06),
                mode="lines+markers", name="trajectory",
                line={"color": "#EDF0FA", "width": 4}, marker={"size": 3, "color": "#EDF0FA"},
            ))
        axis = {"showbackground": False, "gridcolor": "#1E2540",
                "zerolinecolor": "#283050", "color": "#818FB8"}
        fig.update_layout(scene={
            "xaxis": {"title": "manifold x", **axis},
            "yaxis": {"title": "manifold y", **axis},
            "zaxis": {"title": "activation", **axis},
        })
    else:
        dx = float(field.grid_x[1] - field.grid_x[0]) if len(field.grid_x) > 1 else 1.0
        dy = float(field.grid_y[1] - field.grid_y[0]) if len(field.grid_y) > 1 else 1.0
        fig.add_trace(go.Image(
            z=field_rgb(field),
            x0=float(field.grid_x[0]), dx=dx,
            y0=float(field.grid_y[0]), dy=dy,
            hoverinfo="skip", name="feature field",
        ))
        if path is not None:
            p = np.asarray(path, dtype=np.float64)[:, :2]
            fig.add_trace(go.Scatter(
                x=p[:, 0], y=p[:, 1], mode="lines+markers", name="trajectory",
                line={"color": "#EDF0FA", "width": 2},
                marker={"size": 5, "color": "#EDF0FA",
                        "line": {"color": "#080B18", "width": 1}},
                text=[f"layer {l}" for l in range(len(p))], hoverinfo="text",
            ))
        fig.update_xaxes(title="manifold x", gridcolor="#1E2540", color="#818FB8",
                         zeroline=False, constrain="domain")
        fig.update_yaxes(title="manifold y", gridcolor="#1E2540", color="#818FB8",
                         zeroline=False, autorange=True, scaleanchor="x")
    # Name the territories that own real area. Hue tells you *which* feature
    # owns a region; only a label tells you what that feature is about.
    if sae is not None and getattr(sae, "labels", None):
        annotations = []
        for dom in field.domains(k=labels_k):
            text = sae.feature_label(dom.feature)
            if len(text) > 46:
                text = text[:45].rstrip() + "…"
            annotations.append({
                "x": dom.x, "y": dom.y, "text": f"{text}<br>{dom.area:.0%}",
                "showarrow": False, "align": "center",
                "font": {"family": _FONT_MONO, "size": 9, "color": "#EDF0FA"},
                "bgcolor": "rgba(8,11,24,0.72)", "bordercolor": "#1E2540",
                "borderwidth": 1, "borderpad": 3,
            })
        if annotations and not relief:
            fig.update_layout(annotations=annotations)
        elif annotations:
            fig.update_scenes(annotations=[
                {**a, "z": float(np.nanmax(z)) if np.isfinite(z).any() else 1.0}
                for a in annotations])

    n_domains = len(field.features)
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        height=560,
        paper_bgcolor="#080B18", plot_bgcolor="#080B18",
        font={"family": _FONT_SANS, "color": "#EDF0FA", "size": 12},
        showlegend=False,
        hoverlabel={"bgcolor": "#161C33", "bordercolor": "#283050",
                    "font": {"family": _FONT_MONO, "color": "#EDF0FA", "size": 11}},
        title={"text": f"SAE feature field — {n_domains} feature domains · "
                       "hue = dominant feature, brightness = activation, "
                       "rings = magnitude octaves",
               "font": {"size": 12, "color": "#818FB8"}},
    )
    return fig

