"""The shotgun blast (blast.py): one pellet per prompt, every layer.

What a reader of the picture relies on, pinned:
- where every prompt ends on the same token, layer 0 is one point (the
  muzzle is in the data, not drawn in);
- the range is the exact full-space distance, and the open camera never puts
  a pellet further out than it;
- reordering the items, or rotating hidden space, cannot move the picture;
- the monitor's x is cross-fitted, so it reads chance on noise that an
  in-sample direction would split perfectly, and still finds a real signal.
"""
import numpy as np
import pytest

import blast
import statefile
from trajectory import StateTrajectory

N, L, D = 24, 6, 64


def _family(signal=0.0, seed=0, muzzle=True, labels=True):
    """(L, N, D) pellets: a shared layer 0, then states that drift apart with
    depth; the first half carry label 1 and, with `signal`, an offset along
    one direction that grows with depth."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * (N // 2) + [0] * (N // 2))
    h = np.zeros((L, N, D))
    h[0] = rng.normal(size=D) if muzzle else rng.normal(size=(N, D))
    for l in range(1, L):
        h[l] = h[l - 1] + rng.normal(size=(N, D)) * 0.5
        h[l, y == 1, 0] += signal * l / L
    pel = [{"id": f"p{i:02d}", "text": f"prompt {i}",
            "labels": {"condition": int(y[i])} if labels else {}} for i in range(N)]
    return StateTrajectory(hidden=h.astype(np.float32), tokens=[p["id"] for p in pel],
                           meta={"axis": "pellets", "pellets": pel})


def _permuted(traj, order):
    return StateTrajectory(hidden=traj.hidden[:, order], tokens=[traj.tokens[i] for i in order],
                           meta={**traj.meta, "pellets": [traj.meta["pellets"][i] for i in order]})


def test_muzzle_is_one_point():
    t = _family(signal=2.0)
    for lay in (blast.monitor(t, "condition", null_draws=20), blast.open_layout(t)):
        assert np.all(lay.positions[:, 0] == 0)
        assert np.abs(lay.positions[:, -1]).max() > 0


def test_range_is_the_exact_distance():
    t = _family()
    fr = blast.frame(t.hidden)
    X = t.hidden.astype(np.float64)
    for l in range(L):
        want = np.linalg.norm(X[l] - X[l].mean(0), axis=1) / np.linalg.norm(X[l], axis=1).mean()
        assert np.allclose(fr["range"][:, l], want)


def test_open_never_draws_a_pellet_beyond_its_range():
    t = _family(muzzle=False, seed=3)
    lay = blast.open_layout(t)
    screen = np.linalg.norm(lay.positions.astype(np.float64), axis=2)
    assert (screen <= blast.frame(t.hidden)["range"] + 1e-5).all()
    assert ((0 < lay.arrays["shown"]) & (lay.arrays["shown"] <= 1 + 1e-6)).all()


@pytest.mark.parametrize("method", ["monitor", "open"])
def test_reordering_the_items_cannot_move_the_picture(method):
    t = _family(signal=2.0, seed=1)
    order = np.random.default_rng(9).permutation(N)
    build = ((lambda x: blast.monitor(x, "condition", null_draws=20)) if method == "monitor"
             else blast.open_layout)
    a, b = build(t), build(_permuted(t, order))
    assert np.allclose(a.positions[order], b.positions, atol=1e-5)
    if method == "monitor":
        assert np.allclose(a.arrays["auroc"], b.arrays["auroc"], equal_nan=True)


def test_rotating_hidden_space_cannot_move_the_picture():
    t = _family(signal=2.0, seed=2)
    Q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(D, D)))
    r = StateTrajectory(hidden=(t.hidden.astype(np.float64) @ Q).astype(np.float32),
                        tokens=t.tokens, meta=t.meta)
    a, b = blast.open_layout(t), blast.open_layout(r)
    assert np.allclose(a.positions, b.positions, atol=1e-4)
    m, n = blast.monitor(t, "condition", null_draws=20), blast.monitor(r, "condition", null_draws=20)
    assert np.allclose(m.positions[..., 0], n.positions[..., 0], atol=1e-4)
    # y's sign is pinned in hidden-space coordinates, which a rotation moves
    assert np.allclose(np.abs(m.positions[..., 1]), np.abs(n.positions[..., 1]), atol=1e-4)


def test_monitor_x_is_cross_fitted_not_memorised():
    """On noise, a diff-of-means direction fitted to all pellets splits them
    in-sample (24 points in 64-d); the drawn x must not."""
    t = _family(signal=0.0, seed=5)
    lay = blast.monitor(t, "condition", null_draws=100)
    y = blast.label_vector(t, "condition")
    dev = blast.frame(t.hidden)["dev"][-1]
    d = dev[y == 1].mean(0) - dev[y == 0].mean(0)
    assert blast._auroc(dev @ d, y) > 0.9                          # the trap
    a = lay.arrays["auroc"][-1]
    assert lay.arrays["null05"][-1] <= a <= lay.arrays["null95"][-1]


def test_monitor_finds_a_real_signal():
    lay = blast.monitor(_family(signal=6.0, seed=6), "condition", null_draws=100)
    a, hi = lay.arrays["auroc"], lay.arrays["null95"]
    assert np.isnan(a[0])                                          # the muzzle reads nothing
    assert (a[-2:] > hi[-2:]).all() and a[-1] > 0.9
    x, y = lay.positions[:, -1, 0], np.array([1] * (N // 2) + [0] * (N // 2))
    assert x[y == 1].mean() > x[y == 0].mean()                     # positive class to the right


def _twins(grouped):
    """Contrast pairs with no label signal: each pair shares a large common
    state (one question, two answers) and one of each pair is labelled 1."""
    rng = np.random.default_rng(11)
    q = rng.normal(size=(N // 2, D)) * 3.0
    h = np.zeros((L, N, D))
    for l in range(1, L):
        h[l] = np.repeat(q, 2, axis=0) * l + rng.normal(size=(N, D)) * l * 0.5
    pel = [{"id": f"q{i // 2:02d}{'tf'[i % 2]}", "text": "", "labels": {"false": i % 2},
            **({"group": f"q{i // 2:02d}"} if grouped else {})} for i in range(N)]
    return StateTrajectory(hidden=h.astype(np.float32), tokens=[p["id"] for p in pel],
                           meta={"axis": "pellets", "pellets": pel})


def test_contrast_pairs_are_held_out_together():
    """Scored with its twin still in training, a held-out pellet lands on the
    twin's side: far below chance, a picture of the design, not the model."""
    loose = blast.monitor(_twins(grouped=False), "false", null_draws=100)
    paired = blast.monitor(_twins(grouped=True), "false", null_draws=100)
    assert (loose.arrays["auroc"][1:] < loose.arrays["null05"][1:]).sum() >= 3
    inside = ((paired.arrays["null05"][1:] <= paired.arrays["auroc"][1:])
              & (paired.arrays["auroc"][1:] <= paired.arrays["null95"][1:]))
    assert inside.all() and paired.params["grouped"]


def test_monitor_refuses_without_labels():
    with pytest.raises(ValueError, match="open layout"):
        blast.monitor(_family(labels=False), "condition")
    built, skipped = blast.layouts(_family(labels=False))
    assert [b.method for b in built] == ["open"] and skipped == []


def test_labels_are_strictly_binary():
    t = _family()
    t.meta["pellets"][0]["labels"]["condition"] = "0"     # bool("0") is True
    with pytest.raises(ValueError, match="must be 0, 1"):
        blast.label_vector(t, "condition")


def test_pellets_stack_one_position_per_trajectory():
    trajs = [StateTrajectory(hidden=np.full((3, 4, 5), i, np.float32), tokens=list("abcd"),
                             entropy=np.full((3, 4), i, np.float32), meta={"model": "m"})
             for i in range(3)]
    fam = blast.pellets(trajs, ["x", "y", "z"], labels=[{"c": 1}, {"c": 0}, {}])
    assert fam.hidden.shape == (3, 3, 5) and fam.tokens == ["x", "y", "z"]
    assert np.array_equal(fam.entropy[0], [0, 1, 2])
    assert fam.meta["pellets"][1] == {"id": "y", "text": "", "labels": {"c": 0}}
    with pytest.raises(ValueError, match="unique"):
        blast.pellets(trajs, ["x", "x", "z"])
    trajs[1] = StateTrajectory(hidden=np.zeros((3, 4, 6), np.float32), tokens=list("abcd"))
    with pytest.raises(ValueError, match="one depth and width"):
        blast.pellets(trajs, ["x", "y", "z"])


def test_scene_round_trips(tmp_path):
    t = _family(signal=3.0)
    built, skipped = blast.layouts(t, null_draws=20)
    path = tmp_path / "blast.mtj"
    statefile.save_scene(blast.scene(t, built, skipped), path)
    s = statefile.load_scene(path)
    b = s["blast"]
    assert b["schema"] == "mottled-blast/1"
    assert [x["name"] for x in b["layouts"]] == ["monitor · condition", "open"]
    assert np.array_equal(b["layouts"][0]["positions"], built[0].positions)
    assert np.array_equal(b["layouts"][0]["arrays"]["auroc"], built[0].arrays["auroc"],
                          equal_nan=True)
    assert b["range"].shape == (N, L) and b["spread"].shape == (L,)
    # the run is the pellets at the first layout, so older viewers still draw them
    assert np.allclose(s["runs"][0]["points"][..., :2], built[0].positions)
    assert len(s["meta"]["pellets"]) == N


torch = pytest.importorskip("torch")


def test_capture_positions_matches_a_full_capture():
    import tiny
    from capture import capture

    kw = tiny.mt()
    full = capture(kw["model"], "the capital of france is", tokenizer=kw["tokenizer"])
    last = capture(kw["model"], "the capital of france is", tokenizer=kw["tokenizer"],
                   positions=[-1])
    assert last.hidden.shape == (full.n_layers, 1, full.dim)
    assert np.array_equal(last.hidden[:, 0], full.hidden[:, -1])
    assert np.allclose(last.entropy[:, 0], full.entropy[:, -1], atol=1e-5)
    assert last.tokens == full.tokens[-1:] and last.meta["positions"] == [full.n_tokens - 1]
    with pytest.raises(ValueError, match="cannot be sliced"):
        capture(kw["model"], "the capital", tokenizer=kw["tokenizer"], positions=[-1],
                capture_components=True)


def test_run_blast_end_to_end(tmp_path):
    import tiny
    from pipeline import run_blast

    words = ["the capital of france is", "the capital of germany is", "hello world",
             "the quick brown fox", "the lazy dog sat", "a new state", "one two three",
             "the cat sat on the mat"]
    items = [{"id": f"i{k}", "text": w, "labels": {"c": k % 2}} for k, w in enumerate(words)]
    result = run_blast(items, cfg=blast.BlastConfig(model="tiny", null_draws=10), **tiny.mt())
    assert [b.method for b in result["blast"]["layouts"]] == ["monitor", "open"]
    assert result["analysis"]["config"]["kind"] == "blast"
    statefile.save_scene(result, tmp_path / "b.mtj")
    s = statefile.load_scene(tmp_path / "b.mtj")
    assert s["runs"][0]["points"].shape == (len(items), result["traj"].n_layers, 3)
    assert [p["id"] for p in s["meta"]["pellets"]] == [it["id"] for it in items]
