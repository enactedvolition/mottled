import numpy as np

from cache import DiskCache, make_key


def test_key_stability_and_sensitivity():
    a = make_key("prompt", model="tiny", k=5)
    assert a == make_key("prompt", model="tiny", k=5)
    assert a != make_key("prompt", model="tiny", k=6)
    assert len(a) == 32


def test_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    key = make_key("x")
    assert key not in cache
    payload = {"arr": np.arange(6).reshape(2, 3), "meta": {"k": 1}}
    cache.put(key, payload)
    assert key in cache
    out = cache.get(key)
    assert np.array_equal(out["arr"], payload["arr"]) and out["meta"] == {"k": 1}
    cache.clear()
    assert key not in cache


def test_corrupt_entry_dropped(tmp_path):
    cache = DiskCache(tmp_path)
    key = make_key("bad")
    (tmp_path / f"{key}.pkl").write_bytes(b"not a pickle")
    assert cache.get(key, default="fallback") == "fallback"
    assert key not in cache


# ------------------------------------------------- key completeness, location
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

import pipeline  # noqa: E402
from config import MarbleConfig  # noqa: E402


@pytest.mark.parametrize("field, value", [
    ("dtype", "float16"), ("device", "cpu"), ("keep_logits", False),
    ("grid_padding", 0.3), ("marble_lift", 0.1), ("top_k", 7), ("seed", 3),
    ("n_neighbors", 9), ("projection", "umap"),
])
def test_every_knob_that_can_move_a_number_is_in_the_key(field, value):
    cfg = MarbleConfig(model="gpt2")
    assert pipeline.cache_key("t", cfg, "p") != pipeline.cache_key(
        "t", replace(cfg, **{field: value}), "p")


def test_where_the_cache_lives_is_not_in_the_key():
    cfg = MarbleConfig(model="gpt2")
    assert pipeline.cache_key("t", cfg, "p") == pipeline.cache_key(
        "t", replace(cfg, cache_dir="/elsewhere", use_cache=False, frame_ms=1), "p")


def test_the_key_moves_with_mottled_and_the_model_revision(monkeypatch, tmp_path):
    cfg = MarbleConfig(model=str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    before = pipeline.cache_key("t", cfg, "p")
    monkeypatch.setattr(pipeline.provenance_mod, "_version", lambda: "99.0")
    assert pipeline.cache_key("t", cfg, "p") != before
    monkeypatch.undo()
    import os
    os.utime(tmp_path / "config.json", ns=(1, 1))      # the checkpoint changed
    assert pipeline.cache_key("t", cfg, "p") != before


def test_model_revision_of_an_in_memory_model():
    class Cfg:
        _commit_hash = "abc123"

    class M:
        config = Cfg()

    assert pipeline.model_revision("gpt2", M()) == "abc123"


def test_default_cache_dir_is_per_user_and_overridable(monkeypatch, tmp_path):
    import cache

    monkeypatch.setenv("MOTTLED_CACHE_DIR", str(tmp_path / "x"))
    assert cache.default_dir("pipeline") == tmp_path / "x" / "pipeline"
    monkeypatch.delenv("MOTTLED_CACHE_DIR")
    monkeypatch.delenv("MOTTLED_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.default_dir() == tmp_path / "xdg" / "mottled"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert Path.cwd() not in cache.default_dir().parents
    assert DiskCache().dir.name == "pipeline"

