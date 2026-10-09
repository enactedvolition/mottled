"""Weights that stay in cloud storage, one layer on local disk at a time.

These tests run against a real HTTP server — a range-capable one started in
this process over a temp directory — so the range arithmetic, the safetensors
header parsing and the cache eviction are exercised for real, offline.

The claims being pinned are the ones that would be expensive to get wrong:
a remote pass equals a local pass exactly; local disk never exceeds one
layer; a server that ignores `Range` is refused rather than allowed to send
a terabyte; and a batch of prompts costs one download, not one per prompt.
"""
import http.server
import json
import re
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("requests")

from capture import capture  # noqa: E402
from remote import RangeUnsupported, RemoteWeights, unrouted_expert_bytes  # noqa: E402
from stream import stream_capture, stream_capture_batch  # noqa: E402
from tests.test_stream import DummyTokenizer, _dense, _moe  # noqa: E402

PROMPT = "the capital of france is"


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """`SimpleHTTPRequestHandler` does not do ranges; this one does."""

    ignore_ranges = False
    short = None            # "announced" / "silent": each range one byte short
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        try:
            body = open(path, "rb").read()
        except OSError:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        type(self).hits.append((self.path, rng))
        if rng and not self.ignore_ranges:
            first, last = re.match(r"bytes=(\d+)-(\d+)", rng).groups()
            first, last = int(first), min(int(last), len(body) - 1)
            chunk = body[first:last + 1]
            if self.short:
                chunk = chunk[:-1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            if self.short != "silent":
                self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def serve(tmp_path):
    """Serve a directory over HTTP and hand back its base URL."""
    servers = []

    def start(directory, ignore_ranges=False, short=None):
        handler = type("H", (_RangeHandler,),
                       {"ignore_ranges": ignore_ranges, "short": short, "hits": []})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        srv.RequestHandlerClass = lambda *a, **k: handler(
            *a, directory=str(directory), **k)
        srv.RequestHandlerClass.hits = handler.hits
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/", handler

    yield start
    for srv in servers:
        srv.shutdown()


# --------------------------------------------------------------- the claim
def test_remote_capture_is_exactly_the_in_memory_capture(tmp_path, serve):
    """Where the bytes live must not change what the model computes."""
    model, path = _dense(tmp_path)
    base, _ = serve(path)
    mem = capture(model, PROMPT, tokenizer=DummyTokenizer())
    got = stream_capture(base, PROMPT, tokenizer=DummyTokenizer(),
                         cache_dir=tmp_path / "cache")

    np.testing.assert_array_equal(got.hidden, mem.hidden)
    np.testing.assert_array_equal(got.logits, mem.logits)
    assert got.tokens == mem.tokens
    assert got.meta["remote"]["bytes_fetched"] > 0


def test_only_one_layer_is_ever_on_local_disk(tmp_path, serve):
    """The disk bound is the reason this module exists; measure it."""
    _, path = _dense(tmp_path, layers=6)
    base, _ = serve(path)
    cache = tmp_path / "cache"
    traj = stream_capture(base, PROMPT, tokenizer=DummyTokenizer(), cache_dir=cache)

    whole = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    peak = traj.meta["remote"]["peak_cache_bytes"]
    assert 0 < peak < whole / 3          # six layers; one resident at a time
    # and nothing is left behind: every layer file was deleted after its block
    assert [p.name for p in cache.rglob("*layers*.safetensors")] == []


def test_a_server_that_ignores_range_is_refused(tmp_path, serve):
    """Answering 200 to a range request means shipping the whole file. At
    frontier scale that is a terabyte where a layer was asked for, so it must
    fail rather than quietly cost that."""
    _, path = _dense(tmp_path, layers=2)
    base, _ = serve(path, ignore_ranges=True)
    with pytest.raises(RangeUnsupported, match="whole file"):
        stream_capture(base, PROMPT, tokenizer=DummyTokenizer(),
                       cache_dir=tmp_path / "cache")


def test_only_the_layers_bytes_are_fetched(tmp_path, serve):
    """Range requests, not whole shards: shard boundaries are not layer
    boundaries, so fetching by file can cost several times the layer."""
    _, path = _dense(tmp_path, layers=6)
    base, _ = serve(path)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    before = weights.bytes_fetched
    weights.prefixed("model.layers.3.")
    layer = weights.bytes_fetched - before

    whole = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    assert 0 < layer < whole / 4
    # a layer's tensors are contiguous, so it should be very few requests
    assert weights.requests_made < 8


def test_the_shard_index_costs_no_weights(tmp_path, serve):
    """Deciding what to fetch must not require fetching it."""
    _, path = _dense(tmp_path, layers=4)
    base, _ = serve(path)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    assert len(weights) > 0
    whole = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    assert weights.bytes_fetched < whole / 5
    sizes = weights.sizes("model.layers.2.")
    assert sizes and all(n > 0 for n in sizes.values())


def test_unknown_source_is_named_not_guessed(tmp_path):
    with pytest.raises(ValueError, match="local directory|repo id"):
        stream_capture("not a checkpoint at all", PROMPT,
                       tokenizer=DummyTokenizer())


# ------------------------------------------------------------- the economics
def test_a_batch_costs_one_pass_over_the_weights(tmp_path, serve):
    """Egress is the binding cost, so prompts share a pass over the blocks."""
    _, path = _dense(tmp_path, layers=6)
    base, _ = serve(path)
    prompts = ["the capital of france is", "a much longer prompt about geese",
               "short one"]

    one = RemoteWeights(base, cache_dir=tmp_path / "c1")
    one.prefixed("model.layers.0.")
    per_layer = one.bytes_fetched

    trajs = stream_capture_batch(base, prompts, tokenizer=DummyTokenizer(),
                                 cache_dir=tmp_path / "c2")
    assert len(trajs) == 3
    assert trajs[0].meta["block_loads"] == 6           # not 18
    # the whole batch costs about what six layers cost, not eighteen
    assert trajs[0].meta["remote"]["bytes_fetched"] < per_layer * 6 * 1.5


def test_batched_prompts_match_individual_ones(tmp_path):
    """Padding is where batching goes wrong silently: the pads must not reach
    the real tokens, and the positions must survive left-padding.

    The bound is round-off, not zero, and deliberately so: batching changes
    every matmul's shape and therefore its summation order, so the last bits
    move by an amount that depends on the machine's BLAS. It is tight enough
    that a pad actually reaching a real token would break it, and loose enough
    not to encode one CPU's rounding as a promise.
    """
    model, path = _dense(tmp_path, layers=4)
    prompts = ["the capital of france is", "geese", "two words here"]
    tok = DummyTokenizer()
    alone = [capture(model, p, tokenizer=tok) for p in prompts]
    together = stream_capture_batch(path, prompts, tokenizer=DummyTokenizer())

    for solo, batched in zip(alone, together):
        assert solo.tokens == batched.tokens
        assert solo.hidden.shape == batched.hidden.shape
        np.testing.assert_allclose(batched.hidden, solo.hidden, atol=1e-6)
        assert batched.meta["batch"] == 3


def test_batched_routing_is_split_per_prompt(tmp_path):
    model, _ = _moe(tmp_path)
    prompts = ["the capital of france is", "geese"]
    tok = DummyTokenizer()
    alone = [capture(model, p, tokenizer=tok, capture_routing=True) for p in prompts]
    _, path = _moe(tmp_path)
    together = stream_capture_batch(path, prompts, tokenizer=DummyTokenizer(),
                                    capture_routing=True)

    for solo, batched in zip(alone, together):
        batched.validate()
        assert batched.routing.experts.shape == solo.routing.experts.shape
        np.testing.assert_array_equal(batched.routing.experts, solo.routing.experts)


# ------------------------------------------------------- pricing the next step
def test_unrouted_expert_bytes_measures_the_waste(tmp_path, serve):
    """A sparse layer is nearly all experts and a short prompt touches few of
    them. This is the measurement that decides whether prefetching only the
    routed experts is worth its risk."""
    _, path = _moe(tmp_path, layers=4, n_experts=8, k=2)
    base, _ = serve(path)
    traj = stream_capture(base, PROMPT, tokenizer=DummyTokenizer(),
                          capture_routing=True, cache_dir=tmp_path / "cache")
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    report = unrouted_expert_bytes(weights, traj.routing)

    assert report["expert_bytes"] > 0
    assert 0.0 <= report["unrouted_fraction"] < 1.0
    assert report["routed_expert_bytes"] <= report["expert_bytes"]


def test_no_routing_prices_nothing(tmp_path, serve):
    _, path = _dense(tmp_path, layers=2)
    base, _ = serve(path)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    assert unrouted_expert_bytes(weights, None) == {}


def test_header_is_read_once_per_shard(tmp_path, serve):
    """Headers are cached on disk: re-opening a checkpoint should not re-read
    a megabyte of JSON per layer."""
    _, path = _dense(tmp_path, layers=4)
    base, handler = serve(path)
    cache = tmp_path / "cache"
    RemoteWeights(base, cache_dir=cache).prefixed("model.layers.0.")
    first = len(handler.hits)
    handler.hits.clear()
    RemoteWeights(base, cache_dir=cache).prefixed("model.layers.1.")
    assert len(handler.hits) < first
    assert list(cache.glob("*.header.json"))


def test_single_file_checkpoints_need_no_index(tmp_path, serve):
    """Small checkpoints ship one `model.safetensors` and no index; the
    tensor list then has to come from the header itself."""
    _, path = _dense(tmp_path, layers=2)
    assert (path / "model.safetensors").exists()
    assert not (path / "model.safetensors.index.json").exists()
    base, _ = serve(path)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    assert any(k.startswith("model.layers.1.") for k in weights.map)
    assert json.loads((path / "config.json").read_text())["num_hidden_layers"] == 2


@pytest.mark.parametrize("layers", [1, 2])
def test_a_layer_comes_back_whatever_its_order_on_disk(tmp_path, serve, layers):
    """safetensors lays a file out by dtype before name, so a layer that mixes
    dtypes (FP8 weights beside F32 scales, say) is not in name order on disk,
    and with a second layer in the file it is not contiguous either: the other
    layer's tensors sit between its pieces. Every tensor must still come back
    as the bytes it was — out of order alone, the failure was silent."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors = {}
    for layer in range(layers):
        tensors[f"model.layers.{layer}.a"] = torch.randn(16)
        tensors[f"model.layers.{layer}.b"] = torch.randn(16).to(torch.bfloat16)
        tensors[f"model.layers.{layer}.c"] = torch.randn(16)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file(tensors, str(ckpt / "model.safetensors"))
    base, _ = serve(ckpt)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    for layer in range(layers):
        got = weights.prefixed(f"model.layers.{layer}.")
        for name in "abc":
            assert torch.equal(got[name], tensors[f"model.layers.{layer}.{name}"])


@pytest.mark.parametrize("short", ["announced", "silent"])
def test_a_short_range_is_refused_not_cached(tmp_path, serve, short):
    """A 206 of the wrong size is the same lie as a 200 of the whole file:
    what arrived is not the range asked for, and it must not become a cache
    file the next pass trusts."""
    _, path = _dense(tmp_path, layers=2)
    cache = tmp_path / "cache"
    honest, _ = serve(path)
    weights = RemoteWeights(honest, cache_dir=cache)  # header read honestly
    lying, _ = serve(path, short=short)
    # the same checkpoint, now answering from a host that lies about ranges.
    # (A second RemoteWeights pointed at `lying` would not reuse the honest
    # header any more: cache files are keyed by source, so a header from one
    # URL is never trusted for another.)
    weights.base = lying
    with pytest.raises(RuntimeError, match="byte range"):
        weights.prefixed("model.layers.0.")
    assert not list(cache.glob("*.safetensors"))


def test_a_failed_eviction_is_visible(tmp_path, serve, monkeypatch):
    """The disk bound is why this module exists. A cache file that cannot be
    deleted (a mapped file, on some platforms) breaks it, and a broken bound
    has to show: in the receipt a trajectory carries, and as it happens."""
    import pathlib

    _, path = _dense(tmp_path, layers=2)
    base, _ = serve(path)
    weights = RemoteWeights(base, cache_dir=tmp_path / "cache")
    unlink = pathlib.Path.unlink

    def mapped(self, missing_ok=False):
        if self.suffix == ".safetensors":
            raise OSError("file is mapped")
        return unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", mapped)
    weights.prefixed("model.layers.0.")
    with pytest.warns(RuntimeWarning, match="disk"):
        weights.prefixed("model.layers.1.")
    assert weights.receipt()["evictions_failed"] == 1


@pytest.mark.parametrize("chunk", [1, 3, 7, 40])
def test_scatter_routes_bytes_whatever_the_chunking(chunk):
    """The network hands a range over in arbitrary chunks; where a tensor's
    bytes land must not depend on where a chunk happened to end."""
    import io

    from remote import _Scatter

    src = bytes(range(40))                                   # offsets 100..139
    spans = [(100, 108, 18), (112, 120, 0), (130, 140, 8)]   # (src, end, dst)
    fh = io.BytesIO(bytes(26))
    sink = _Scatter(fh, 0, 100, spans)
    for i in range(0, len(src), chunk):
        sink.write(src[i:i + chunk])
    assert fh.getvalue() == src[12:20] + src[30:40] + src[0:8]


def test_one_cache_directory_serves_many_sources(tmp_path, serve):
    """Cache files are keyed by source (base URL = repo + revision, plus the
    commit when the host names it), not by file name: two checkpoints that
    both call their weights `model.safetensors` must never read each other's
    header or layer files out of a shared cache."""
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, path_a = _dense(tmp_path / "a", layers=2)
    model_b, path_b = _dense(tmp_path / "b", layers=3)
    base_a, _ = serve(path_a)
    base_b, _ = serve(path_b)
    cache = tmp_path / "shared"

    a = RemoteWeights(base_a, cache_dir=cache)
    a.cache_bytes = 1 << 30                      # keep A's layer file around
    a.prefixed("model.layers.0.")
    b = RemoteWeights(base_b, cache_dir=cache)
    assert a.key != b.key
    assert any(k.startswith("model.layers.2.") for k in b.map)   # B's own header
    got = b.prefixed("model.layers.1.")
    want = model_b.model.layers[1].state_dict()
    for name, tensor in got.items():
        assert torch.equal(tensor, want[name]), name
    # the same source again reuses what it cached
    assert RemoteWeights(base_a, cache_dir=cache).key == a.key


def test_the_cache_key_follows_the_commit(tmp_path, serve):
    _, path = _dense(tmp_path, layers=2)
    base, _ = serve(path)
    one = RemoteWeights(base, cache_dir=tmp_path / "c")
    two = RemoteWeights(base, cache_dir=tmp_path / "c")
    two.commit, two._key = "deadbeef" * 5, None   # the hub moved `main`
    assert one.key != two.key
