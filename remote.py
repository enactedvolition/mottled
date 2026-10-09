"""Weights that live in cloud storage, one layer on local disk at a time.

`stream.py` removed the RAM ceiling: only the block that is currently running
needs to exist. This module removes the *disk* ceiling the same way. The
checkpoint stays where it was published; each block is fetched immediately
before it runs, written to a local cache file, read, and deleted. Peak local
disk becomes one layer, not the whole model.

    from remote import RemoteWeights
    from stream import stream_capture

    traj = stream_capture("hf://moonshotai/Kimi-K2-Instruct", "The capital of")

The mechanism is HTTP range requests against the published `.safetensors`
files. That matters more than it sounds: shard boundaries are *not* layer
boundaries. Qwen3-30B-A3B packs up to five layers per shard and splits one
layer across two of them, so "download the shard you need" can cost five
times what the layer costs. Reading byte ranges out of the shards makes the
fetch granularity the tensor, not the file, whatever the publisher chose.

What is fetched is written out as a real, small safetensors file — the same
format, just this layer — so the bytes on disk are inspectable, reusable,
and read back by the same library that reads any other checkpoint.

Three things this module refuses to do quietly:

- **Fall back to whole-file downloads.** If a server ignores `Range` it
  answers `200` with the entire file. At frontier scale that is a terabyte
  arriving where seventeen gigabytes were asked for, so it raises instead.
- **Hide what it spent.** Every trajectory captured this way carries
  `bytes_fetched`, `requests_made` and `peak_cache_bytes`. Egress is the
  real cost of this design and it should be on the receipt.
- **Pretend the cache is free.** `cache_bytes` is a hard budget, and the
  peak is measured rather than assumed.
"""

from __future__ import annotations

import json
import os
import struct
import time
import warnings
from pathlib import Path

_HF = "https://huggingface.co"
_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "vocab.json", "merges.txt")


class RangeUnsupported(RuntimeError):
    """The server answered a range request with the whole file."""


def _resolve_base(source: str, revision: str) -> str:
    """Turn `hf://org/model`, `org/model` or a bare URL into a base URL."""
    s = str(source)
    if s.startswith("hf://"):
        s = s[len("hf://"):]
    elif s.startswith(("http://", "https://")):
        return s.rstrip("/") + "/"
    if s.count("/") != 1 or not all(part for part in s.split("/")):
        raise ValueError(
            f"cannot tell what {source!r} is. Expected a local directory, a "
            f"hub repo id (`org/model` or `hf://org/model`), or a base URL.")
    return f"{_HF}/{s}/resolve/{revision}/"


class RemoteWeights:
    """A `ShardIndex` whose bytes are somewhere else.

    Exposes the one method `stream.StreamedBlocks` actually uses —
    `prefixed(prefix)` — so it drops in wherever a local checkpoint went,
    and the streaming machinery above it does not change at all.
    """

    def __init__(self, source, cache_dir=None, cache_bytes: int | None = None,
                 revision: str = "main", token: str | None = None,
                 coalesce_gap: int = 1 << 20, timeout: int = 120,
                 retries: int = 4):
        import requests

        self.base = _resolve_base(source, revision)
        self.source = str(source)
        if cache_dir:
            self.cache = Path(cache_dir)
        else:
            import cache as cache_mod

            self.cache = cache_mod.default_dir("remote")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cache_bytes = cache_bytes
        self.coalesce_gap = int(coalesce_gap)
        self.timeout, self.retries = timeout, max(1, int(retries))

        self.session = requests.Session()
        token = token or os.environ.get("HF_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

        # receipt
        self.bytes_fetched = 0
        self.requests_made = 0
        self.peak_cache_bytes = 0
        self.evictions_failed = 0
        self.layer_bytes: dict[str, int] = {}

        self._headers: dict[str, tuple[dict, int]] = {}
        self._files: list[Path] = []          # LRU of cached layer files
        # The hub commit the files resolved to, when the host says
        # (X-Repo-Commit). Every cache file is keyed by it plus the base URL
        # (which carries repo and revision): one cache directory serves many
        # repos, and a file name alone ("model.safetensors", a layer prefix)
        # names the same thing in every one of them.
        self.commit: str | None = None
        self._key: str | None = None
        self.map = self._load_index()

    # ------------------------------------------------------------ transport
    def _get(self, filename: str, start: int | None = None,
             end: int | None = None, sink=None):
        """One request. `[start, end)` is a half-open byte range.

        With `sink` the body is streamed to that file object and never fully
        held in memory — which is what makes a 17 GB layer a disk cost rather
        than a RAM cost. Without it the bytes are returned.
        """
        import requests

        url = self.base + filename
        headers = {}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end - 1}"
        want = None if start is None else end - start

        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, headers=headers, timeout=self.timeout,
                                     stream=sink is not None)
                if r.status_code == 404:
                    r.close()
                    return None
                r.raise_for_status()
                if want is not None and r.status_code != 206:
                    r.close()
                    raise RangeUnsupported(
                        f"{url} answered {r.status_code} to a range request, "
                        f"meaning it is sending the whole file where "
                        f"{want / 1e9:.2f} GB was asked for. Streaming a "
                        f"checkpoint from a host that ignores Range is not a "
                        f"slow path, it is a different order of cost — "
                        f"refusing rather than silently downloading it.")
                sent = r.headers.get("Content-Length")
                if want is not None and sent is not None and int(sent) != want:
                    r.close()
                    raise RuntimeError(f"{url} announced {sent} bytes for a "
                                       f"{want}-byte range")
                self.requests_made += 1
                commit = r.headers.get("X-Repo-Commit")
                if commit and self.commit is None:
                    self.commit = commit
                if sink is None:
                    body = r.content
                    self.bytes_fetched += len(body)
                    return body
                got = 0
                for chunk in r.iter_content(chunk_size=1 << 22):
                    sink.write(chunk)
                    got += len(chunk)
                self.bytes_fetched += got
                if want is not None and got != want:
                    # a 206 of the wrong size is the same lie as a 200 of the
                    # whole file, and what it wrote must not become a cache
                    # file: raising here keeps the caller from renaming it
                    raise RuntimeError(f"{url} sent {got} bytes for a "
                                       f"{want}-byte range")
                return got
            except RangeUnsupported:
                raise
            except requests.RequestException as e:      # network only
                last = e
                if attempt == self.retries - 1:
                    break
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"could not fetch {url}: {last}")

    def _load_index(self) -> dict[str, str]:
        """tensor name -> shard filename, without downloading any weights."""
        body = self._get(_INDEX)
        if body is not None:
            return dict(json.loads(body)["weight_map"])
        header, _ = self._header(_SINGLE)
        return {k: _SINGLE for k in header if k != "__metadata__"}

    @property
    def key(self) -> str:
        """Cache-file prefix for this source: a hash of the base URL (repo +
        revision) and, when known, the commit it resolved to. Fixed at first
        use, so files written in one session stay addressable in it."""
        if self._key is None:
            import hashlib

            ident = f"{self.base}@{self.commit or ''}"
            self._key = hashlib.sha256(ident.encode()).hexdigest()[:16]
        return self._key

    def _header(self, filename: str) -> tuple[dict, int]:
        """The safetensors header, cached. Two small requests, once per shard.

        A safetensors file is an 8-byte little-endian header length, that many
        bytes of JSON naming every tensor and its byte span, then the data.
        Reading the header is how a byte range gets computed without the file.
        """
        if filename in self._headers:
            return self._headers[filename]
        cached = self.cache / f"{self.key}-{filename.replace('/', '_')}.header.json"
        if cached.exists():
            blob = json.loads(cached.read_text())
            got = (blob["header"], blob["offset"])
        else:
            size = struct.unpack("<Q", self._get(filename, 0, 8))[0]
            header = json.loads(self._get(filename, 8, 8 + size))
            got = (header, 8 + size)
            cached.write_text(json.dumps({"header": header, "offset": 8 + size}))
        self._headers[filename] = got
        return got

    # ------------------------------------------------------------ the cache
    def _evict(self, keeping: Path | None = None) -> None:
        if self.cache_bytes is None:
            budget = 0                       # default: one layer, then gone
        else:
            budget = int(self.cache_bytes)
        while self._files:
            total = sum(p.stat().st_size for p in self._files if p.exists())
            if total <= budget:
                break
            victim = self._files.pop(0)
            if victim == keeping:
                self._files.append(victim)   # never evict what we just wrote
                break
            try:
                victim.unlink(missing_ok=True)
            except OSError:
                # A mapped file cannot be unlinked on every platform. Keep it
                # and say so rather than pretending the budget held: the
                # bound is why this module exists, so breaking it must show.
                self._files.append(victim)
                self.evictions_failed += 1
                warnings.warn(f"could not delete {victim}; the local disk "
                              f"bound is not holding", RuntimeWarning)
                break

    def _remember(self, path: Path) -> None:
        self._files.append(path)
        self.peak_cache_bytes = max(
            self.peak_cache_bytes,
            sum(p.stat().st_size for p in self._files if p.exists()))

    # ------------------------------------------------------------- fetching
    @staticmethod
    def _coalesce(spans, gap):
        """Merge byte spans separated by less than `gap`.

        Within a shard a layer's tensors are laid out contiguously — all 2327
        of Kimi K2's layer-5 tensors sit end to end — so a layer that would
        otherwise be thousands of requests collapses to one.
        """
        spans = sorted(spans)
        out = [list(spans[0])]
        for start, end in spans[1:]:
            if start - out[-1][1] <= gap:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        return [tuple(s) for s in out]

    def prefixed(self, prefix: str) -> dict:
        """Every tensor whose name starts with `prefix`, keyed without it.

        Fetches exactly the byte ranges those tensors occupy, writes them to
        one local safetensors file, reads it, and lets the cache evict it.
        """
        from safetensors import safe_open

        self._evict()
        wanted = sorted(k for k in self.map if k.startswith(prefix))
        if not wanted:
            return {}

        by_file: dict[str, list[str]] = {}
        for k in wanted:
            by_file.setdefault(self.map[k], []).append(k)

        # Lay the tensors out afresh, in name order, in the local file.
        header, cursor = {}, 0
        plan = []                                   # (file, src_start, src_end, dst)
        for filename, keys in by_file.items():
            src_header, data_start = self._header(filename)
            spans = []
            for k in keys:
                meta = src_header[k]
                b, e = meta["data_offsets"]
                header[k] = {"dtype": meta["dtype"], "shape": meta["shape"],
                             "data_offsets": [cursor, cursor + (e - b)]}
                spans.append((data_start + b, data_start + e, cursor))
                cursor += e - b
            plan.append((filename, spans))

        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * (-(len(blob) + 8) % 8)        # keep the data 8-aligned
        path = self.cache / f"{self.key}-{prefix.strip('.').replace('.', '_') or 'root'}.safetensors"
        base = 8 + len(blob)

        if not path.exists():
            tmp = path.with_suffix(".partial")
            with open(tmp, "wb") as fh:
                fh.write(struct.pack("<Q", len(blob)))
                fh.write(blob)
                fh.truncate(base + cursor)
                for filename, spans in plan:
                    merged = self._coalesce([(s, e) for s, e, _ in spans],
                                            self.coalesce_gap)
                    for start, end in merged:
                        inside = [sp for sp in spans
                                  if start <= sp[0] and sp[1] <= end]
                        self._get(filename, start, end,
                                  sink=_Scatter(fh, base, start, inside))
            tmp.replace(path)
        self.layer_bytes[prefix] = cursor
        self._remember(path)

        out = {}
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                out[k[len(prefix):]] = f.get_tensor(k)
        return out

    def sizes(self, prefix: str) -> dict[str, int]:
        """Byte size of every tensor under `prefix`, without fetching any.

        Used to price a fetch before making it — notably to ask how much of
        an MoE layer is experts no token will route to.
        """
        out = {}
        for k in self.map:
            if not k.startswith(prefix):
                continue
            meta = self._header(self.map[k])[0][k]
            b, e = meta["data_offsets"]
            out[k[len(prefix):]] = e - b
        return out

    def fetch_config(self) -> Path:
        """Pull the small metadata files into a directory `transformers` can
        load from, so a config and tokenizer cost kilobytes, not weights."""
        meta = self.cache / f"meta-{self.key}"
        meta.mkdir(exist_ok=True)
        body = self._get("config.json")
        if body is None:
            raise FileNotFoundError(f"no config.json under {self.base}")
        (meta / "config.json").write_bytes(body)
        for name in _TOKENIZER_FILES:
            body = self._get(name)
            if body is not None:
                (meta / name).write_bytes(body)
        return meta

    def release(self) -> None:
        """Drop every cached layer file. Called when a pass is over, so the
        last layer is not left behind — eviction otherwise runs one fetch
        late, since a file must outlive the read that mapped it."""
        self.cache_bytes = None
        keep = []
        for path in self._files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                keep.append(path)
        self._files = keep

    def receipt(self) -> dict:
        """What this cost, for the trajectory's `meta`."""
        return {
            "source": self.source,
            "bytes_fetched": self.bytes_fetched,
            "requests_made": self.requests_made,
            "peak_cache_bytes": self.peak_cache_bytes,
            "evictions_failed": self.evictions_failed,
            "cache_dir": str(self.cache),
        }

    def __len__(self) -> int:
        return len(self.map)


class _Scatter:
    """Sink for one coalesced range: each wanted tensor's bytes go to its
    place in the local file, and the bytes between tensors are dropped.

    A coalesced range arrives in source order, gaps included, while the local
    file is packed in name order. Written as one block, the two only agree
    when a layer sits on disk in name order with nothing between its tensors,
    and safetensors lays a file out by dtype before name: a layer that mixes
    dtypes came back with its bytes under the wrong names."""

    def __init__(self, fh, base: int, start: int, spans):
        self.fh, self.base, self.at = fh, base, start
        self.spans = sorted(spans)             # (src_start, src_end, dst)
        self.i = 0

    def write(self, chunk: bytes) -> None:
        off = 0
        while off < len(chunk) and self.i < len(self.spans):
            s, e, dst = self.spans[self.i]
            at = self.at + off
            if at >= e:
                self.i += 1
            elif at < s:                           # between tensors
                off += min(s - at, len(chunk) - off)
            else:
                n = min(e - at, len(chunk) - off)
                self.fh.seek(self.base + dst + (at - s))
                self.fh.write(chunk[off:off + n])
                off += n
        self.at += len(chunk)


def unrouted_expert_bytes(weights: RemoteWeights, routing, prefix="model.layers.") -> dict:
    """How much of what was fetched went to experts nothing routed to.

    A sparse layer's bytes are almost entirely experts, and a short prompt
    touches a small fraction of them. This measures that fraction *after* the
    fact, from the routing the capture already recorded — so the case for
    fetching only the routed experts is a number rather than an intuition.

    It is deliberately only a measurement. Acting on it means predicting the
    route before the layer runs, which is a different and much more delicate
    change; see the roadmap.
    """
    if routing is None:
        return {}
    total = used = 0
    for row, layer in enumerate(routing.layers.tolist()):
        sizes = weights.sizes(f"{prefix}{layer}.")
        live = {int(e) for e in routing.experts[row].ravel().tolist()}
        for name, n in sizes.items():
            if ".experts." not in name:
                continue
            tail = name.split(".experts.")[1].split(".")[0]
            if not tail.isdigit():
                return {}                    # fused layout: not separable
            total += n
            if int(tail) in live:
                used += n
    if not total:
        return {}
    return {"expert_bytes": total, "routed_expert_bytes": used,
            "unrouted_fraction": round(1 - used / total, 4)}
