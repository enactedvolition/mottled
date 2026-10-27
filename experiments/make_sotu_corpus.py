"""Derive experiments/sotu_corpus.jsonl from the @stdlib/datasets-sotu package.

    python -I experiments/make_sotu_corpus.py PACKAGE_DIR experiments/sotu_corpus.jsonl

PACKAGE_DIR is the extracted npm tarball's `package/` directory (see
experiments/README.md for the tarball, its integrity hash and why it was
extracted rather than installed). Only the package's data JSON is read; none
of its code is executed.

Selection is by SHA-256 rather than a seeded RNG, so the rule reads the same
in any language and does not depend on a library's sampling algorithm staying
put between versions:

  - documents: the N_DOCS addresses whose sha256(f"{SEED}:doc:{id}") is
    smallest, `id` being the data file's stem (e.g. 1790_george_washington_n);
  - excerpt: whitespace collapsed to single spaces; an offset of
    sha256(f"{SEED}:offset:{id}") mod (len(text) - SPAN), advanced past the
    next ". " so the excerpt opens on a sentence; SPAN characters from there,
    cut back to the last space.

An excerpt rather than the whole address keeps the committed file small; the
experiment reads only its first few hundred tokens.
"""
import hashlib
import json
import sys
from pathlib import Path

PACKAGE = "@stdlib/datasets-sotu"
VERSION = "0.2.3"
SEED = "mottled-lens-patch-v1"
N_DOCS = 80
SPAN = 1000


def _h(*parts: str) -> int:
    return int(hashlib.sha256(":".join((SEED,) + parts).encode()).hexdigest(), 16)


def excerpt(doc_id: str, text: str) -> tuple[int, str]:
    text = " ".join(text.split())
    offset = _h("offset", doc_id) % (len(text) - SPAN)
    start = text.find(". ", offset)
    # an address with no sentence break after the offset cannot supply a
    # sentence-initial excerpt; failing names it rather than shortening it
    if start < 0 or start + 2 + SPAN > len(text):
        raise ValueError(f"{doc_id}: no sentence start leaves {SPAN} characters")
    chunk = text[start + 2:start + 2 + SPAN]
    return start + 2, chunk[:chunk.rfind(" ")]


def main(package_dir: str, out: str) -> None:
    pkg = Path(package_dir)
    meta = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
    if (meta.get("name"), meta.get("version")) != (PACKAGE, VERSION):
        raise SystemExit(f"expected {PACKAGE}@{VERSION}, found "
                         f"{meta.get('name')}@{meta.get('version')}")
    files = sorted((pkg / "data").glob("*.json"))
    chosen = sorted(sorted(files, key=lambda f: _h("doc", f.stem))[:N_DOCS])
    lines = []
    for f in chosen:
        doc = json.loads(f.read_text(encoding="utf-8"))
        offset, text = excerpt(f.stem, doc["text"])
        lines.append(json.dumps({"id": f.stem, "year": doc["year"], "name": doc["name"],
                                 "party": doc["party"], "offset": offset, "text": text},
                                ensure_ascii=False, sort_keys=True))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    Path(out).write_bytes(data)
    print(f"{len(lines)} of {len(files)} documents, {len(data)} bytes, "
          f"sha256 {hashlib.sha256(data).hexdigest()}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.split("\n\n")[1])
    main(sys.argv[1], sys.argv[2])
