"""The `.mtj` manifest schemas, and the validator behind `mottled validate`.

One source of truth: the schemas below are plain Python dicts in JSON Schema
(draft 2020-12) form. `python -m codegen` writes them to `docs/schema/` for
other languages and tools, and `tests/test_codegen.py` fails if the
published files drift from this module — so the files are generated, never
hand-edited.

The validator here needs no third-party package. It implements the subset of
JSON Schema these schemas use, plus the checks a schema cannot express: the
container's byte layout, strict JSON (no NaN), that every array a manifest
names exists and fits the blob, and that the shapes agree with each other.

    problems = validate_bytes(Path("scene.mtj").read_bytes())
    [p for p in problems if p.level == "error"]   # empty: the file is valid

Array references are marked in the schema with the annotation keyword
``"x-mtj-array": true`` (a JSON Schema validator ignores unknown keywords);
that is how the validator knows which strings must name an array.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass

SCENE_SCHEMA = "mottled-scene/1"
TRAJECTORY_SCHEMA = "mottled-trajectory/1"
ANALYSIS_PATTERN = r"^mottled-analysis/[0-9]+$"
DIALECT = "https://json-schema.org/draft/2020-12/schema"
ID_BASE = "https://raw.githubusercontent.com/enactedvolition/mottled/main/docs/schema/"

MAGIC = b"MTRJ"
VERSION = 1
ALIGN = 16
ITEMSIZE = {"float16": 2, "float32": 4, "int32": 4}

# --------------------------------------------------------------- the schemas
_REF = {"type": "string", "minLength": 1, "x-mtj-array": True,
        "description": "name of an entry in `arrays`"}
_NULLABLE_NUMBER = {"type": ["number", "null"]}
_STRINGS = {"type": "array", "items": {"type": "string"}}

_DEFS = {
    "arrayRef": {
        "type": "object",
        "description": "where one typed array lives in the blob",
        "required": ["dtype", "shape", "offset", "length"],
        "properties": {
            "dtype": {"type": "string",
                      "description": "float16 | float32 | int32; readers skip "
                                     "arrays of a dtype they do not know"},
            "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "offset": {"type": "integer", "minimum": 0,
                       "description": "byte offset from the start of the blob; "
                                      "a multiple of 16"},
            "length": {"type": "integer", "minimum": 0,
                       "description": "byte length = product(shape) x item size"},
        },
    },
    "arrays": {"type": "object", "additionalProperties": {"$ref": "#/$defs/arrayRef"}},
    "topk": {
        "type": "array", "description": "[layer][token][rank] = [token, probability]",
        "items": {"type": "array", "items": {"type": "array", "items": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "prefixItems": [{"type": "string"}, _NULLABLE_NUMBER]}}},
    },
    "inspector": {
        "type": "object",
        "properties": {"tokens": _STRINGS, "idx": _REF, "sim": _REF,
                       "component_shares": _REF},
    },
    "analysis": {
        "type": "object",
        "description": "the analysis record (provenance.record): what produced the file",
        "required": ["schema", "created", "mottled", "config"],
        "properties": {
            "schema": {"type": "string", "pattern": ANALYSIS_PATTERN},
            "created": {"type": "string"},
            "mottled": {"type": ["string", "null"]},
            "config": {"type": "object"},
            "prompts": _STRINGS,
        },
    },
}


def _header(kind: str, schema_id: str) -> dict:
    return {
        "format": {"const": "mottled-trajectory",
                   "description": "the container family (both kinds share it)"},
        "schema": {"const": schema_id,
                   "description": "this schema's id; absent in files written "
                                  "before schema ids existed"},
        "version": {"const": VERSION},
        "kind": {"const": kind},
        "meta": {"type": "object"},
        "analysis": {"$ref": "#/$defs/analysis"},
        "arrays": {"$ref": "#/$defs/arrays"},
    }


def scene_schema() -> dict:
    run = {
        "type": "object",
        "required": ["label", "tokens", "points"],
        "properties": {
            "label": {"type": "string"},
            "prompt": {"type": "string"},
            "model": {"type": ["string", "null"]},
            "tokens": _STRINGS,
            "trajectory_labels": _STRINGS,
            "points": {**_REF, "description": "(trajectories, points, 3) draped paths"},
            "entropy": {**_REF, "description": "(layers, tokens)"},
            "quality": {**_REF, "description": "(layers, tokens) neighborhood preservation"},
            "attention": {**_REF, "description": "(layers-1, tokens, tokens)"},
            "topk": {"$ref": "#/$defs/topk"},
            "generation": {"type": "object"},
            "inspector": {"$ref": "#/$defs/inspector"},
            "features": {
                "type": "object",
                "properties": {"source": {"type": ["string", "null"]},
                               "hook": {"type": ["string", "null"]},
                               "best_layer": {"type": "integer"},
                               "recon_error": _REF, "top_id": _REF, "top_act": _REF},
            },
        },
    }
    comparison = {
        "type": "object",
        "properties": {"label": {"type": "string"},
                       "hausdorff": _NULLABLE_NUMBER,
                       "dtw_normalized": _NULLABLE_NUMBER,
                       "shared_tokens": {"type": "integer"},
                       "onset_layer": {"type": "integer"},
                       "readout_changed": {"type": ["integer", "null"]}},
    }
    return {
        "$schema": DIALECT,
        "$id": ID_BASE + "mottled-scene-1.schema.json",
        "title": SCENE_SCHEMA,
        "description": "Manifest of a kind:\"scene\" .mtj container: a viewer-ready "
                       "bundle (docs/mtj-format.md). Unknown fields are allowed and "
                       "must be ignored by readers.",
        "type": "object",
        "required": ["format", "version", "kind", "terrain", "runs", "arrays"],
        "properties": {
            **_header("scene", SCENE_SCHEMA),
            "terrain": {"type": "object", "required": ["x", "y", "z"],
                        "properties": {"x": _REF, "y": _REF, "z": _REF,
                                       "density": _REF, "se": _REF}},
            "runs": {"type": "array", "minItems": 1, "items": run},
            "comparisons": {"type": "array", "items": comparison},
            "blast": {"type": "object",
                      "properties": {"schema": {"type": "string",
                                                "pattern": r"^mottled-blast/[0-9]+$"},
                                     "layouts": {"type": "array"}}},
        },
        "$defs": _DEFS,
    }


def trajectory_schema() -> dict:
    return {
        "$schema": DIALECT,
        "$id": ID_BASE + "mottled-trajectory-1.schema.json",
        "title": TRAJECTORY_SCHEMA,
        "description": "Manifest of a kind:\"trajectory\" .mtj container: one "
                       "StateTrajectory at full fidelity (docs/mtj-format.md). A "
                       "stream is several containers back to back. Unknown fields "
                       "are allowed and must be ignored by readers.",
        "type": "object",
        "required": ["format", "version", "kind", "tokens", "arrays"],
        "properties": {
            **_header("trajectory", TRAJECTORY_SCHEMA),
            "tokens": _STRINGS,
            "vocab": _STRINGS,
            "topk": {"$ref": "#/$defs/topk"},
            "inspector": {"$ref": "#/$defs/inspector"},
        },
        "$defs": _DEFS,
    }


SCHEMAS = {SCENE_SCHEMA: scene_schema, TRAJECTORY_SCHEMA: trajectory_schema}
KIND_SCHEMA = {"scene": SCENE_SCHEMA, "trajectory": TRAJECTORY_SCHEMA}


def schema_files() -> dict[str, dict]:
    """file name -> schema, as published under docs/schema/."""
    return {"mottled-scene-1.schema.json": scene_schema(),
            "mottled-trajectory-1.schema.json": trajectory_schema()}


# ------------------------------------------------------- the subset validator
@dataclass(frozen=True)
class Problem:
    level: str      # "error" | "warning"
    where: str      # e.g. "container 0: runs[1].points"
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _path(base: str, key) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]"
    return f"{base}.{key}" if base else str(key)


def check_schema(value, schema: dict, root: dict, where: str = "",
                 refs: list | None = None) -> list[tuple[str, str]]:
    """Validate `value` against `schema` (the subset these schemas use).
    Returns (where, message) pairs; appends array references to `refs`."""
    errs: list[tuple[str, str]] = []
    here = where or "manifest"
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        return check_schema(value, root["$defs"][name], root, where, refs)
    if "const" in schema and value != schema["const"]:
        return [(here, f"must be {json.dumps(schema['const'])}, got {json.dumps(value)}")]
    if "enum" in schema and value not in schema["enum"]:
        return [(here, f"must be one of {schema['enum']}, got {json.dumps(value)}")]
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](value) for t in types):
            got = type(value).__name__ if value is not None else "null"
            return [(here, f"must be {' or '.join(types)}, got {got}")]
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append((here, "must not be empty"))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append((here, f"{value!r} does not match {schema['pattern']}"))
        if schema.get("x-mtj-array") and refs is not None and value:
            refs.append((here, value))
    if _TYPES["number"](value) and "minimum" in schema and value < schema["minimum"]:
        errs.append((here, f"must be >= {schema['minimum']}, got {value}"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errs.append((here, f"missing required field {key!r}"))
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, sub in value.items():
            if key in props:
                errs += check_schema(sub, props[key], root, _path(where, key), refs)
            elif isinstance(extra, dict):
                errs += check_schema(sub, extra, root, _path(where, key), refs)
            elif extra is False:
                errs.append((_path(where, key), "unexpected field"))
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append((here, f"needs at least {schema['minItems']} item(s)"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append((here, f"has more than {schema['maxItems']} items"))
        prefix = schema.get("prefixItems", [])
        for i, item in enumerate(value):
            sub = prefix[i] if i < len(prefix) else schema.get("items")
            if sub is not None:
                errs += check_schema(item, sub, root, _path(where, i), refs)
    return errs


# ----------------------------------------------------- the container checks
def _strict_json(text: str):
    def reject(token):
        raise ValueError(f"non-standard JSON number {token} (write non-finite "
                         f"numbers as null; docs/mtj-format.md, \"Numbers\")")
    return json.loads(text, parse_constant=reject)


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def validate_bytes(raw: bytes, strict: bool = False) -> list[Problem]:
    """Every problem with a .mtj file or stream (one or more containers).

    Errors make the file invalid; warnings are things a current writer would
    not produce but a reader copes with (a missing schema id from an older
    writer, an unknown dtype). `strict=True` promotes warnings to errors.
    """
    problems: list[Problem] = []
    pos, index = 0, 0
    while True:
        where = f"container {index}"
        found, end = _validate_one(raw, pos, where)
        problems += found
        if end is None:
            break
        pos, index = end, index + 1
        rest = raw[pos:]
        if not rest.strip(b"\x00 "):
            break
        if rest[:4] != MAGIC:
            problems.append(Problem("error", f"byte {pos}",
                                    f"{len(rest)} trailing byte(s) after the last "
                                    "container that are not another container"))
            break
    if strict:
        problems = [Problem("error", p.where, p.message) for p in problems]
    return problems


def _validate_one(raw: bytes, pos: int, where: str) -> tuple[list[Problem], int | None]:
    P = lambda level, w, m: Problem(level, f"{where}: {w}" if w else where, m)  # noqa: E731
    if raw[pos:pos + 4] == MAGIC and len(raw) - pos < 12:
        return [P("error", "header", f"truncated: {len(raw) - pos} bytes, the header "
                                     "needs 12")], None
    if raw[pos:pos + 4] != MAGIC:
        return [P("error", "", "not a .mtj container (bad magic; expected b'MTRJ')")], None
    version, mlen = struct.unpack_from("<II", raw, pos + 4)
    if version != VERSION:
        return [P("error", "header", f"unsupported version {version} (this reader "
                                     f"knows {VERSION})")], None
    start = pos + 12 + mlen
    if start > len(raw):
        return [P("error", "header", f"manifest length {mlen} runs past the end "
                                     f"of the file")], None
    if start % ALIGN:
        # writers pad the manifest so the blob starts 16-byte aligned
        problems_align = [P("warning", "header", "blob does not start on a 16-byte boundary")]
    else:
        problems_align = []
    try:
        manifest = _strict_json(raw[pos + 12:start].decode("utf-8"))
    except UnicodeDecodeError as exc:
        return [P("error", "manifest", f"not UTF-8: {exc}")], None
    except ValueError as exc:
        return [P("error", "manifest", f"not strict JSON: {exc}")], None
    if not isinstance(manifest, dict):
        return [P("error", "manifest", "must be a JSON object")], None

    problems = list(problems_align)
    kind = manifest.get("kind")
    schema_id = KIND_SCHEMA.get(kind)
    if schema_id is None:
        problems.append(P("error", "kind", f"unknown kind {kind!r} (expected "
                                           f"{' or '.join(map(repr, KIND_SCHEMA))})"))
        schema = None
    else:
        schema = SCHEMAS[schema_id]()
        if "schema" not in manifest:
            problems.append(P("warning", "schema",
                              f"no schema id (written before ids existed); "
                              f"expected {schema_id!r}"))

    refs: list[tuple[str, str]] = []
    if schema is not None:
        for w, msg in check_schema(manifest, schema, schema, "", refs):
            problems.append(P("error", w, msg))

    arrays = manifest.get("arrays") if isinstance(manifest.get("arrays"), dict) else {}
    blob_len, end = len(raw) - start, start
    shapes: dict[str, list] = {}
    for name, ref in arrays.items():
        if not isinstance(ref, dict) or not all(
                isinstance(ref.get(k), int) for k in ("offset", "length")):
            continue                                     # the schema said so
        off, length = ref["offset"], ref["length"]
        end = max(end, start + off + length)
        w = f"arrays[{name!r}]"
        if off % ALIGN:
            problems.append(P("error", w, f"offset {off} is not a multiple of {ALIGN}"))
        if off + length > blob_len:
            problems.append(P("error", w, f"bytes {off}..{off + length} run past the "
                                          f"end of the blob ({blob_len} bytes)"))
        item = ITEMSIZE.get(ref.get("dtype"))
        if item is None:
            problems.append(P("warning", w, f"unknown dtype {ref.get('dtype')!r}; "
                                            "readers will skip this array"))
        elif isinstance(ref.get("shape"), list) and length != _prod(ref["shape"]) * item:
            problems.append(P("error", w, f"length {length} != {ref['shape']} x "
                                          f"{item} bytes ({ref['dtype']})"))
        else:
            shapes[name] = list(ref.get("shape") or [])
    for w, name in refs:
        if name not in arrays:
            problems.append(P("error", w, f"names array {name!r}, which is not in `arrays`"))

    problems += _shape_checks(manifest, shapes, raw, start, arrays, P)
    return problems, end


def _shape_checks(manifest, shapes, raw, start, arrays, P) -> list[Problem]:
    out = []
    kind = manifest.get("kind")
    if kind == "trajectory":
        tokens = manifest.get("tokens") or []
        hidden = shapes.get("hidden")
        if "hidden" not in arrays:
            out.append(P("error", "arrays", "a trajectory needs a 'hidden' array"))
        elif hidden is not None:
            if len(hidden) != 3:
                out.append(P("error", "arrays['hidden']", f"must be (L, T, D), got {hidden}"))
            else:
                L, T, _ = hidden
                if T != len(tokens):
                    out.append(P("error", "tokens", f"{len(tokens)} tokens but hidden "
                                                   f"has T={T}"))
                if shapes.get("entropy") not in (None, [L, T]):
                    out.append(P("error", "arrays['entropy']",
                                 f"shape {shapes['entropy']} != (L, T) = {[L, T]}"))
                if not _all_finite(raw, start, arrays["hidden"]):
                    out.append(P("error", "arrays['hidden']", "contains NaN or Infinity"))
    elif kind == "scene":
        for i, run in enumerate(manifest.get("runs") or []):
            if not isinstance(run, dict):
                continue
            T = len(run.get("tokens") or [])
            for key in ("entropy", "quality"):
                s = shapes.get(run.get(key)) if isinstance(run.get(key), str) else None
                if s is not None and (len(s) != 2 or s[1] != T):
                    out.append(P("error", f"runs[{i}].{key}",
                                 f"shape {s} is not (layers, {T} tokens)"))
            s = shapes.get(run.get("points")) if isinstance(run.get("points"), str) else None
            if s is not None and len(s) != 3:
                out.append(P("error", f"runs[{i}].points",
                             f"shape {s} is not (trajectories, points, coords)"))
    return out


def _all_finite(raw: bytes, start: int, ref: dict) -> bool:
    item = ITEMSIZE.get(ref.get("dtype"))
    if item is None:
        return True
    lo = start + ref["offset"]
    buf = raw[lo:lo + ref["length"]]
    try:
        import numpy as np

        dt = {"float16": "<f2", "float32": "<f4", "int32": "<i4"}[ref["dtype"]]
        return bool(np.isfinite(np.frombuffer(buf, dtype=dt)).all())
    except ImportError:                                  # pragma: no cover
        fmt = {"float16": "e", "float32": "f", "int32": "i"}[ref["dtype"]]
        return all(math.isfinite(v) for (v,) in struct.iter_unpack("<" + fmt, buf))
