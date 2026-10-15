"""Generated files stay generated, and the viewer treats scene strings as data.

`python -m codegen` writes docs/schema/*.json (from mtjschema.py) and the
viewer's tokens.js, style.css :root block and the Streamlit theme (from
design_tokens.py). If any of them is edited by hand, or the source changes
without a regeneration, this fails and says which file.
"""
import json
import re
from pathlib import Path

import codegen
import design_tokens as T
import mtjschema

ROOT = Path(__file__).resolve().parent.parent


def test_every_generated_file_is_in_sync():
    stale = [str(p.relative_to(ROOT)) for p in codegen.stale()]
    assert stale == [], f"run `python -m codegen`: {stale}"
    assert codegen.main(["--check"]) == 0


def test_check_mode_names_a_stale_file(monkeypatch, capsys):
    real = codegen.outputs

    def drifted():
        out = real()
        out[ROOT / "viewer" / "tokens.js"] += "// drift\n"
        return out
    monkeypatch.setattr(codegen, "outputs", drifted)
    assert codegen.main(["--check"]) == 1
    assert "viewer/tokens.js" in capsys.readouterr().err


def test_published_schemas_are_the_python_ones():
    for name, schema in mtjschema.schema_files().items():
        assert json.loads((ROOT / "docs" / "schema" / name).read_text()) == schema


def test_viewer_palette_comes_from_the_token_module():
    js = (ROOT / "viewer" / "tokens.js").read_text()
    assert json.dumps(T.MARBLE_COLORS, indent=2).replace("\n", "\n  ") in js
    main = (ROOT / "viewer" / "main.js").read_text()
    assert "const PALETTE = MOTTLED_TOKENS.marbleColors;" in main
    html = (ROOT / "viewer" / "index.html").read_text()
    assert html.index('src="tokens.js"') < html.index('src="main.js"')


# ------------------------------------------------------------ viewer safety
_SCENE_OBJECTS = r"(c|p|run|meta|scene|r|lg|pellet|feat|step|n|m|b|s|e|lay|summary|rec)"
_SAFE = re.compile(r"^\s*(esc|fmt)\(|toFixed\(|\.length\b|^\s*Math\."
                   r"|\?\s*\"[^\"]*\"\s*:\s*\"[^\"]*\"\s*$")   # x ? "" : "s"
# sinks that never parse HTML: assignments to textContent/title, the status
# helpers (textContent), and Error messages
_TEXT_SINKS = re.compile(r"\.textContent\s*=|\.title\s*=|setModelStatus\(|showMessage\(|"
                         r"new Error\(|textContent =")


def test_every_scene_value_in_viewer_markup_is_escaped():
    """A scene is untrusted input (?file= takes any URL). Every value read
    off a scene object and interpolated into markup must go through esc()
    (or be formatted as a number). This is the regression test for the
    unescaped comparison and provenance cells."""
    lines = (ROOT / "viewer" / "main.js").read_text().splitlines()
    offenders = []
    for i, line in enumerate(lines, 1):
        previous = lines[i - 2] if i > 1 else ""
        if _TEXT_SINKS.search(line) or previous.rstrip().endswith("textContent ="):
            continue
        for m in re.finditer(r"\$\{([^{}]*)\}", line):
            expr = m.group(1)
            if _SAFE.search(expr):
                continue
            if re.match(rf"^\s*{_SCENE_OBJECTS}(\.|\[)", expr):
                offenders.append(f"main.js:{i}: ${{{expr.strip()}}}")
    assert offenders == []


def test_the_two_reported_sinks_are_escaped():
    main = (ROOT / "viewer" / "main.js").read_text()
    assert "${esc(c.shared_tokens" in main
    assert "seed ${esc(p.seed)}" in main


def test_viewer_has_a_content_security_policy():
    html = (ROOT / "viewer" / "index.html").read_text()
    m = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert m, "viewer/index.html has no CSP"
    policy = {d.split()[0]: d.split()[1:] for d in m.group(1).split("; ")}
    assert policy["script-src"] == ["'self'"]          # no inline, no eval
    assert policy["object-src"] == ["'none'"] and policy["base-uri"] == ["'none'"]
    assert "<script>" not in html and not re.search(r"\son[a-z]+=", html)
