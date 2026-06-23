"""
Tier 0 unit tests -- run without Streamlit, no OpenAI calls, no live app.

Usage:
    python _tier0_tests.py
"""
import sys
import os
import re
import json
import types
import unittest.mock as mock
from collections import Counter
from typing import List

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Minimal stubs so the module can be imported without Streamlit/OpenAI ────
st_stub = types.ModuleType("streamlit")
for _attr in ["set_page_config", "title", "markdown", "subheader", "caption",
              "progress", "empty", "fragment", "info", "stop", "success",
              "warning", "error", "button", "download_button", "sidebar",
              "text_input", "number_input", "selectbox", "checkbox", "multiselect",
              "file_uploader", "expander", "columns", "dataframe", "write",
              "session_state", "text"]:
    setattr(st_stub, _attr, mock.MagicMock())
st_stub.session_state = {}
sys.modules["streamlit"] = st_stub

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **kw: None
sys.modules["dotenv"] = dotenv_stub

openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = mock.MagicMock()
openai_stub.AsyncOpenAI = mock.MagicMock()
sys.modules["openai"] = openai_stub

os.environ.setdefault("OPENAI_API_KEY", "test-key-dummy")

# ── Import the app module ────────────────────────────────────────────────────
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "app",
    pathlib.Path(__file__).parent / "forsta_translation_qa_app.py"
)
app = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(app)
except SystemExit:
    pass

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((status, name, detail))
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Step 1: call_consistency_model returns [] on non-retryable failure ===")

class _BadClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                raise RuntimeError("Injected failure")

_orig_get = getattr(app, "get_llm_client", None)
app.get_llm_client = lambda: _BadClient()

ctx_stub = mock.MagicMock()
ctx_stub.language_code = "es"
ctx_stub.locale_code = "es-MX"

try:
    result = app.call_consistency_model(ctx_stub, [{"english_phrase": "x", "translations": []}])
    check("call_consistency_model returns [] not raises", result == [], repr(result))
except Exception as e:
    check("call_consistency_model returns [] not raises", False, f"raised {e!r}")

if _orig_get:
    app.get_llm_client = _orig_get


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Step 2: structure validation used by process_row_async ===")

is_ok, msg = app.validate_translation_structure("You have {N} items", "Tienes articulos")
check("structure: {N} missing => not ok", not is_ok, msg)
check("message mentions placeholder", "placeholders" in msg, msg)

is_ok2, msg2 = app.validate_translation_structure("{count} results", "{count} resultados")
check("structure: present placeholder => ok", is_ok2, msg2)


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Step 3: extract_html_tags and HTML check in validate_translation_structure ===")

tags_b = app.extract_html_tags("<b>Hello</b><br>")
check("extract_html_tags finds <b>", "<b>" in tags_b, repr(tags_b))
check("extract_html_tags finds </b>", "</b>" in tags_b, repr(tags_b))
check("extract_html_tags finds <br>", "<br>" in tags_b, repr(tags_b))

tags_empty = app.extract_html_tags("")
check("extract_html_tags on empty string", tags_empty == [], repr(tags_empty))

is_ok3, msg3 = app.validate_translation_structure("<b>Hola</b><br>", "Hola")
check("html-tag check: missing b/br => not ok", not is_ok3, msg3)
check("html-tag message mentions 'HTML tags'", "HTML tags" in msg3, msg3)

is_ok4, msg4 = app.validate_translation_structure("<b>Hi</b>", "<b>Hola</b>")
check("html-tag check: matching tags => ok", is_ok4, msg4)


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Step 4: _safe_json helper ===")

resp_null = mock.MagicMock()
resp_null.choices[0].finish_reason = None
resp_null.choices[0].message.content = None
try:
    app._safe_json(resp_null)
    check("_safe_json raises on null content", False, "did not raise")
except app._RetryableModelError as e:
    check("_safe_json raises _RetryableModelError on null content", True, str(e))
except Exception as e:
    check("_safe_json raises _RetryableModelError on null content", False, f"wrong exc: {e!r}")

resp_len = mock.MagicMock()
resp_len.choices[0].finish_reason = "length"
resp_len.choices[0].message.content = '{"x": 1}'
try:
    app._safe_json(resp_len)
    check("_safe_json raises on finish_reason=length", False, "did not raise")
except app._RetryableModelError as e:
    check("_safe_json raises _RetryableModelError on length", True, str(e))
except Exception as e:
    check("_safe_json raises _RetryableModelError on length", False, f"wrong exc: {e!r}")

resp_fence = mock.MagicMock()
resp_fence.choices[0].finish_reason = "stop"
resp_fence.choices[0].message.content = '```json\n{"key": "val"}\n```'
try:
    d = app._safe_json(resp_fence)
    check("_safe_json strips code fence", d == {"key": "val"}, repr(d))
except Exception as e:
    check("_safe_json strips code fence", False, f"raised: {e!r}")

resp_ok = mock.MagicMock()
resp_ok.choices[0].finish_reason = "stop"
resp_ok.choices[0].message.content = '{"a": 1}'
d2 = app._safe_json(resp_ok)
check("_safe_json parses valid JSON", d2 == {"a": 1}, repr(d2))

resp_bad = mock.MagicMock()
resp_bad.choices[0].finish_reason = "stop"
resp_bad.choices[0].message.content = "not json {"
try:
    app._safe_json(resp_bad)
    check("_safe_json raises on malformed JSON", False, "did not raise")
except app._RetryableModelError as e:
    check("_safe_json raises _RetryableModelError on malformed JSON", True, str(e))
except Exception as e:
    check("_safe_json raises _RetryableModelError on malformed JSON", False, f"wrong exc: {e!r}")


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Step 5: numeric multiset validation ===")

is_ok5, msg5 = app.validate_translation_structure("You scored 100", "Obtuviste 1000")
check("100 vs 1000: not ok (old substring check would pass)", not is_ok5, msg5)

is_ok6, msg6 = app.validate_translation_structure("Ages 18-34", "Edades 34-18")
check("reversed range 18-34 vs 34-18: not ok", not is_ok6, msg6)
check("reversed range message mentions 'reversed'", "reversed" in msg6.lower(), msg6)

is_ok7, msg7 = app.validate_translation_structure("Between 5 and 5", "Entre 5")
check("dropped duplicate 5: not ok", not is_ok7, msg7)

is_ok8, msg8 = app.validate_translation_structure("1,000 items", "1.000 articulos")
check("separator normalization 1,000 vs 1.000: ok", is_ok8, msg8)

is_ok9, msg9 = app.validate_translation_structure("Ages 18-34", "Edades 18-34")
check("correct range 18-34: ok", is_ok9, msg9)


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"{passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("FAILED checks:")
    for s, name, detail in results:
        if s == "FAIL":
            print(f"  x {name}: {detail}")
    sys.exit(1)
else:
    print("All checks passed.")
