import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.interpreter import _clean_results, _parse_json_response, _STRIP_KEYS


def _fake_response(text):
    """Mimic an anthropic Message: .content is a list of typed blocks."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


# --- _parse_json_response ------------------------------------------------

def test_parses_plain_json():
    resp = _fake_response('{"consensus": true, "summary": "ok"}')
    assert _parse_json_response(resp) == {"consensus": True, "summary": "ok"}


def test_strips_json_code_fence():
    resp = _fake_response('```json\n{"a": 1}\n```')
    assert _parse_json_response(resp) == {"a": 1}


def test_strips_bare_code_fence():
    resp = _fake_response('```\n{"a": 1}\n```')
    assert _parse_json_response(resp) == {"a": 1}


def test_falls_back_to_outermost_object_when_prose_wraps_json():
    resp = _fake_response('Here is the result:\n{"a": 1, "b": [2, 3]}\nHope that helps!')
    assert _parse_json_response(resp) == {"a": 1, "b": [2, 3]}


def test_empty_response_raises():
    with pytest.raises(ValueError, match="empty response"):
        _parse_json_response(_fake_response("   "))


def test_unrecoverable_text_raises_json_error():
    import json
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response(_fake_response("not json at all"))


# --- _clean_results ------------------------------------------------------

def test_clean_results_strips_bulky_keys():
    results = [
        {
            "method": "psm",
            "ate": -1.2,
            "propensity_scores": [0.1, 0.2],
            "matched_weights": [1, 0],
            "rdd_plot": "<base64>",
        }
    ]
    cleaned = _clean_results(results)
    assert cleaned == [{"method": "psm", "ate": -1.2}]
    # every stripped key is gone; kept keys are untouched
    assert not (_STRIP_KEYS & set(cleaned[0]))
