"""Input boundary tests for the local Observatory HTTP API."""

from jspace_server.app import (
    ALLOWED_ORIGIN,
    ANALYSIS_GATE,
    _is_json_content_type,
    _origin_allowed,
)


def test_observatory_origin_policy_allows_only_local_ui_or_direct_clients():
    assert _origin_allowed(None)
    assert _origin_allowed(ALLOWED_ORIGIN)
    assert not _origin_allowed("https://example.com")
    assert not _origin_allowed("null")


def test_observatory_requires_json_content_type():
    assert _is_json_content_type("application/json")
    assert _is_json_content_type("application/json; charset=utf-8")
    assert not _is_json_content_type(None)
    assert not _is_json_content_type("text/plain")


def test_observatory_allows_only_one_analysis_at_a_time():
    assert ANALYSIS_GATE.acquire(blocking=False)
    try:
        assert not ANALYSIS_GATE.acquire(blocking=False)
    finally:
        ANALYSIS_GATE.release()
