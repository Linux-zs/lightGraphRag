import hashlib

from src.source_location import locate_source_text


def metadata(text):
    return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "page_spans": [
        {"page": 1, "start": 0, "end": 3}, {"page": 2, "start": 3, "end": len(text)},
    ]}


def test_exact_chunk_crossing_page_boundary():
    result = locate_source_text("abcdef", "cde", metadata("abcdef"))
    assert result["start"] == 2
    assert result["end"] == 5
    assert result["pages"] == [1, 2]


def test_ambiguous_missing_or_stale_text_has_no_location():
    assert locate_source_text("abcabc", "abc", metadata("abcabc")) is None
    assert locate_source_text("abc", "other", metadata("abc")) is None
    assert locate_source_text("abc", "abc", metadata("old")) is None
