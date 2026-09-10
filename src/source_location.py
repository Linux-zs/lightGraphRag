"""Conservative exact source locations; ambiguous text never gets a guessed page."""

import hashlib


def locate_source_text(raw_text: str, chunk_text: str, metadata: dict) -> dict | None:
    if not chunk_text or metadata.get("text_sha256") != hashlib.sha256(raw_text.encode("utf-8")).hexdigest():
        return None
    start = raw_text.find(chunk_text)
    if start < 0 or raw_text.find(chunk_text, start + 1) >= 0:
        return None
    end = start + len(chunk_text)
    pages = [span["page"] for span in metadata.get("page_spans", [])
             if not span.get("text_missing") and span["start"] < end and span["end"] > start]
    return {"start": start, "end": end, "pages": pages, "text_sha256": metadata["text_sha256"]}
