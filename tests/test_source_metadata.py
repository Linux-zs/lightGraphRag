import hashlib

from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGService


def test_source_metadata_survives_manifest_reopen_and_tracks_replacement(tmp_path):
    config = {"paths": {"data_dir": str(tmp_path)}}
    service = LightRAGService(config, workspace="kb")
    doc = Document("doc", "source.txt", "source.txt", "txt", "first", {
        "page_spans": [{"page": 1, "start": 0, "end": 5}],
        "page_count": 1, "ocr_performed": False,
    })
    item = service.register_upload(doc)
    reopened = LightRAGService(config, workspace="kb")
    saved = reopened._load_manifest()["documents"][item["doc_id"]]
    assert saved["source_metadata"]["page_spans"] == doc.metadata["page_spans"]
    assert saved["source_metadata"]["text_sha256"] == hashlib.sha256(b"first").hexdigest()
    # New source without page information must not inherit obsolete offsets.
    doc.raw_text = "replacement"
    doc.metadata = {"lightrag_doc_id": item["doc_id"]}
    changed = reopened.register_upload(doc)
    assert "page_spans" not in changed["source_metadata"]
    assert changed["source_metadata"]["text_sha256"] == changed["content_sha256"]
