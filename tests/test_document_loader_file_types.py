from pathlib import Path

from src.doc_processor.loader import DocumentLoader
from src.doc_processor.parsers.txt_parser import TxtParser


def test_loader_supports_expected_upload_types():
    loader = DocumentLoader()
    supported = set(loader._ext_to_parser.keys())

    assert ".txt" in supported
    assert ".md" in supported
    assert ".pdf" in supported
    assert ".docx" in supported
    assert ".doc" not in supported


def test_txt_parser_reads_utf8_text(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("第一行\nsecond line", encoding="utf-8")

    doc = TxtParser().parse(path)

    assert doc.file_name == "sample.txt"
    assert doc.file_type == "txt"
    assert "第一行" in doc.raw_text
    assert doc.metadata["line_count"] == 2
