from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.doc_processor.parsers import pdf_parser


class Reader:
    metadata = {}

    def __init__(self, pages):
        self.pages = pages
        self.close = Mock()

    def __iter__(self):
        return iter(self.pages)

    def __len__(self):
        return len(self.pages)


def test_pdf_offsets_preserve_original_page_numbers_including_blank_pages(monkeypatch):
    texts = ["First page\n", "", "Third page\n"]
    reader = Reader([SimpleNamespace(get_text=Mock(return_value=text)) for text in texts])
    monkeypatch.setattr(pdf_parser.fitz, "open", lambda _: reader)
    result = pdf_parser.PdfParser().parse(Path("example.pdf"))
    spans = result.metadata["page_spans"]
    assert [span["page"] for span in spans] == [1, 2, 3]
    assert result.metadata["pages_without_text"] == [2]
    for span, expected in zip(spans, texts):
        assert result.raw_text[span["start"]:span["end"]] == expected
    assert result.metadata["ocr_performed"] is False
    reader.close.assert_called_once()


def test_pdf_handle_closes_on_extraction_failure(monkeypatch):
    reader = Reader([SimpleNamespace(get_text=Mock(side_effect=RuntimeError("extraction failed")))])
    monkeypatch.setattr(pdf_parser.fitz, "open", lambda _: reader)
    with pytest.raises(RuntimeError, match="extraction failed"):
        pdf_parser.PdfParser().parse(Path("example.pdf"))
    reader.close.assert_called_once()
