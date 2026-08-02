"""Unit tests for server.py helper functions that had NO test coverage.

Covers:
  - _strip_citation_section(): strips trailing "引用文档" section
  - _detect_repetition_degenerate(): detects LLM repetition loops
  - _is_contaminated_text(): detects LLM garbage output
  - _sanitize_chunk_text(): normalizes chunk text before LLM prompt

These are pure functions — no server needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.server import (
    _strip_citation_section,
    _detect_repetition_degenerate,
    _is_contaminated_text,
    _sanitize_chunk_text,
)


# ===========================================================================
# _strip_citation_section
# ===========================================================================

class TestStripCitationSection:
    """Tests for _strip_citation_section()."""

    def test_empty_string_returns_empty(self):
        assert _strip_citation_section("") == ""

    def test_none_returns_none(self):
        assert _strip_citation_section(None) is None

    def test_normal_answer_unchanged(self):
        """A normal answer without any citation section should be unchanged."""
        text = "ABC是示例金融交易系统[1]。它包含行情系统和交易系统[2]。"
        assert _strip_citation_section(text) == text

    def test_strips_header_citation_section(self):
        """Should remove everything from '引用文档：' header to end."""
        text = (
            "ABC是示例金融交易系统[1]。\n\n"
            "引用文档：\n"
            "[1] 培训文档.md\n"
            "[2] 系统说明.docx\n"
        )
        result = _strip_citation_section(text)
        assert "引用文档" not in result
        assert "培训文档.md" not in result
        assert "系统说明.docx" not in result
        assert "ABC是示例金融交易系统[1]。" in result

    def test_strips_markdown_header_citation_section(self):
        """Should remove '## 引用文档' markdown header section."""
        text = (
            "答案内容[1]。\n\n"
            "## 引用文档\n"
            "[1] doc1.md\n"
            "[2] doc2.md\n"
        )
        result = _strip_citation_section(text)
        assert "引用文档" not in result
        assert "doc1.md" not in result
        assert "答案内容[1]。" in result

    def test_strips_bare_trailing_citation_entries(self):
        """Should remove trailing block of [数字] entries without header."""
        text = (
            "这是答案内容。\n\n"
            "[1] 文档A.md\n"
            "[2] 文档B.md\n"
            "[3] 文档C.md\n"
        )
        result = _strip_citation_section(text)
        assert "文档A.md" not in result
        assert "文档B.md" not in result
        assert "这是答案内容。" in result

    def test_keeps_citation_markers_in_body(self):
        """[数字] markers inside the answer body should be preserved."""
        text = "第一点[1]，第二点[2]，第三点[3]。"
        result = _strip_citation_section(text)
        assert result == text

    def test_does_not_strip_if_trailing_content_exists(self):
        """If meaningful content follows the [数字] block, don't strip."""
        text = (
            "[1] doc1.md\n"
            "[2] doc2.md\n"
            "\n这是正文内容，不应被删除。\n"
        )
        result = _strip_citation_section(text)
        assert "这是正文内容，不应被删除。" in result

    def test_strips_trailing_blank_lines_after_cut(self):
        """Trailing blank lines left after cutting should be removed."""
        text = "答案[1]。\n\n\n引用文档：\n[1] doc.md\n"
        result = _strip_citation_section(text)
        assert not result.endswith("\n\n")
        assert result.strip() == "答案[1]。"

    def test_returns_original_if_result_empty(self):
        """If stripping would leave empty text, return original."""
        text = "引用文档：\n[1] doc.md\n"
        result = _strip_citation_section(text)
        # The result after stripping would be empty, so return original
        assert result == text


# ===========================================================================
# _detect_repetition_degenerate
# ===========================================================================

class TestDetectRepetitionDegenerate:
    """Tests for _detect_repetition_degenerate()."""

    def test_short_text_not_degenerate(self):
        """Text shorter than 60 chars is never degenerate."""
        is_degen, safe_idx = _detect_repetition_degenerate("短文本")
        assert is_degen is False
        assert safe_idx == len("短文本")

    def test_normal_text_not_degenerate(self):
        """Normal varied text should not be flagged."""
        text = "这是示例行情系统的说明文档。" * 5  # >60 chars, varied
        is_degen, _ = _detect_repetition_degenerate(text)
        assert is_degen is False

    def test_line_repetition_detected(self):
        """5+ consecutive identical non-empty lines → degenerate."""
        line = "这是一行重复的行内容用于测试"  # 14 chars per line
        text = "\n".join([line] * 8)  # 14*8 + 7 = 119 chars > 60
        is_degen, safe_idx = _detect_repetition_degenerate(text)
        assert is_degen is True
        assert safe_idx < len(text)

    def test_char_pattern_repetition_detected(self):
        """Same short pattern repeated 8+ times at end → degenerate."""
        text = "这是正常的开头文字内容用来超过六十个字符的阈值检测" + "ab" * 20  # prefix ~26 + 40 = 66
        is_degen, safe_idx = _detect_repetition_degenerate(text)
        assert is_degen is True
        assert safe_idx < len(text)

    def test_single_char_repetition_detected(self):
        """Last 30 chars being >80% same char → degenerate."""
        # Ensure total length > 60 (prefix must be > 30 chars)
        text = "这是一段足够长的正常中文开头文字内容用来确保超过六十个字符的检测阈值" + "x" * 30
        assert len(text) >= 60
        is_degen, safe_idx = _detect_repetition_degenerate(text)
        assert is_degen is True
        assert safe_idx <= len(text)

    def test_safe_idx_does_not_exceed_text_length(self):
        """safe_idx should never exceed len(text)."""
        text = "重复的行\n" * 10
        is_degen, safe_idx = _detect_repetition_degenerate(text)
        if is_degen:
            assert safe_idx <= len(text)
            assert safe_idx >= 0

    def test_normal_long_text_safe_idx_equals_length(self):
        """For non-degenerate text, safe_idx should equal full length."""
        text = "示例行情系统包含多个组件，每个组件有不同功能。" * 3
        is_degen, safe_idx = _detect_repetition_degenerate(text)
        assert is_degen is False
        assert safe_idx == len(text)


# ===========================================================================
# _is_contaminated_text
# ===========================================================================

class TestIsContaminatedText:
    """Tests for _is_contaminated_text()."""

    def test_empty_string_not_contaminated(self):
        assert _is_contaminated_text("") is False

    def test_short_text_not_contaminated(self):
        """Text shorter than 80 chars is never contaminated."""
        assert _is_contaminated_text("短文本") is False

    def test_normal_chinese_text_not_contaminated(self):
        """Normal Chinese text should not be flagged."""
        text = "示例行情系统是金融交易系统的核心组件，负责接收和处理实时行情数据。" * 3
        assert _is_contaminated_text(text) is False

    def test_noise_lines_detected(self):
        """5+ lines of only noise characters → contaminated."""
        # Each line needs to be substantial enough to exceed 80-char threshold
        text = "这是正常的开头文字用来超过长度阈值检测" + "\n" + "\n".join(["* * * * * * *"] * 6) + "\n正常结尾文字"
        assert len(text) >= 80  # ensure meets length threshold
        assert _is_contaminated_text(text) is True

    def test_excessive_backslash_quotes_detected(self):
        """More than 30 escaped quotes → contaminated."""
        # Pad to exceed 80-char threshold
        text = "这是正常的开头文字内容用来超过长度阈值" + '\\"' * 35  # prefix ~20 + 70 = 90
        assert len(text) >= 80
        assert _is_contaminated_text(text) is True

    def test_pure_binary_stream_detected(self):
        """Long pure binary/hex stream without CJK → contaminated."""
        text = "a1b2c3d4e5f6" * 20  # ~240 chars, no CJK, all alnum
        assert len(text) >= 80
        assert _is_contaminated_text(text) is True

    def test_normal_english_text_not_contaminated(self):
        """English text WITH some CJK should not be flagged (mixed content)."""
        text = (
            "ABC trading system 是示例金融交易系统的核心组件。 "
            "It provides real-time market data and trading functionality. "
            "系统包含多个组件协同工作，确保行情数据的实时传输和处理。 "
        )
        assert _is_contaminated_text(text) is False

    def test_pure_english_not_contaminated(self):
        """Pure English text (no CJK) with real word structure is NOT contaminated.

        P1-10 fix: Pattern B previously flagged ANY text without CJK as
        contamination, which wrongly nuked legitimate English/technical answers
        (containing long technical words, URLs, code). Pattern B now exempts
        text that contains enough real English words (runs of >=2 letters
        making up a meaningful share of the content). Pure binary/hex streams
        (no word structure) are still caught — see test_pure_binary_stream_detected.
        """
        text = (
            "The ABC trading system is a financial trading platform. "
            "It provides real-time market data and trading functionality. "
            "The system consists of multiple components working together. "
        )
        # Normal English with real words is legitimate, not contamination
        assert _is_contaminated_text(text) is False


# ===========================================================================
# _sanitize_chunk_text
# ===========================================================================

class TestSanitizeChunkText:
    """Tests for _sanitize_chunk_text()."""

    def test_empty_string_returns_empty(self):
        assert _sanitize_chunk_text("") == ""

    def test_normal_text_unchanged(self):
        text = "这是正常的中文文本内容。"
        assert _sanitize_chunk_text(text) == text

    def test_strips_non_breaking_spaces(self):
        """U+00A0 should be normalized to regular space."""
        text = "文本\xa0内容"
        result = _sanitize_chunk_text(text)
        assert "\xa0" not in result
        assert " " in result

    def test_normalizes_windows_line_endings(self):
        """\\r\\n should become \\n."""
        text = "第一行\r\n第二行"
        result = _sanitize_chunk_text(text)
        assert "\r\n" not in result
        assert "\n" in result

    def test_collapses_multiple_blank_lines(self):
        """3+ consecutive newlines → at most 2."""
        text = "段落一\n\n\n\n\n段落二"
        result = _sanitize_chunk_text(text)
        assert "\n\n\n" not in result

    def test_drops_noise_only_lines(self):
        """Lines with only noise chars (<=80 chars) should be dropped."""
        text = "正常文本\n* * *\n更多正常文本"
        result = _sanitize_chunk_text(text)
        assert "* * *" not in result
        assert "正常文本" in result

    def test_collapses_multiple_spaces(self):
        """Multiple consecutive spaces/tabs → single space."""
        text = "文本    内容\t\t更多"
        result = _sanitize_chunk_text(text)
        assert "    " not in result
        assert "\t\t" not in result
