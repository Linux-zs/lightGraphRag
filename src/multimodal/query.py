"""Query routing for image provenance, not image content keywords."""
import re


def is_image_source_query(text: str) -> bool:
    return bool(re.search(
        r"出自|出处|来源|哪.{0,16}(?:文档|文件|资料|页)|所在.{0,8}(?:文档|文件)"
        r"|which.{0,24}(?:document|file)|where.{0,24}from|source",
        text, re.IGNORECASE,
    ))
