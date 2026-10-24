"""Extract image occurrences without inventing source page numbers."""
from __future__ import annotations

from pathlib import Path
import fitz
from docx import Document as WordDocument

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_images(path: Path):
    """Yield (image bytes, location, related source text); warnings as bytes=None."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        yield path.read_bytes(), {"kind": "image"}, ""
    elif suffix == ".pdf":
        with fitz.open(path) as document:
            for number, page in enumerate(document, 1):
                blocks = [b for b in page.get_text("blocks") if len(b) > 6 and b[6] == 0]
                for block in page.get_text("dict")["blocks"]:
                    if block["type"] != 1:
                        continue
                    bbox = block["bbox"]
                    nearest = sorted(blocks, key=lambda b: abs(b[1] - bbox[1]))[:3]
                    # Convert PDF image encodings to a supported lossless raster.
                    try:
                        pix = fitz.Pixmap(block["image"])
                        if pix.colorspace and pix.colorspace.n != 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        raw = pix.tobytes("png")
                        yield raw, {"kind": "embedded", "page": number, "bbox": list(bbox),
                                    "association": "same_page_nearest_text"}, "\n".join(b[4] for b in nearest)
                    except Exception:
                        yield None, {"warning": f"第 {number} 页图片无法解码"}, ""
                if not page.get_text().strip() or page.get_drawings():
                    scale = min(1.5, 1536 / max(page.rect.width, page.rect.height))
                    yield page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png"), {
                        "kind": "page_image", "page": number, "association": "same_page"}, page.get_text()[:12000]
    elif suffix == ".docx":
        document = WordDocument(path)
        section = ""
        paragraphs = list(document.element.body.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"))
        text = lambda p: "".join(p.xpath(".//w:t/text()"))
        for index, paragraph in enumerate(paragraphs):
            styles = paragraph.xpath("./w:pPr/w:pStyle/@w:val")
            if styles and str(styles[0]).lower().startswith("heading"):
                section = text(paragraph)
            context = "\n".join(text(p) for p in paragraphs[max(0, index-1):index+2])
            for blip in paragraph.xpath(".//a:blip"):
                rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                if not rid:
                    yield None, {"warning": f"段落 {index+1} 外链图片未提取"}, ""
                    continue
                part = document.part.related_parts.get(rid)
                if part is None:
                    yield None, {"warning": f"段落 {index+1} 图片关系缺失"}, ""
                    continue
                in_table = bool(paragraph.xpath("ancestor::w:tc"))
                yield part.blob, {"kind": "embedded", "paragraph": index+1, "in_table": in_table, "section": section,
                                  "association": "adjacent_paragraphs"}, context
            if paragraph.xpath('.//*[local-name()="relIds" or local-name()="OLEObject" or local-name()="imagedata"]'):
                yield None, {"warning": f"段落 {index+1} 的 SmartArt/OLE/VML 对象未渲染"}, ""
