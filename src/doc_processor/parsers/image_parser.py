"""Images are registered as documents; visual content has a separate index."""
from pathlib import Path
from .base_parser import BaseParser, Document
from src.multimodal.provider import prepare_image


class ImageParser(BaseParser):
    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

    def parse(self, file_path: Path) -> Document:
        _, _, width, height = prepare_image(file_path.read_bytes())
        return Document("image_" + file_path.stem, file_path.name, str(file_path), "image", "",
                        {"width": width, "height": height})
