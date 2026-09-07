"""Non-destructive SDK deduplication view for logical-document replacements."""

from contextlib import contextmanager

from lightrag.base import DocStatus
from lightrag.utils_pipeline import doc_status_field, normalize_document_file_path


class ReplacementStatusView:
    def __init__(self, storage, replaced_ids):
        self.storage = storage
        self.replaced_ids = set(replaced_ids)

    def __getattr__(self, name):
        return getattr(self.storage, name)

    async def get_docs_by_statuses(self, statuses):
        rows = await self.storage.get_docs_by_statuses(statuses)
        return {key: value for key, value in rows.items() if key not in self.replaced_ids}

    async def get_doc_by_file_basename(self, basename):
        for doc_id, row in (await self.get_docs_by_statuses(list(DocStatus))).items():
            if normalize_document_file_path(doc_status_field(row, "file_path", "")) == basename:
                return doc_id, row
        return None

    async def get_doc_by_content_hash(self, content_hash):
        for doc_id, row in (await self.get_docs_by_statuses(list(DocStatus))).items():
            if doc_status_field(row, "content_hash", "") == content_hash:
                return doc_id, row
        return None


@contextmanager
def allow_document_replacement(rag, replaced_ids):
    if not replaced_ids:
        yield
        return
    storage = rag.doc_status
    rag.doc_status = ReplacementStatusView(storage, replaced_ids)
    try:
        yield
    finally:
        rag.doc_status = storage
