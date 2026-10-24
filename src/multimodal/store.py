"""SQLite provenance with immutable, atomically published vector generations."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from nano_vectordb import NanoVectorDB

from .extract import extract_images
from .provider import MultimodalProvider, prepare_image


class ImageStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, width INTEGER, height INTEGER);
                CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, name TEXT, revision TEXT,
                    generation TEXT, signature TEXT, status TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS occurrences(doc_id TEXT, image_id TEXT, location TEXT, context TEXT);
                CREATE TABLE IF NOT EXISTS derived(image_id TEXT, signature TEXT, payload TEXT,
                    PRIMARY KEY(image_id, signature));
                CREATE TABLE IF NOT EXISTS attachments(id TEXT PRIMARY KEY, created REAL, session_id TEXT);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "images.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def asset_path(self, image_id: str, kind="image") -> Path:
        if len(image_id) != 64 or any(c not in "0123456789abcdef" for c in image_id):
            raise ValueError("无效图片 ID")
        return self.root / "assets" / image_id / {"image": "image.jpg", "thumb": "thumb.jpg", "original": "original"}[kind]

    def put(self, raw: bytes) -> str:
        normalized, thumbnail, width, height = prepare_image(raw)
        image_id = hashlib.sha256(raw).hexdigest()
        path = self.asset_path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self.asset_path(image_id, "original").write_bytes(raw)
            path.write_bytes(normalized)
            self.asset_path(image_id, "thumb").write_bytes(thumbnail)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO assets VALUES(?,?,?)", (image_id, width, height))
        return image_id

    def list_documents(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT id,name,status,error FROM documents")]

    def occurrences(self, doc_id: str | None = None):
        with self.db() as db:
            query = "SELECT o.*, d.name AS doc_name FROM occurrences o JOIN documents d ON d.id=o.doc_id WHERE d.generation<>''"
            rows = db.execute(query + (" AND o.doc_id=?" if doc_id else ""), (doc_id,) if doc_id else ())
            return [{**dict(row), "location": json.loads(row["location"])} for row in rows]

    async def index(self, doc_id: str, name: str, path: Path, provider: MultimodalProvider,
                    cancelled=lambda: False, progress=None, valid=lambda: True, chunks=None):
        started = time.perf_counter()
        calls_before = dict(getattr(provider, "calls", {})) if isinstance(getattr(provider, "calls", None), dict) else {}
        revision = hashlib.sha256(path.read_bytes()).hexdigest()
        signature = json.dumps(provider.signature(), sort_keys=True)
        generation = uuid.uuid4().hex
        directory = self.root / "generations" / generation
        directory.mkdir(parents=True)
        warnings, unsupported, occurrences, vectors, texts = [], [], [], {}, {}
        with self.db() as db:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status='processing',error=''",
                       (doc_id, name, revision, "", signature, "processing", ""))
        try:
            extracted = await asyncio.to_thread(lambda: list(extract_images(path)))
            for index, (raw, location, context) in enumerate(extracted):
                if cancelled():
                    raise asyncio.CancelledError()
                if progress:
                    await progress(index, len(extracted))
                if raw is None:
                    unsupported.append(location["warning"])
                    continue
                try:
                    image_id = self.put(raw)
                    if image_id not in vectors:
                        with self.db() as db:
                            cached = db.execute("SELECT payload FROM derived WHERE image_id=? AND signature=?", (image_id, signature)).fetchone()
                        if cached:
                            payload = json.loads(cached[0])
                        else:
                            image = self.asset_path(image_id).read_bytes()
                            description = await provider.describe(image)
                            vector = await provider.embed(image=image)
                            derived_text = (description["ocr"] + "\n" + description["description"]).strip()
                            cfg = provider.models["embedding"]
                            # UTF-8 bytes upper-bound token count; cover all OCR text, never a truncated prefix.
                            size = max(1, min(int(cfg.get("embed_max_chars", 480)), int(cfg.get("embed_max_tokens", 480)) // 4))
                            text_vectors = []
                            for offset in range(0, len(derived_text), size):
                                if cancelled():
                                    raise asyncio.CancelledError()
                                part = derived_text[offset:offset+size].strip()
                                if part:
                                    text_vectors.append(await provider.embed(text=part, visual=False))
                            payload = {**description, "vector": vector, "text_vectors": text_vectors}
                            with self.db() as db:
                                db.execute("INSERT OR REPLACE INTO derived VALUES(?,?,?)", (image_id, signature, json.dumps(payload)))
                        vectors[image_id] = payload["vector"]
                        for part, vector in enumerate(payload.get("text_vectors", [])):
                            texts[f"{image_id}:{part}"] = vector
                    related = []
                    for chunk in chunks or []:
                        chunk_text = str(chunk.get("text") or "").strip()
                        pages = (chunk.get("source_location") or {}).get("pages", [])
                        if (location.get("page") in pages or (len(chunk_text) >= 20 and chunk_text in context)):
                            related.append(chunk["chunk_id"])
                    location = {**location, "chunk_ids": related}
                    occurrences.append((doc_id, image_id, json.dumps(location), context[:12000]))
                except Exception as exc:
                    warnings.append(f"图片 {index+1}: {exc}")
            # Never replace a usable generation with incomplete extraction or model output.
            if warnings:
                raise ValueError("部分完成，未发布新版本：" + "；".join(warnings))
            for kind, values, purpose in (("visual", vectors, "multimodal_embedding"), ("text", texts, "embedding")):
                db = NanoVectorDB(provider.models[purpose]["embed_dim"], storage_file=str(directory / f"{kind}.json"))
                if values:
                    db.upsert([{"__id__": key, "__vector__": np.asarray(value)} for key, value in values.items()])
                db.save()
                vector_path = directory / f"{kind}.json"
                from src.index_validation import validate_vectors
                validate_vectors(json.loads(vector_path.read_text(encoding="utf-8")), str(vector_path.name))
                with vector_path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            if cancelled():
                raise asyncio.CancelledError()
            if not valid() or not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != revision:
                raise ValueError("文档已删除或发生变化，图片索引未发布")
            with self.db() as db:
                if not db.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone():
                    raise ValueError("文档已删除，图片索引未发布")
                db.execute("DELETE FROM occurrences WHERE doc_id=?", (doc_id,))
                db.executemany("INSERT INTO occurrences VALUES(?,?,?,?)", occurrences)
                db.execute("UPDATE documents SET name=?,revision=?,generation=?,signature=?,status=?,error=? WHERE id=?",
                           (name, revision, generation, signature, "partial" if unsupported else "indexed", "；".join(unsupported), doc_id))
            # Publication succeeded; a housekeeping failure must not roll it back.
            try:
                self.collect()
            except (OSError, sqlite3.Error):
                logging.getLogger(__name__).exception("Image generation published; obsolete file cleanup deferred")
            return {"images": len(vectors), "occurrences": len(occurrences), "status": "partial" if unsupported else "indexed", "warnings": unsupported,
                    "seconds": round(time.perf_counter()-started, 3),
                    "api_calls": {k: provider.calls[k]-v for k,v in calls_before.items()}}
        except BaseException as exc:
            shutil.rmtree(directory, ignore_errors=True)
            with self.db() as db:
                db.execute("UPDATE documents SET status=?,error=? WHERE id=?",
                           ("cancelled" if isinstance(exc, asyncio.CancelledError) else "partial" if warnings else "failed", str(exc), doc_id))
            raise

    async def search(self, provider: MultimodalProvider, text="", image: bytes | None = None, top_k=10):
        with self.db() as db:
            documents = [dict(r) for r in db.execute("SELECT * FROM documents WHERE generation<>''")]
        if not documents:
            return []
        signature = json.dumps(provider.signature(), sort_keys=True)
        if any(d["signature"] != signature for d in documents):
            raise ValueError("图片索引模型或预处理配置已变化，请重建全部图片索引")
        channels = []
        if image is not None:
            channels.append(("visual", await provider.embed(image=image), "multimodal_embedding"))
        if text.strip() and image is None:
            channels.extend([("visual", await provider.embed(text=text), "multimodal_embedding"),
                             ("text", await provider.embed(text=text, visual=False), "embedding")])
        # Re-read after network awaits: a generation may have been published or
        # deleted meanwhile. Everything below is synchronous on the event loop.
        with self.db() as db:
            documents = [dict(r) for r in db.execute("SELECT * FROM documents WHERE generation<>''")]
        if any(d["signature"] != signature for d in documents):
            raise ValueError("图片索引模型或预处理配置已变化，请重建全部图片索引")
        scores = {}
        similarities = {}
        for kind, vector, purpose in channels:
            hits = {}
            for document in documents:
                path = self.root / "generations" / document["generation"] / f"{kind}.json"
                if not path.is_file():
                    raise ValueError("图片索引文件缺失，请重建图片索引")
                db = NanoVectorDB(provider.models[purpose]["embed_dim"], storage_file=str(path))
                for hit in db.query(np.asarray(vector), top_k=top_k * (10 if kind == "text" else 1)):
                    key = hit["__id__"].split(":", 1)[0]
                    hits[key] = max(hits.get(key, -1), float(hit["__metrics__"]))
            for rank, (key, _) in enumerate(sorted(hits.items(), key=lambda pair: pair[1], reverse=True), 1):
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                similarities.setdefault(key, {})[kind] = hits[key]
        sources = self.occurrences()
        result = []
        with self.db() as db:
            for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
                row = db.execute("SELECT payload FROM derived WHERE image_id=? AND signature=?", (key, signature)).fetchone()
                description = json.loads(row[0]) if row else {}
                result.append({"image_id": key, "score": score, "ocr": description.get("ocr", ""),
                               "similarities": similarities.get(key, {}),
                               "description": description.get("description", ""),
                               "sources": [s for s in sources if s["image_id"] == key]})
        return result

    def remove_document(self, doc_id):
        with self.db() as db:
            db.execute("DELETE FROM occurrences WHERE doc_id=?", (doc_id,))
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.collect()

    def invalidate(self, doc_id):
        with self.db() as db:
            db.execute("UPDATE documents SET status='stale',generation='' WHERE id=?", (doc_id,))

    def clear(self):
        with self.db() as db:
            db.execute("DELETE FROM occurrences")
            db.execute("DELETE FROM documents")
        self.collect()

    def collect(self):
        with self.db() as db:
            orphaned = [r[0] for r in db.execute("SELECT id FROM assets WHERE id NOT IN (SELECT image_id FROM occurrences)")]
            for key in orphaned:
                db.execute("DELETE FROM derived WHERE image_id=?", (key,))
                db.execute("DELETE FROM assets WHERE id=?", (key,))
                shutil.rmtree(self.asset_path(key).parent, ignore_errors=True)
            active = {r[0] for r in db.execute("SELECT generation FROM documents WHERE generation<>''")}
        directory = self.root / "generations"
        if directory.exists():
            for path in directory.iterdir():
                if path.is_dir() and path.name not in active:
                    shutil.rmtree(path)

    def attachment(self, raw: bytes):
        self.expire_attachments()
        image, _, _, _ = prepare_image(raw)
        key = uuid.uuid4().hex
        directory = self.root / "attachments"
        directory.mkdir(exist_ok=True)
        (directory / f"{key}.jpg").write_bytes(image)
        with self.db() as db:
            db.execute("INSERT INTO attachments VALUES(?,?,NULL)", (key, time.time()))
        return key

    def attachment_path(self, key, session_id=None):
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("无效附件 ID")
        with self.db() as db:
            row = db.execute("SELECT * FROM attachments WHERE id=?", (key,)).fetchone()
        if not row or (row["session_id"] and row["session_id"] != session_id) or (not row["session_id"] and time.time()-row["created"] > 86400):
            raise ValueError("附件不存在、已过期或不属于当前会话")
        return self.root / "attachments" / f"{key}.jpg"

    def bind_attachment(self, key, session_id):
        self.attachment_path(key, session_id)
        with self.db() as db:
            db.execute("UPDATE attachments SET session_id=? WHERE id=?", (session_id, key))

    def expire_attachments(self, session_id=None):
        with self.db() as db:
            rows = db.execute("SELECT id FROM attachments WHERE session_id=?" if session_id else
                              "SELECT id FROM attachments WHERE session_id IS NULL AND created<?",
                              (session_id,) if session_id else (time.time()-86400,)).fetchall()
            for row in rows:
                (self.root / "attachments" / f"{row[0]}.jpg").unlink(missing_ok=True)
                db.execute("DELETE FROM attachments WHERE id=?", (row[0],))
