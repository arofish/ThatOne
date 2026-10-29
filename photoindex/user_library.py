
"""v1.0.0: mutable offline photo catalog. Do not import frozen M2 outputs here.

Copy into user's repository manually after reviewing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from photoindex.indexer import index_directory, init_db
from photoindex.scanner import scan_images
from photoindex.retrieval import encode_images, encode_texts, rank_images
from photoindex.search_service import (
    SearchResult, SearchResponse, SearchValidationError,
    ThumbnailNotFoundError, PreviewChangedError, ThumbnailReadError,
    check_original_status,
)

RUN_ID = "thatone-library-v1"
MODEL_WEIGHTS_SHA256 = "7b7b583c210c867410bc6bdb8a55fe14eec62999e0a9ea31ff222dc501f9cfbe"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IndexedImage:
    asset_id: int
    relative_path: str
    original_path: str
    preview_path: Path
    preview_sha256: str


@dataclass(frozen=True)
class Snapshot:
    ids: tuple[int, ...]
    vectors: np.ndarray
    items: dict[int, IndexedImage]


class LibraryBusyError(RuntimeError):
    pass


class LibraryIndexError(RuntimeError):
    pass


class UserLibraryService:
    """One process, one model; SQLite publishes completed source scans atomically."""

    def __init__(
        self, *, data_root: Path, model, processor,
        image_encoder: Callable = encode_images,
        text_encoder: Callable = encode_texts,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.db = self.data_root / "library.db"
        self.preview_dir = self.data_root / "thumbnails"
        self.preview_dir.mkdir(exist_ok=True)
        self.model = model
        self.processor = processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.run_id = RUN_ID
        self._model_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thatone-index")
        self._jobs: dict[str, dict] = {}
        self._active_job: str | None = None
        self._init_db()
        self._snapshot = self._load_snapshot()

    @classmethod
    def from_release(cls, *, model_dir: Path, data_root: Path) -> "UserLibraryService":
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        model_dir = Path(model_dir).resolve(strict=True)
        # Preserve the M2 model identity; never silently download a substitute.
        manifest_path = model_dir.parent / "model-files.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("model_id") != "OFA-Sys/chinese-clip-vit-base-patch16"
                or manifest.get("model_revision") != "36e679e65c2a2fead755ae21162091293ad37834"):
            raise LibraryIndexError("Bundled model identity mismatch")
        expected = manifest.get("model_files_sha256", {})
        for name in ("config.json", "preprocessor_config.json", "vocab.txt", "pytorch_model.bin"):
            if name not in expected or hash_file(model_dir / name) != expected[name]:
                raise LibraryIndexError(f"Bundled model file mismatch: {name}")
        if expected["pytorch_model.bin"] != MODEL_WEIGHTS_SHA256:
            raise LibraryIndexError("Bundled weights differ from M2 baseline")
        processor = ChineseCLIPProcessor.from_pretrained(
            str(model_dir), use_fast=False, local_files_only=True,
        )
        model = ChineseCLIPModel.from_pretrained(str(model_dir), local_files_only=True)
        model.to("cpu", dtype=torch.float32)
        model.eval()
        return cls(data_root=data_root, model=model, processor=processor)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db, timeout=30)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            with conn:
                yield conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            init_db(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    asset_id INTEGER PRIMARY KEY REFERENCES assets(id),
                    file_sha256 TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    preview_name TEXT NOT NULL,
                    preview_sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0,1))
                )
            """)

    def _load_snapshot(self) -> Snapshot:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT i.asset_id, a.path, s.root_path,
                       i.vector_blob, i.preview_name, i.preview_sha256
                FROM image_index i
                JOIN assets a ON a.id=i.asset_id
                JOIN sources s ON s.id=a.source_id
                WHERE i.active=1 ORDER BY i.asset_id
            """).fetchall()
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        items: dict[int, IndexedImage] = {}
        for asset_id, relative, source_string, blob, preview_name, preview_hash in rows:
            source = Path(source_string).resolve(strict=False)
            rel = Path(relative)
            original = (source / rel).resolve(strict=False)
            if rel.is_absolute() or not original.is_relative_to(source):
                raise LibraryIndexError("Source path escapes root")
            if Path(preview_name).name != preview_name:
                raise LibraryIndexError("Unsafe preview filename")
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4):
                raise LibraryIndexError(f"Invalid vector for Asset {asset_id}")
            ids.append(asset_id)
            vectors.append(vector)
            items[asset_id] = IndexedImage(
                asset_id, relative, str(original),
                self.preview_dir / preview_name, preview_hash,
            )
        matrix = np.stack(vectors).astype(np.float32) if vectors else np.empty((0,512), dtype=np.float32)
        matrix.setflags(write=False)
        return Snapshot(tuple(ids), matrix, items)

    @property
    def asset_ids(self) -> tuple[int, ...]:
        with self._state_lock:
            return self._snapshot.ids

    def library_status(self) -> dict:
        with self._state_lock:
            snapshot_count = len(self._snapshot.ids)
            jobs = [dict(j) for j in self._jobs.values()]
        with self._connect() as conn:
            sources = [dict(id=row[0], path=row[1]) for row in conn.execute("SELECT id, root_path FROM sources ORDER BY id")]
        # Offline sources are retained; never infer deletion from inaccessible root.
        for source in sources:
            source["available"] = Path(source["path"]).is_dir()
        return {"run_id": RUN_ID, "asset_count": snapshot_count, "sources": sources, "jobs": jobs[-5:]}

    def start_import(self, root: Path) -> str:
        selected = Path(root).expanduser().resolve(strict=True)
        if not selected.is_dir():
            raise NotADirectoryError(selected)
        if selected == self.data_root or selected.is_relative_to(self.data_root):
            raise ValueError("Do not import the application's index directory")
        with self._connect() as conn:
            prior = [Path(r[0]).resolve(strict=False) for r in conn.execute("SELECT root_path FROM sources")]
        for existing in prior:
            if selected != existing and (selected.is_relative_to(existing) or existing.is_relative_to(selected)):
                raise ValueError("Overlapping source directories are not supported")
        with self._state_lock:
            if self._active_job is not None:
                raise LibraryBusyError("An import is already running")
            job_id = uuid.uuid4().hex
            self._active_job = job_id
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "scanned": 0, "reused": 0, "encoded": 0, "removed": 0, "error": None}
        self._executor.submit(self._run_job, job_id, selected)
        return job_id

    def job_status(self, job_id: str) -> dict | None:
        with self._state_lock:
            job = self._jobs.get(job_id)
            return None if job is None else dict(job)

    def _update_job(self, job_id: str, **fields) -> None:
        with self._state_lock:
            self._jobs[job_id].update(fields)

    def _run_job(self, job_id: str, root: Path) -> None:
        try:
            self._update_job(job_id, status="scanning")
            counts = self._reindex_source(root, job_id)
            self._update_job(job_id, status="completed", **counts)
        except Exception as exc:
            # Avoid showing private absolute file paths in the client.
            self._update_job(job_id, status="failed", error=type(exc).__name__)
        finally:
            with self._state_lock:
                self._active_job = None

    def _reindex_source(self, root: Path, job_id: str) -> dict:
        # scan_images raises for offline roots / permission failures. No deletion occurs.
        files = scan_images(root)
        self._update_job(job_id, scanned=len(files), status="indexing")
        # Reuse M1's stable Source/Asset IDs and date candidate logic.
        source_id = index_directory(self.db, root)
        with self._connect() as conn:
            path_to_id = dict(conn.execute("SELECT path,id FROM assets WHERE source_id=?", (source_id,)).fetchall())
            existing = {
                row[0]: (row[1], row[2], row[3])
                for row in conn.execute("""
                    SELECT i.asset_id, i.file_sha256, i.preview_name, i.preview_sha256
                    FROM image_index i JOIN assets a ON a.id=i.asset_id
                    WHERE a.source_id=?
                """, (source_id,)).fetchall()
            }
        present_ids: set[int] = set()
        reused_ids: list[int] = []
        changed: list[tuple[int, Path, str]] = []
        for file_path in files:
            relative = file_path.relative_to(root).as_posix()
            asset_id = path_to_id[relative]
            present_ids.add(asset_id)
            digest = hash_file(file_path)  # correct even when mtime/size are unchanged
            record = existing.get(asset_id)
            preview_ok = False
            if record is not None and record[0] == digest:
                old_preview = self.preview_dir / record[1]
                preview_ok = (
                    old_preview.is_file()
                    and hash_file(old_preview) == record[2]
                )
            if preview_ok:
                reused_ids.append(asset_id)
            else:
                changed.append((asset_id, file_path, digest))
        self._update_job(job_id, status="encoding", reused=len(reused_ids))
        staged: list[tuple[int, str, bytes, str, str, Path]] = []
        with TemporaryDirectory(dir=self.data_root, prefix=".staging-") as temp:
            tmp = Path(temp)
            png_paths: list[Path] = []
            for asset_id, file_path, before_hash in changed:
                png_path = tmp / f"asset-{asset_id}.png"
                with Image.open(file_path) as source:
                    source = ImageOps.exif_transpose(source)
                    image = source.convert("RGB")
                    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
                    image.save(png_path, format="PNG")
                if hash_file(file_path) != before_hash:
                    raise LibraryIndexError("Original changed during indexing")
                png_paths.append(png_path)
            if png_paths:
                with self._model_lock:
                    vectors = self.image_encoder(png_paths, self.model, self.processor, "cpu")
                if vectors.shape != (len(changed),512) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                    raise LibraryIndexError("Invalid image embeddings")
                if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
                    raise LibraryIndexError("Non-unit image embeddings")
                for (asset_id, _, file_hash), vector, png_path in zip(changed, vectors, png_paths):
                    preview_hash = hash_file(png_path)
                    preview_name = f"asset-{asset_id}-{preview_hash}.png"
                    staged.append((asset_id, file_hash, vector.tobytes(), preview_name, preview_hash, png_path))
            self._update_job(job_id, status="publishing", encoded=len(staged))
            # Immutable content-addressed PNG; orphan on crash is harmless.
            for _,_,_,preview_name,_,src in staged:
                target = self.preview_dir / preview_name
                if not target.exists():
                    os.replace(src, target)
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""
                    UPDATE image_index SET active=0
                    WHERE asset_id IN (SELECT id FROM assets WHERE source_id=?)
                """, (source_id,))
                conn.executemany("UPDATE image_index SET active=1 WHERE asset_id=?", [(i,) for i in reused_ids])
                conn.executemany("""
                    INSERT INTO image_index (
                        asset_id,file_sha256,vector_blob,preview_name,preview_sha256,indexed_at,active
                    ) VALUES (?,?,?,?,?,?,1)
                    ON CONFLICT(asset_id) DO UPDATE SET
                    file_sha256=excluded.file_sha256,
                    vector_blob=excluded.vector_blob,
                    preview_name=excluded.preview_name,
                    preview_sha256=excluded.preview_sha256,
                    indexed_at=excluded.indexed_at,
                    active=1
                """, [(asset_id,digest,sqlite3.Binary(vec),name,png_sha,utc_now()) for asset_id,digest,vec,name,png_sha,_ in staged])
                removed = conn.execute("""
                    SELECT COUNT(*) FROM image_index i
                    JOIN assets a ON a.id=i.asset_id
                    WHERE a.source_id=? AND i.active=0
                """, (source_id,)).fetchone()[0]
                conn.commit()
        refreshed = self._load_snapshot()
        with self._state_lock:
            self._snapshot = refreshed
        return {"scanned": len(files), "reused":len(reused_ids), "encoded":len(staged), "removed":removed}

    def search(self, query: str) -> SearchResponse:
        if not isinstance(query,str) or not query.strip() or len(query)>1000:
            raise SearchValidationError("Invalid query")
        with self._state_lock:
            snap = self._snapshot
        if not snap.ids:
            return SearchResponse(query, RUN_ID, [])
        encoded = self.processor.tokenizer(query, add_special_tokens=True, truncation=False)
        max_length = self.model.config.text_config.max_position_embeddings
        if len(encoded["input_ids"]) > max_length:
            raise SearchValidationError("Query exceeds token limit")
        with self._model_lock:
            q = self.text_encoder([query],self.model,self.processor,"cpu")
            if q.shape != (1,512):
                raise LibraryIndexError("Wrong text vector shape")
            top5 = rank_images(list(snap.ids), snap.vectors, q[0])[:5]
        results = []
        for asset_id, score in top5:
            item = snap.items[asset_id]
            results.append(SearchResult(
                asset_id=asset_id, score=score, relative_path=item.relative_path,
                original_path=item.original_path,
                original_status=check_original_status(item.original_path),
                thumbnail_url=f"/assets/{asset_id}/thumbnail",
            ))
        return SearchResponse(query,RUN_ID,results)

    def thumbnail_bytes(self, asset_id:int) -> bytes:
        with self._state_lock:
            item = self._snapshot.items.get(asset_id)
        if item is None:
            raise ThumbnailNotFoundError("Unknown image")
        try:
            data = item.preview_path.read_bytes()
        except FileNotFoundError as exc:
            raise ThumbnailNotFoundError("Missing preview") from exc
        except OSError as exc:
            raise ThumbnailReadError("Cannot read preview") from exc
        if hash_bytes(data) != item.preview_sha256:
            raise PreviewChangedError("Preview differs from published index")
        return data

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
