
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photoindex import cache_validation as cv


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


class CacheFixtures(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name).resolve()
        self.retrieval = self.root / "local-data/retrieval"

        self.run = (
            self.retrieval
            / "runs"
            / cv.RUN_ID
        )

        self.model_dir = (
            self.root
            / "local-data/models/chinese-clip-base"
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.run.mkdir(parents=True, exist_ok=True)

        self.ids = list(range(1, 51))

        self.manifest_path = (
            self.retrieval / "manifest.json"
        )
        self.images_path = self.retrieval / "images.json"
        self.ids_path = self.run / "asset_ids.json"
        self.meta_path = self.run / "run_metadata.json"
        self.lock_path = (
            self.root
            / "local-data/models/chinese-clip-base-lock.json"
        )
        self.vectors_path = self.run / "image_vectors.npy"
        self.db_path = self.root / "local-data/photoindex.db"

        self.source = self.root / "originals"
        self.source.mkdir()

        manifest = {
            "dataset_id": "fixture-dataset",
            "source_id": 7,
            "asset_ids": self.ids,
            "queries": [],
        }

        put_json(self.manifest_path, manifest)
        put_json(self.ids_path, self.ids)

        image_rows = []
        input_rows = []

        for asset_id in self.ids:
            relative = (
                "local-data/retrieval/thumbnails/"
                f"asset-{asset_id}.png"
            )

            preview = self.root / relative
            preview.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            data = (
                b"PNG fixture "
                + str(asset_id).encode("ascii")
            )
            preview.write_bytes(data)

            image_rows.append({
                "asset_id": asset_id,
                "png_path": relative,
            })

            input_rows.append({
                "asset_id": asset_id,
                "png_path": relative,
                "width": 9,
                "height": 9,
                "sha256": digest(data),
            })

        put_json(
            self.images_path,
            {"images": image_rows},
        )

        vectors = np.zeros(
            (50, 512),
            dtype=np.float32,
        )

        for i in range(50):
            vectors[i, i] = 1.0

        np.save(self.vectors_path, vectors)

        self.vector_hash = digest(
            self.vectors_path.read_bytes()
        )
        self.manifest_hash = digest(
            self.manifest_path.read_bytes()
        )
        self.weights_hash = digest(b"fake weights")

        files = {
            "config.json": b"config",
            "preprocessor_config.json": b"processor",
            "vocab.txt": b"vocab",
            "pytorch_model.bin": b"fake weights",
        }

        file_hashes = {}

        for name, data in files.items():
            (self.model_dir / name).write_bytes(data)
            file_hashes[name] = digest(data)

        self.metadata = {
            "run_id": cv.RUN_ID,
            "model_id": cv.MODEL_ID,
            "model_revision": cv.MODEL_REVISION,
            "dataset_id": "fixture-dataset",
            "manifest_sha256": self.manifest_hash,
            "images_json_sha256": digest(
                self.images_path.read_bytes()
            ),
            "image_vectors_sha256": self.vector_hash,
            "device": "cpu",
            "dtype": "float32",
            "vector_shape": [50, 512],
            "model_weights_sha256": self.weights_hash,
            "processor_config_sha256": file_hashes[
                "preprocessor_config.json"
            ],
            "model_files_sha256": file_hashes,
            "inputs": input_rows,
        }

        put_json(self.meta_path, self.metadata)

        put_json(self.lock_path, {
            "model_id": cv.MODEL_ID,
            "revision": cv.MODEL_REVISION,
            "weight_sha256": self.weights_hash,
        })

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE sources (
                    id INTEGER PRIMARY KEY,
                    root_path TEXT NOT NULL
                );

                CREATE TABLE assets (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER,
                    path TEXT NOT NULL
                );
                """
            )

            conn.execute(
                "INSERT INTO sources VALUES (?, ?)",
                (7, str(self.source)),
            )

            conn.executemany(
                "INSERT INTO assets VALUES (?, ?, ?)",
                [
                    (i, 7, f"pic-{i}.jpg")
                    for i in self.ids
                ],
            )

        patches = [
            patch.object(
                cv, "MANIFEST_SHA256", self.manifest_hash
            ),
            patch.object(
                cv, "VECTORS_SHA256", self.vector_hash
            ),
            patch.object(
                cv, "WEIGHTS_SHA256", self.weights_hash
            ),
        ]

        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def write_metadata(self):
        put_json(self.meta_path, self.metadata)

    def test_valid_fixture_allows_missing_originals(self):
        result = cv.load_verified_v2_cache(self.root)

        self.assertEqual(len(result.asset_ids), 50)
        self.assertEqual(
            result.image_vectors.shape,
            (50, 512),
        )
        self.assertFalse(
            result.image_vectors.flags.writeable
        )
        self.assertEqual(
            result.asset_paths[1][0],
            "pic-1.jpg",
        )
        self.assertEqual(len(result.thumbnails), 50)

    def test_wrong_revision_fails(self):
        self.metadata["model_revision"] = "not-m2"
        self.write_metadata()

        with self.assertRaisesRegex(
            cv.CacheValidationError, "revision"
        ):
            cv.load_verified_v2_cache(self.root)

    def test_duplicate_asset_ids_fails(self):
        ids = list(self.ids)
        ids[1] = ids[0]
        put_json(self.ids_path, ids)

        with self.assertRaises(cv.CacheValidationError):
            cv.load_verified_v2_cache(self.root)

    def test_shuffled_rows_fails(self):
        ids = list(self.ids)
        ids[0], ids[1] = ids[1], ids[0]
        put_json(self.ids_path, ids)

        with self.assertRaisesRegex(
            cv.CacheValidationError, "row order"
        ):
            cv.load_verified_v2_cache(self.root)

    def test_preview_tampering_fails(self):
        preview = (
            self.root
            / self.metadata["inputs"][0]["png_path"]
        )

        preview.write_bytes(b"replaced preview")

        with self.assertRaisesRegex(
            cv.CacheValidationError, "SHA-256 mismatch"
        ):
            cv.load_verified_v2_cache(self.root)

    def test_preview_escape_fails(self):
        relative = "local-data/outside.png"
        preview = self.root / relative
        preview.write_bytes(b"outside")

        images = json.loads(
            self.images_path.read_text(encoding="utf-8")
        )

        images["images"][0]["png_path"] = relative
        put_json(self.images_path, images)

        self.metadata["inputs"][0]["png_path"] = relative
        self.metadata["inputs"][0]["sha256"] = digest(
            b"outside"
        )
        self.metadata["images_json_sha256"] = digest(
            self.images_path.read_bytes()
        )
        self.write_metadata()

        with self.assertRaisesRegex(
            cv.CacheValidationError, "escapes"
        ):
            cv.load_verified_v2_cache(self.root)

    def test_original_escape_fails(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE assets SET path=? WHERE id=1",
                ("../outside.jpg",),
            )

        with self.assertRaisesRegex(
            cv.CacheValidationError, "escapes"
        ):
            cv.load_verified_v2_cache(self.root)

    def test_weight_tampering_fails(self):
        (
            self.model_dir / "pytorch_model.bin"
        ).write_bytes(b"new fake weights")

        with self.assertRaisesRegex(
            cv.CacheValidationError, "SHA-256 mismatch"
        ):
            cv.load_verified_v2_cache(self.root)


if __name__ == "__main__":
    unittest.main()
