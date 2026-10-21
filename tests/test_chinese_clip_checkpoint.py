import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.chinese_clip_checkpoint import (
    MODEL_ID, REQUIRED_FILES, REVISION, check_revision,
    finalize_weight, require_model_files, write_model_lock,
)


class TestChineseClipCheckpoint(unittest.TestCase):
    def test_verified_partial_is_promoted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "pytorch_model.bin.part"
            partial.write_bytes(b"complete fixture")
            digest = hashlib.sha256(b"complete fixture").hexdigest()
            target = finalize_weight(root, digest)
            self.assertEqual(target.read_bytes(), b"complete fixture")
            self.assertFalse(partial.exists())

    def test_invalid_partial_preserves_both_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "pytorch_model.bin"
            partial = root / "pytorch_model.bin.part"
            target.write_bytes(b"old")
            partial.write_bytes(b"incomplete")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                finalize_weight(root, hashlib.sha256(b"complete").hexdigest())
            self.assertEqual(partial.read_bytes(), b"incomplete")
            self.assertEqual(target.read_bytes(), b"old")

    def test_verified_target_needs_no_partial(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "pytorch_model.bin"
            target.write_bytes(b"complete")
            digest = hashlib.sha256(b"complete").hexdigest()
            self.assertEqual(finalize_weight(root, digest), target)

    def test_partial_is_not_a_loadable_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REQUIRED_FILES[:-1]:
                (root / name).write_text("fixture", encoding="utf-8")
            (root / "pytorch_model.bin.part").write_bytes(b"incomplete")
            with self.assertRaisesRegex(FileNotFoundError, r"\.part download"):
                require_model_files(root)

    def test_empty_file_prevents_completion_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "model-lock.json"
            for name in REQUIRED_FILES:
                (root / name).write_bytes(b"fixture")
            (root / "vocab.txt").write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "vocab.txt"):
                write_model_lock(root, lock)
            self.assertFalse(lock.exists())

    def test_completion_records_revision_and_removes_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REQUIRED_FILES:
                (root / name).write_bytes(b"fixture")
            state = root / ".download-revision"
            state.write_text(REVISION, encoding="utf-8")
            lock = root / "model-lock.json"
            write_model_lock(root, lock)
            saved = json.loads(lock.read_text(encoding="utf-8"))
            self.assertEqual(saved["model_id"], MODEL_ID)
            self.assertEqual(saved["revision"], REVISION)
            self.assertFalse(state.exists())

    def test_different_revision_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".download-revision").write_text("other", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different revision"):
                check_revision(root, root / "model-lock.json")
