import tempfile
import unittest
from pathlib import Path

from photoindex.cache_validation import (
    CacheValidationError,
    load_verified_v2_cache,
    project_path,
)


class CacheValidationTests(unittest.TestCase):

    def test_missing_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CacheValidationError):
                load_verified_v2_cache(Path(temp))

    def test_path_inside_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            boundary = root / "thumbnails"
            boundary.mkdir()

            result = project_path(
                root,
                "thumbnails/asset-1.png",
                boundary,
            )

            self.assertEqual(
                result,
                boundary / "asset-1.png",
            )

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            boundary = root / "thumbnails"
            boundary.mkdir()

            with self.assertRaises(CacheValidationError):
                project_path(
                    root,
                    "../outside.png",
                    boundary,
                )


if __name__ == "__main__":
    unittest.main()