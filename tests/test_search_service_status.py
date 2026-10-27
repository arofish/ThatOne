import tempfile
import unittest
from pathlib import Path

from photoindex.search_service import check_original_status


class CheckOriginalStatusTests(unittest.TestCase):

    def test_existing_file_is_available(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jpg"
            path.write_bytes(b"test image bytes")

            self.assertEqual(
                check_original_status(str(path)),
                "available",
            )

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing.jpg"

            self.assertEqual(
                check_original_status(str(path)),
                "missing",
            )

    def test_directory_is_not_original(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(
                check_original_status(temp),
                "missing",
            )


if __name__ == "__main__":
    unittest.main()