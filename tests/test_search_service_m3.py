
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photoindex.search_service import (
    PreviewChangedError,
    SearchService,
    SearchValidationError,
    ThumbnailNotFoundError,
)


class FakeTokenizer:
    model_max_length = 8

    def __call__(
        self,
        value,
        *,
        add_special_tokens=True,
        truncation=False,
    ):
        assert add_special_tokens is True
        assert truncation is False

        return {
            "input_ids": (
                [101] + list(range(len(value))) + [102]
            )
        }


class FakeProcessor:
    tokenizer = FakeTokenizer()


class SearchServiceTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        original = self.root / "a.jpg"
        original.write_bytes(b"test original")

        missing = self.root / "missing.jpg"

        thumbnail = self.root / "one.png"
        thumbnail.write_bytes(b"png v1")
        self.thumbnail = thumbnail

        image_vectors = np.zeros(
            (2, 512),
            dtype=np.float32,
        )

        image_vectors[0, 0] = 1.0
        image_vectors[1, 1] = 1.0

        digest = hashlib.sha256(
            b"png v1"
        ).hexdigest()

        self.service = SearchService(
            model=object(),
            processor=FakeProcessor(),
            asset_ids=(1, 2),
            image_vectors=image_vectors,
            asset_paths={
                1: ("a.jpg", str(original)),
                2: ("missing.jpg", str(missing)),
            },
            thumbnails={
                1: (thumbnail, digest),
                2: (thumbnail, digest),
            },
            run_id="test-v2",
            text_token_limit=8,
        )

    def test_two_queries_keep_missing_original(self):
        query_vector = np.zeros(
            (1, 512),
            dtype=np.float32,
        )
        query_vector[0, 1] = 1.0

        with (
            patch(
                "photoindex.search_service.encode_texts",
                return_value=query_vector,
            ) as encoder,
            patch(
                "photoindex.retrieval.encode_images",
                side_effect=AssertionError(
                    "Search must not re-encode images"
                ),
            ) as image_encoder,
        ):
            first = self.service.search("hello")
            second = self.service.search("hello")

        self.assertEqual(
            [r.asset_id for r in first.results],
            [2, 1],
        )

        self.assertEqual(
            first.results[0].original_status,
            "missing",
        )

        self.assertEqual(encoder.call_count, 2)
        image_encoder.assert_not_called()

        self.assertEqual(
            second.run_id,
            "test-v2",
        )

    def test_invalid_queries(self):
        invalid = [
            "",
            " \t ",
            "x" * 1001,
            "x" * 7,
            None,
        ]

        for query in invalid:
            with self.subTest(query=str(query)[:20]):
                with self.assertRaises(
                    SearchValidationError
                ):
                    self.service.search(query)

    def test_original_query_is_not_rewritten(self):
        query_vector = np.zeros(
            (1, 512),
            dtype=np.float32,
        )
        query_vector[0, 0] = 1.0

        with patch(
            "photoindex.search_service.encode_texts",
            return_value=query_vector,
        ) as encoder:
            result = self.service.search(" 你好 ")

        self.assertEqual(result.query, " 你好 ")
        encoder.assert_called_once_with(
            [" 你好 "],
            self.service.model,
            self.service.processor,
            "cpu",
        )

    def test_thumbnail_integrity(self):
        self.assertEqual(
            self.service.thumbnail_bytes(1),
            b"png v1",
        )

        with self.assertRaises(
            ThumbnailNotFoundError
        ):
            self.service.thumbnail_bytes(99)

        self.thumbnail.write_bytes(b"png v2")

        with self.assertRaises(
            PreviewChangedError
        ):
            self.service.thumbnail_bytes(1)

        self.thumbnail.unlink()

        with self.assertRaises(
            ThumbnailNotFoundError
        ):
            self.service.thumbnail_bytes(1)


if __name__ == "__main__":
    unittest.main()
