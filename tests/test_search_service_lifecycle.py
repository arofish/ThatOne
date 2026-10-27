
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch

import numpy as np

from photoindex.cache_validation import (
    CacheValidationError,
    VerifiedCache,
)
from photoindex.search_service import SearchService


def fake_data():
    vectors = np.zeros((2, 512), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0

    data = VerifiedCache(
        run_id="fixture-v2",
        model_dir=Path("/unused-model-dir"),
        asset_ids=(1, 2),
        image_vectors=vectors,
        asset_paths={
            1: ("one.jpg", "/nonexistent/one.jpg"),
            2: ("two.jpg", "/nonexistent/two.jpg"),
        },
        thumbnails={
            1: (
                Path("/nonexistent/one.png"),
                "0" * 64,
            ),
            2: (
                Path("/nonexistent/two.png"),
                "0" * 64,
            ),
        },
    )

    model = MagicMock()
    model.to.return_value = model
    model.config.text_config.max_position_embeddings = 64

    processor = MagicMock()
    processor.tokenizer.model_max_length = 64
    processor.tokenizer.return_value = {
        "input_ids": [101, 23, 102],
    }

    text_vector = np.zeros(
        (1, 512),
        dtype=np.float32,
    )
    text_vector[0, 0] = 1.0

    return data, model, processor, text_vector


class LifecycleTests(unittest.TestCase):

    def test_load_once_and_reuse_for_two_queries(self):
        data, model, processor, text_vector = fake_data()

        with (
            patch(
                "photoindex.search_service.load_verified_v2_cache",
                return_value=data,
            ) as loader,

            patch(
                "transformers.ChineseCLIPModel.from_pretrained",
                return_value=model,
            ) as load_model,

            patch(
                "transformers.ChineseCLIPProcessor.from_pretrained",
                return_value=processor,
            ) as load_processor,

            patch(
                "photoindex.search_service.encode_texts",
                return_value=text_vector,
            ) as encoder,

            patch(
                "photoindex.retrieval.encode_images",
                side_effect=AssertionError(
                    "Unexpected image encoding"
                ),
            ) as image_encoder,
        ):
            service = SearchService.from_project(
                Path("/not-a-real-project")
            )

            first = service.search("第一条")
            second = service.search("第二条")

        loader.assert_called_once()
        load_model.assert_called_once()
        load_processor.assert_called_once()

        self.assertEqual(encoder.call_count, 2)
        image_encoder.assert_not_called()

        self.assertEqual(first.results[0].asset_id, 1)
        self.assertEqual(second.results[0].asset_id, 1)
        self.assertEqual(first.run_id, "fixture-v2")

    def test_invalid_cache_fails_before_model_load(self):
        with (
            patch(
                "photoindex.search_service.load_verified_v2_cache",
                side_effect=CacheValidationError("broken"),
            ) as loader,

            patch(
                "transformers.ChineseCLIPModel.from_pretrained",
            ) as load_model,

            patch(
                "transformers.ChineseCLIPProcessor.from_pretrained",
            ) as load_processor,
        ):
            with self.assertRaises(CacheValidationError):
                SearchService.from_project(
                    Path("/not-a-real-project")
                )

        loader.assert_called_once()
        load_model.assert_not_called()
        load_processor.assert_not_called()

    def test_concurrent_searches_are_serialized(self):
        data, model, processor, text_vector = fake_data()

        service = SearchService(
            model=model,
            processor=processor,
            asset_ids=data.asset_ids,
            image_vectors=data.image_vectors,
            asset_paths=data.asset_paths,
            thumbnails=data.thumbnails,
            run_id=data.run_id,
            text_token_limit=64,
        )

        guard = Lock()
        active = 0
        peak = 0

        def encode_stub(*args):
            nonlocal active, peak

            with guard:
                active += 1
                peak = max(peak, active)

            try:
                time.sleep(0.01)
                return text_vector.copy()
            finally:
                with guard:
                    active -= 1

        with patch(
            "photoindex.search_service.encode_texts",
            side_effect=encode_stub,
        ) as encoder:

            with ThreadPoolExecutor(
                max_workers=4
            ) as pool:
                responses = list(
                    pool.map(
                        service.search,
                        ["你好"] * 12,
                    )
                )

        self.assertEqual(encoder.call_count, 12)
        self.assertEqual(len(responses), 12)
        self.assertEqual(peak, 1)

        self.assertTrue(
            all(
                response.results[0].asset_id == 1
                for response in responses
            )
        )


if __name__ == "__main__":
    unittest.main()
