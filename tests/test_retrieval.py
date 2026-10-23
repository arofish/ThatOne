import importlib.util
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "Requires NumPy")
class TestRanking(unittest.TestCase):

    def test_ranking_and_tie_breaking(self):
        from photoindex.retrieval import rank_images

        # 生成三个 512 维单位向量
        vectors = np.zeros((3, 512), dtype=np.float32)

        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        vectors[2, 0] = 1.0

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0

        results = rank_images(
            asset_ids=[20, 30, 10],
            image_vectors=vectors,
            query_vector=query,
        )

        # Asset 10 和 20 同分，按 ID 升序
        self.assertEqual(
            [asset_id for asset_id, _ in results],
            [10, 20, 30]
        )

        self.assertEqual(
            [score for _, score in results],
            [1.0, 1.0, 0.0]
        )

    def test_duplicate_ids(self):
        from photoindex.retrieval import rank_images

        vectors = np.zeros((2, 512), dtype=np.float32)
        vectors[:, 0] = 1.0

        query = vectors[0]

        with self.assertRaises(ValueError):
            rank_images([1, 1], vectors, query)

    def test_invalid_shape(self):
        from photoindex.retrieval import rank_images

        vectors = np.ones((2, 512), dtype=np.float32)
        query = np.ones(128, dtype=np.float32)

        with self.assertRaises(ValueError):
            rank_images([1, 2], vectors, query)

    def test_zero_vector(self):
        from photoindex.retrieval import rank_images

        vectors = np.zeros((1, 512), dtype=np.float32)

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0

        with self.assertRaises(ValueError):
            rank_images([1], vectors, query)


@unittest.skipUnless(
    np is not None
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None,
    "Requires AI environment",
)
class TestTextEncoding(unittest.TestCase):
    def test_text_encoding_matches_joint_forward_without_pooler(self):
        """A tiny random model checks the real API without downloading weights."""
        import torch
        from transformers import ChineseCLIPConfig, ChineseCLIPModel
        from photoindex.retrieval import encode_texts

        torch.manual_seed(0)
        config = ChineseCLIPConfig(
            text_config={
                "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "max_position_embeddings": 16,
            },
            vision_config={
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "image_size": 8, "patch_size": 4,
            },
            projection_dim=512,
        )
        model = ChineseCLIPModel(config).eval()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

        def processor(**kwargs):
            return inputs

        actual = encode_texts(["中文查询"], model, processor, "cpu")
        with torch.inference_mode():
            self.assertIsNone(model.text_model(**inputs).pooler_output)
            expected = model(
                **inputs, pixel_values=torch.zeros((1, 3, 8, 8)),
                return_dict=True,
            ).text_embeds.cpu().numpy()
        self.assertEqual(actual.shape, (1, 512))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
