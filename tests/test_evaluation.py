import unittest

from photoindex.evaluation import evaluate_at_k


class TestEvaluation(unittest.TestCase):

    def test_task_example(self):
        # 任务单中的标准示例
        ranked = [3, 1, 2, 4, 5]
        relevant = {1, 4, 8}

        result = evaluate_at_k(ranked, relevant)

        self.assertEqual(
            set(result),
            {"precision", "recall", "hit"}
        )

        self.assertAlmostEqual(
            result["precision"], 0.4
        )
        self.assertAlmostEqual(
            result["recall"], 2 / 3
        )
        self.assertEqual(
            result["hit"], 1.0
        )

    def test_no_hits(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {6, 7}

        result = evaluate_at_k(ranked, relevant)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["hit"], 0.0)

    def test_all_results_relevant(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}

        result = evaluate_at_k(ranked, relevant)

        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["hit"], 1.0)

    def test_relevant_result_after_k(self):
        ranked = [1, 2, 3, 4, 5, 6]
        relevant = {6}

        result = evaluate_at_k(ranked, relevant, k=5)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["hit"], 0.0)

    def test_fewer_results_than_k(self):
        ranked = [1, 2]
        relevant = {1, 2, 3}

        result = evaluate_at_k(ranked, relevant, k=5)

        # Precision 分母固定为 5
        self.assertAlmostEqual(result["precision"], 0.4)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["hit"], 1.0)

    def test_empty_ranking(self):
        ranked = []
        relevant = {1, 2}

        result = evaluate_at_k(ranked, relevant)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["hit"], 0.0)

    def test_no_matching_query(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = set()

        result = evaluate_at_k(ranked, relevant)

        self.assertEqual(result["precision"], 0.0)
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["hit"])

    def test_empty_ranking_and_no_matches(self):
        result = evaluate_at_k([], set())

        self.assertEqual(result["precision"], 0.0)
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["hit"])

    def test_duplicate_ids(self):
        ranked = [1, 2, 1, 3, 4]

        with self.assertRaises(ValueError):
            evaluate_at_k(ranked, {1, 2})

    def test_duplicate_after_k(self):
        ranked = [1, 2, 3, 4, 5, 1]

        with self.assertRaises(ValueError):
            evaluate_at_k(ranked, {1}, k=5)

    def test_invalid_k(self):
        invalid_values = [
            0,
            -1,
            1.5,
            1.0,
            "5",
            None,
            True,
            False,
        ]

        for k in invalid_values:
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    evaluate_at_k([1, 2], {1}, k=k)

    def test_custom_k(self):
        ranked = [1, 2, 3, 4]
        relevant = {2, 4}

        result = evaluate_at_k(ranked, relevant, k=3)

        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertEqual(result["hit"], 1.0)

    def test_inputs_not_modified(self):
        ranked = [3, 1, 2, 4, 5]
        relevant = {1, 4, 8}

        original_ranked = ranked.copy()
        original_relevant = relevant.copy()

        evaluate_at_k(ranked, relevant)

        self.assertEqual(ranked, original_ranked)
        self.assertEqual(relevant, original_relevant)


if __name__ == "__main__":
    unittest.main()