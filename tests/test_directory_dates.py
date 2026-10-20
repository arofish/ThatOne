import unittest
from pathlib import Path

from photoindex.directory_dates import extract_date_candidates


class TestDirectoryDates(unittest.TestCase):

    def test_valid_dates(self):
        cases = [
            "2024-02-29/a.jpg",
            "2025-12-31/b.png",
            "2000-02-29/c.jpeg",
        ]

        for value in cases:
            with self.subTest(path=value):
                result = extract_date_candidates(Path(value))

                expected_date = Path(value).parent.name

                self.assertEqual(len(result), 1)
                self.assertEqual(
                    result[0]["date_value"],
                    expected_date
                )
                self.assertEqual(
                    result[0]["raw_fragment"],
                    expected_date
                )
                self.assertEqual(
                    result[0]["source_directory"],
                    expected_date
                )
    def test_invalid_dates_and_formats(self):
        invalid = [
            "2023-02-29/a.jpg",   # 非闰年
            "1900-02-29/a.jpg",   # 世纪年不是闰年
            "2024-13-01/a.jpg",   # 月份越界
            "2024-04-31/a.jpg",   # 日期越界
            "0000-01-01/a.jpg",   # 年份零
            "2024-2-9/a.jpg",     # 位数不符合要求
            "20240229/a.jpg",     # 没有分隔符
            "旅行_2024-02-29/a.jpg",
            "2024-02-29_已修/a.jpg",
            "２０２４-０２-２９/a.jpg",  # 非 ASCII 数字
        ]

        for value in invalid:
            with self.subTest(path=value):
                self.assertEqual(
                    extract_date_candidates(Path(value)),
                    []
                )

    def test_no_directory_date(self):
        cases = [
            "a.jpg",
            "旅行/初筛/a.jpg",
            "2024-02-29.jpg",
            "旅行/2024-02-29.jpg",
        ]

        for value in cases:
            with self.subTest(path=value):
                self.assertEqual(
                    extract_date_candidates(Path(value)),
                    []
                )
    def test_multiple_nested_dates(self):
        path = Path(
            "旅行/2024-02-29/挑选/2024-03-01/a.jpg"
        )

        result = extract_date_candidates(path)

        expected = [
            {
                "date_value": "2024-02-29",
                "raw_fragment": "2024-02-29",
                "source_directory": "旅行/2024-02-29",
            },
            {
                "date_value": "2024-03-01",
                "raw_fragment": "2024-03-01",
                "source_directory": (
                    "旅行/2024-02-29/挑选/2024-03-01"
                ),
            },
        ]

        self.assertEqual(result, expected)

    def test_same_date_different_sources(self):
        path = Path(
            "2024-02-29/挑选/2024-02-29/a.jpg"
        )

        result = extract_date_candidates(path)

        self.assertEqual(len(result), 2)

        self.assertEqual(
            [r["date_value"] for r in result],
            ["2024-02-29", "2024-02-29"]
        )

        self.assertEqual(
            [r["source_directory"] for r in result],
            [
                "2024-02-29",
                "2024-02-29/挑选/2024-02-29",
            ]
        )

    def test_invalid_input_paths(self):
        invalid = [
            Path("/tmp/2024-02-29/a.jpg"),
            Path("../2024-02-29/a.jpg"),
            Path("旅行/../2024-02-29/a.jpg"),
        ]

        for path in invalid:
            with self.subTest(path=str(path)):
                with self.assertRaises(ValueError):
                    extract_date_candidates(path)