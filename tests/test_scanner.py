import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from photoindex.scanner import scan_images


class TestScanImages(unittest.TestCase):
    def test_single_directory_filtering(self):
        # 准备：所有测试数据都在新建的临时目录内，不操作 photos。
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("b.png", "a.JPG", "c.JpEg", "notes.txt"):
                (root / name).write_bytes(b"test")
            (root / "fake.jpg").mkdir()

            # 调用 + 断言：同时检查筛选、绝对路径与排序。
            expected = [root / "a.JPG", root / "b.png", root / "c.JpEg"]
            self.assertEqual(scan_images(root), expected)
        # 离开 with 后，刚才创建的临时目录和内容会自动清理。

    def test_empty_directory(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(scan_images(Path(directory)), [])

    def test_missing_root(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "not_created"
            # assertRaises 检查调用是否抛出指定类型的异常。
            with self.assertRaises(FileNotFoundError):
                scan_images(missing)

    def test_file_root(self):
        with TemporaryDirectory() as directory:
            file_root = Path(directory) / "a.jpg"
            file_root.write_bytes(b"test")
            with self.assertRaises(NotADirectoryError):
                scan_images(file_root)

    def test_root_metadata_permission_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # 临时替换实际读取操作，让它确定地抛错，不依赖本机权限。
            with patch.object(Path, "stat", side_effect=PermissionError("denied")) as mocked:
                with self.assertRaises(PermissionError):
                    scan_images(root)
                mocked.assert_called_once()
            # patch 已结束，恢复真实操作后才清理临时目录。

    def test_child_metadata_permission_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "a.jpg").write_bytes(b"test")
            # 当前只有一个子项：这次 lstat 失败必须暴露，不能返回 []。
            with patch.object(Path, "lstat", side_effect=PermissionError("denied")) as mocked:
                with self.assertRaises(PermissionError):
                    scan_images(root)
                mocked.assert_called_once()

    def test_directory_read_permission_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(Path, "iterdir", side_effect=PermissionError("denied")) as mocked:
                with self.assertRaises(PermissionError):
                    scan_images(root)
                mocked.assert_called_once()


    def test_nested_directories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()

            # 创建根目录中的文件
            (root / "A.JPG").touch()
            (root / "notes.txt").touch()

            # 创建 fake.jpg 目录及其子文件
            (root / "fake.jpg").mkdir()
            (root / "fake.jpg" / "inside.png").touch()

            # 创建多层嵌套目录
            (root / "album" / "deep").mkdir(parents=True)
            (root / "album" / "deep" / "b.jpeg").touch()

            # 创建隐藏目录
            (root / ".hidden").mkdir()
            (root / ".hidden" / "c.PNG").touch()

            # 调用待测试函数
            result = scan_images(root)

            # 验证结果
            expected = sorted(
                [
                    (root / "A.JPG").resolve(),
                    (root / "fake.jpg" / "inside.png").resolve(),
                    (root / "album" / "deep" / "b.jpeg").resolve(),
                    (root / ".hidden" / "c.PNG").resolve(),
                ],
                key=str,
            )
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
