import os
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
    def test_symlink_handling(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            base = Path(tmp)

            # 创建实际文件和目录
            dir_link = base / "dir_link"
            (root / "real_image.jpg").touch()
            (root / "real_dir").mkdir()
            (root / "real_dir" / "nested_image.png").touch()

            # 创建符号链接
            dir_link.symlink_to(root)
            (root / "symlink_to_image.jpg").symlink_to(root / "real_image.jpg")
            (root / "symlink_to_dir").symlink_to(root / "real_dir")
            (root / "symlink_to_parent").symlink_to(root)
            (root / "symlink_to_nonexistent").symlink_to(root / "nonexistent_file.jpg")

            # 调用待测试函数
            result = scan_images(root)
            link_result = scan_images(dir_link)

            # 验证结果，符号链接不应被包含
            expected = sorted(
                [
                    (root / "real_image.jpg").resolve(),
                    (root / "real_dir" / "nested_image.png").resolve(),
                ],
                key=str,
            )
            self.assertEqual(result, expected)
            self.assertEqual(link_result, expected)
    def test_related_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()

            (root / "image2.png").touch()
            (root / "image1.jpg").touch()

            # 构造相对路径（相对于当前工作目录）
            relative_root = Path(os.path.relpath(root, Path.cwd()))

            # 重复调用
            result1 = scan_images(relative_root)
            result2 = scan_images(relative_root)

            # 1. 返回绝对路径
            for path in result1:
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.exists())

            # 2. 两次调用结果一致
            self.assertEqual(result1, result2)

            # 3. 结果有序
            self.assertEqual(result1, sorted(result1, key=str))

            # 4. 结果无重复
            self.assertEqual(len(result1), len(set(result1)))

            # 5. 结果与预期一致
            expected = [
                (root / "image1.jpg").resolve(),
                (root / "image2.png").resolve()
            ]
            self.assertEqual(result1, expected)
    def test_symlinks_outside_root(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)

            # 创建两个兄弟目录
            root = base / "root"
            outside = base / "outside"

            root.mkdir()
            outside.mkdir()

            # 扫描根内的真实图片
            local_image = root / "local.jpg"
            local_image.write_bytes(b"local")

            # 扫描根外的图片
            external_image = outside / "external.jpg"
            external_image.write_bytes(b"external")

            external_dir = outside / "album"
            external_dir.mkdir()

            nested_external_image = external_dir / "external.png"
            nested_external_image.write_bytes(b"external nested")

            # 创建指向根外的文件链接
            file_link = root / "external.jpg"
            file_link.symlink_to(external_image)

            # 创建指向根外的目录链接
            dir_link = root / "external_album"
            dir_link.symlink_to(
                external_dir,
                target_is_directory=True
            )

            # 确认链接创建成功
            self.assertTrue(file_link.is_symlink())
            self.assertTrue(dir_link.is_symlink())

            # 执行扫描
            result = scan_images(root)

            # 只应该包含真实的本地图片
            expected = [local_image.resolve()]

            self.assertEqual(result, expected)

            # 明确验证根外图片没有进入结果
            self.assertNotIn(external_image.resolve(), result)
            self.assertNotIn(nested_external_image.resolve(), result)
    def test_hidden_image_n_unsupported_extensions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            unsupported_extensions = Path(tmp) / "unsupported"
            unsupported_extensions.mkdir()
            (unsupported_extensions / "file.txt").touch()
            (unsupported_extensions / "file.gif").touch()
            (root / ".hidden_image.jpg").touch()

            expected = [(root / ".hidden_image.jpg").resolve()]
            self.assertEqual(scan_images(root), expected)
            self.assertEqual(scan_images(unsupported_extensions),[])


    def test_nested_directory_permission_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            root = root.resolve()

            # 根目录图片
            image = root / "image.jpg"
            image.write_bytes(b"image")

            # 正常子目录
            normal = root / "normal"
            normal.mkdir()
            (normal / "normal.jpg").write_bytes(b"normal")

            # 模拟无权限访问的子目录
            blocked = root / "blocked"
            blocked.mkdir()
            (blocked / "secret.jpg").write_bytes(b"secret")

            # 保存真实的 iterdir 方法
            original_iterdir = Path.iterdir

            # 用于确认异常模拟确实被触发
            blocked_calls = []

            def fake_iterdir(path):
                # 保证根目录按指定顺序产出子项
                if path == root:
                    return iter([image, normal, blocked])

                # 只在遍历 blocked 时抛出异常
                if path == blocked:
                    blocked_calls.append(path)
                    raise PermissionError("Cannot read nested directory")

                # 其他目录正常遍历
                return original_iterdir(path)

            # 使用 mock 替换 iterdir
            with patch.object(
                Path,
                "iterdir",
                autospec=True,
                side_effect=fake_iterdir
            ) as mocked:

                # 必须抛出指定异常，而不是返回部分结果
                with self.assertRaisesRegex(
                    PermissionError,
                    "Cannot read nested directory"
                ):
                    scan_images(root)

            # 确认确实尝试枚举 blocked 目录
            self.assertEqual(blocked_calls, [blocked])
            mocked.assert_any_call(blocked)

    def test_read_only_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()

            read_only_image = root / "read_only_image.jpg"
            read_only_image.touch()
            # 设置文件为只读
            read_only_image.chmod(0o444)

            # 调用 scan_images，应该能正常读取只读文件
            result = scan_images(root)
            expected = [read_only_image.resolve()]
            self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
