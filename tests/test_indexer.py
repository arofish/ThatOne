import sqlite3
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from contextlib import closing

from photoindex import indexer


class TestIndexer(unittest.TestCase):

    def test_rollback(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)

            root = base / "photos"
            root.mkdir()

            db_path = base / "index.db"

            # 第一次成功索引
            a = root / "a.jpg"
            a.write_bytes(b"old")

            indexer.index_directory(db_path, root)

            # 查询数据库当前状态
            def read_assets():
                with closing(sqlite3.connect(db_path)) as conn:
                    return conn.execute(
                        """
                        SELECT id, path, size_bytes, mtime_ns
                        FROM assets
                        ORDER BY id
                        """
                    ).fetchall()

            before = read_assets()

            # 准备第二轮扫描
            a.write_bytes(b"modified content")
            (root / "b.png").write_bytes(b"new")
            (root / "c.jpeg").write_bytes(b"new")

            # 保留真实读取函数
            original = indexer.read_metadata

            def faulty_read(path):
                if path.name == "c.jpeg":
                    raise PermissionError("Simulated failure")

                return original(path)

            # 模拟第三张图片读取失败
            with patch.object(
                indexer,
                "read_metadata",
                side_effect=faulty_read
            ):
                with self.assertRaises(PermissionError):
                    indexer.index_directory(db_path, root)

            # 数据库状态应完全不变
            after = read_assets()

            self.assertEqual(before, after)