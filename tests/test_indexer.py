import sqlite3
import unittest

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from photoindex import indexer
from photoindex.indexer import init_db, index_directory

class TestIndexer(unittest.TestCase):
    def db_state(self, db_path: Path):
        """重新打开数据库，读取全部 Sources 和 Assets。"""

        with closing(sqlite3.connect(db_path)) as conn:
            sources = conn.execute(
                """
                SELECT id, root_path
                FROM sources
                ORDER BY id
                """
            ).fetchall()

            assets = conn.execute(
                """
                SELECT id, source_id, path,
                       extension, size_bytes, mtime_ns
                FROM assets
                ORDER BY id
                """
            ).fetchall()

        return sources, assets

    def file_state(self, root: Path):
        """记录目录成员及文件内容、大小、修改时间。"""

        paths = list(root.rglob("*"))

        members = sorted(
            p.relative_to(root).as_posix()
            for p in paths
        )

        files = {
            p.relative_to(root).as_posix(): (
                p.read_bytes(),
                p.stat().st_size,
                p.stat().st_mtime_ns,
            )
            for p in paths
            if p.is_file()
        }

        return members, files

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

    def test_persistence_and_upsert(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            image = root / "a.JPG"
            image.write_bytes(b"original")

            # 第一次扫描
            source_id = index_directory(db_path, root)

            sources1, assets1 = self.db_state(db_path)

            self.assertEqual(len(sources1), 1)
            self.assertEqual(len(assets1), 1)

            self.assertEqual(source_id, sources1[0][0])
            self.assertEqual(assets1[0][1], source_id)

            # 路径应为相对路径
            self.assertEqual(assets1[0][2], "a.JPG")

            # 扩展名与大小正确
            self.assertEqual(assets1[0][3], ".jpg")
            self.assertEqual(assets1[0][4], len(b"original"))
            self.assertEqual(
                assets1[0][5],
                image.stat().st_mtime_ns
            )

            original_asset_id = assets1[0][0]

            # 修改图片内容
            image.write_bytes(b"modified image content")

            # 第二次扫描
            second_source_id = index_directory(db_path, root)

            sources2, assets2 = self.db_state(db_path)

            # Source ID 不变
            self.assertEqual(source_id, second_source_id)

            # Asset ID 不变
            self.assertEqual(assets2[0][0], original_asset_id)

            # 数据没有重复
            self.assertEqual(len(sources2), 1)
            self.assertEqual(len(assets2), 1)

            # 元数据更新
            self.assertEqual(
                assets2[0][4],
                len(b"modified image content")
            )
            self.assertEqual(
                assets2[0][5],
                image.stat().st_mtime_ns
            )

    def test_multiple_sources(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root_a = base / "photos_a"
            root_b = base / "photos_b"

            root_a.mkdir()
            root_b.mkdir()

            # 两个来源中具有相同相对路径的文件
            (root_a / "same.jpg").write_bytes(b"image A")
            (root_b / "same.jpg").write_bytes(b"image B")

            source_a = index_directory(db_path, root_a)
            source_b = index_directory(db_path, root_b)

            sources, assets = self.db_state(db_path)

            # 两个资料源应有不同 ID
            self.assertNotEqual(source_a, source_b)

            # 两个来源、两个 Asset
            self.assertEqual(len(sources), 2)
            self.assertEqual(len(assets), 2)

            actual = {
                (row[1], row[2])
                for row in assets
            }

            expected = {
                (source_a, "same.jpg"),
                (source_b, "same.jpg"),
            }

            self.assertEqual(actual, expected)

    def test_root_symlink(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            real_root = base / "real_root"
            real_root.mkdir()

            album = real_root / "album"
            album.mkdir()

            (album / "image.JPG").write_bytes(b"image")

            # 创建指向真实根目录的符号链接
            link_root = base / "link_root"
            link_root.symlink_to(
                real_root,
                target_is_directory=True
            )

            # 分别通过真实路径和链接路径扫描
            real_source_id = index_directory(
                db_path,
                real_root
            )

            link_source_id = index_directory(
                db_path,
                link_root
            )

            sources, assets = self.db_state(db_path)

            # 必须复用同一个 Source
            self.assertEqual(real_source_id, link_source_id)

            # 不应重复生成 Source 和 Asset
            self.assertEqual(len(sources), 1)
            self.assertEqual(len(assets), 1)

            # Source 保存真实根目录
            self.assertEqual(
                sources[0][1],
                str(real_root.resolve())
            )

            # Asset 保存相对路径
            self.assertEqual(
                assets[0][2],
                "album/image.JPG"
            )

    def test_original_files_unchanged(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            (root / "a.jpg").write_bytes(b"original image")
            (root / "notes.txt").write_bytes(b"some notes")

            album = root / "album"
            album.mkdir()

            (album / "b.png").write_bytes(b"nested image")

            # 记录文件系统状态
            before = self.file_state(root)

            # 执行两次索引
            index_directory(db_path, root)
            index_directory(db_path, root)

            # 再次记录文件系统状态
            after = self.file_state(root)

            # 文件内容、大小、修改时间、成员均不变
            self.assertEqual(before, after)

    def test_new_source_rollback(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            (root / "a.jpg").write_bytes(b"image")

            with patch.object(
                indexer,
                "read_metadata",
                side_effect=PermissionError("denied")
            ):
                with self.assertRaises(PermissionError):
                    index_directory(db_path, root)

            sources, assets = self.db_state(db_path)

            self.assertEqual(sources, [])
            self.assertEqual(assets, [])

    def test_foreign_key_enabled(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"

            with closing(sqlite3.connect(db_path)) as conn:
                init_db(conn)

                # 确认外键检查已经启用
                enabled = conn.execute(
                    "PRAGMA foreign_keys"
                ).fetchone()

                self.assertEqual(enabled, (1,))

                # 尝试引用不存在的 Source
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """
                        INSERT INTO assets (
                            source_id,
                            path,
                            extension,
                            size_bytes,
                            mtime_ns
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (999, "ghost.jpg", ".jpg", 100, 0)
                    )

                conn.rollback()

    def test_negative_size_rejected(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            source_id = index_directory(db_path, root)

            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """
                        INSERT INTO assets (
                            source_id,
                            path,
                            extension,
                            size_bytes,
                            mtime_ns
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            "invalid.jpg",
                            ".jpg",
                            -100,
                            0,
                        )
                    )

                conn.rollback()

    def test_repeated_initialization(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            # 创建一张图片并完成首次索引
            (root / "a.jpg").write_bytes(b"original image")
            source_id = index_directory(db_path, root)

            # 保存数据库原有状态
            before = self.db_state(db_path)

            # 连续两次初始化同一个数据库
            with closing(sqlite3.connect(db_path)) as conn:
                init_db(conn)
                init_db(conn)

                # 外键检查仍然开启
                self.assertEqual(
                    conn.execute("PRAGMA foreign_keys").fetchone(),
                    (1,)
                )

            # 重新打开数据库，读取状态
            after = self.db_state(db_path)

            # 原来的 Source 和 Asset 都不能丢失
            self.assertEqual(before, after)

            # 确认首次索引确实产生了数据
            self.assertEqual(len(after[0]), 1)
            self.assertEqual(len(after[1]), 1)
            self.assertEqual(after[0][0][0], source_id)

    def test_added_and_missing_files(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            # 第一轮：只有 a.jpg
            a = root / "a.jpg"
            a.write_bytes(b"image A")

            source_id = index_directory(db_path, root)

            sources1, assets1 = self.db_state(db_path)

            self.assertEqual(len(assets1), 1)
            self.assertEqual(assets1[0][2], "a.jpg")

            original_a_id = assets1[0][0]

            # 第二轮：新增 b.png
            b = root / "b.png"
            b.write_bytes(b"image B")

            second_id = index_directory(db_path, root)

            sources2, assets2 = self.db_state(db_path)

            self.assertEqual(second_id, source_id)
            self.assertEqual(len(assets2), 2)

            # 建立 path -> Asset ID 的映射
            asset_ids = {
                row[2]: row[0]
                for row in assets2
            }

            self.assertEqual(
                set(asset_ids),
                {"a.jpg", "b.png"}
            )

            # a.jpg 应继续保留原来的 Asset ID
            self.assertEqual(
                asset_ids["a.jpg"],
                original_a_id
            )

            # 第三轮：移走 a.jpg
            a.unlink()

            third_id = index_directory(db_path, root)

            sources3, assets3 = self.db_state(db_path)

            self.assertEqual(third_id, source_id)

            # 旧记录不应被自动删除
            self.assertEqual(assets3, assets2)
            self.assertEqual(sources3, sources2)

    def test_invalid_root_does_not_modify_database(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            # 创建一个合法资料源
            valid_root = base / "valid_photos"
            valid_root.mkdir()

            (valid_root / "a.jpg").write_bytes(b"image")

            index_directory(db_path, valid_root)

            # 保存已有数据库状态
            before = self.db_state(db_path)

            # 构造不存在的资料源
            missing_root = base / "not_existing"

            self.assertFalse(missing_root.exists())

            # 必须抛出 FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                index_directory(db_path, missing_root)

            # 检查数据库未发生变化
            after = self.db_state(db_path)

            self.assertEqual(before, after)

    def test_database_write_failure_rolls_back(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            # 第一轮：建立已经提交的 Asset
            a = root / "a.jpg"
            a.write_bytes(b"old")

            index_directory(db_path, root)

            # 记录第一轮数据库状态
            before = self.db_state(db_path)

            # 第二轮：修改 a.jpg，并新增 b.png
            a.write_bytes(b"updated data")
            b = root / "b.png"
            b.write_bytes(b"new image")

            # 创建数据库触发器
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("""
                    CREATE TRIGGER fail_second_asset_insert
                    BEFORE INSERT ON assets
                    WHEN NEW.path = 'b.png'
                    BEGIN
                        SELECT CASE
                            WHEN (
                                SELECT size_bytes
                                FROM assets
                                WHERE source_id = NEW.source_id
                                AND path = 'a.jpg'
                            ) = LENGTH('updated data')
                            THEN RAISE(
                                ABORT,
                                'second write failed'
                            )
                            ELSE RAISE(
                                ABORT,
                                'first update not observed'
                            )
                        END;
                    END;
                """)
                conn.commit()

            # 第二轮扫描必须在数据库写入时失败
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "second write failed"
            ):
                index_directory(db_path, root)

            # 重新打开数据库检查结果
            after = self.db_state(db_path)

            # 整批修改必须回滚
            self.assertEqual(before, after)

    def full_db_state(self, db_path: Path):
        with closing(sqlite3.connect(db_path)) as conn:
            sources = conn.execute(
                "SELECT * FROM sources ORDER BY id"
            ).fetchall()

            assets = conn.execute(
                "SELECT * FROM assets ORDER BY id"
            ).fetchall()

            candidates = conn.execute(
                """
                SELECT *
                FROM asset_date_candidates
                ORDER BY id
                """
            ).fetchall()

        return sources, assets, candidates


    def test_date_candidates_persistence(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"
            root.mkdir()

            (root / "a.jpg").write_bytes(b"image A")

            nested = root / "旅行/2024-02-29/挑选/2024-03-01"
            nested.mkdir(parents=True)
            (nested / "b.jpg").write_bytes(b"image B")

            source_id = index_directory(db_path, root)

            before = self.full_db_state(db_path)
            sources, assets, candidates = before

            # 一个 Source，两张 Asset，两个日期候选
            self.assertEqual(len(sources), 1)
            self.assertEqual(len(assets), 2)
            self.assertEqual(len(candidates), 2)

            # 查询候选对应的图片
            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT
                        a.path,
                        c.date_value,
                        c.raw_fragment,
                        c.source_directory
                    FROM asset_date_candidates AS c
                    JOIN assets AS a ON a.id = c.asset_id
                    ORDER BY c.source_directory
                    """
                ).fetchall()

            self.assertEqual(rows, [
                (
                    "旅行/2024-02-29/挑选/2024-03-01/b.jpg",
                    "2024-02-29",
                    "2024-02-29",
                    "旅行/2024-02-29",
                ),
                (
                    "旅行/2024-02-29/挑选/2024-03-01/b.jpg",
                    "2024-03-01",
                    "2024-03-01",
                    "旅行/2024-02-29/挑选/2024-03-01",
                ),
            ])

            # 重复扫描
            second_source_id = index_directory(db_path, root)
            after = self.full_db_state(db_path)

            self.assertEqual(second_source_id, source_id)

            # Asset 记录不变
            self.assertEqual(before[1], after[1])

            # 候选数量不增加
            self.assertEqual(len(after[2]), 2)

    def test_candidates_multiple_sources(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root_a = base / "source_a"
            root_b = base / "source_b"

            for root in (root_a, root_b):
                folder = root / "2024-02-29"
                folder.mkdir(parents=True)
                (folder / "a.jpg").write_bytes(b"image")

            source_a = index_directory(db_path, root_a)
            source_b = index_directory(db_path, root_b)

            self.assertNotEqual(source_a, source_b)

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT
                        a.source_id,
                        a.id,
                        a.path,
                        c.date_value
                    FROM assets AS a
                    JOIN asset_date_candidates AS c
                        ON a.id = c.asset_id
                    ORDER BY a.source_id
                    """
                ).fetchall()

            self.assertEqual(len(rows), 2)

            # 同名路径属于两个不同的 Asset
            self.assertNotEqual(rows[0][1], rows[1][1])

            self.assertEqual(
                {row[0] for row in rows},
                {source_a, source_b}
            )

            for row in rows:
                self.assertEqual(row[2], "2024-02-29/a.jpg")
                self.assertEqual(row[3], "2024-02-29")

    def test_candidate_insert_failure_rolls_back(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "index.db"

            root = base / "photos"

            folder_a = root / "2024-02-29"
            folder_a.mkdir(parents=True)

            a = folder_a / "a.jpg"
            a.write_bytes(b"old")

            index_directory(db_path, root)

            # 获取已有 Asset ID，并人为插入过时候选
            with closing(sqlite3.connect(db_path)) as conn:
                asset_id = conn.execute(
                    """
                    SELECT id FROM assets
                    WHERE path = ?
                    """,
                    ("2024-02-29/a.jpg",)
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO asset_date_candidates (
                        asset_id, date_value,
                        raw_fragment, source_directory
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        "2020-01-01",
                        "2020-01-01",
                        "旧目录/2020-01-01",
                    )
                )
                conn.commit()

            # 保存数据库状态
            before = self.full_db_state(db_path)

            # 第二次扫描前修改 a，并新增 b
            new_content = b"updated image A"
            a.write_bytes(new_content)

            folder_b = root / "2024-03-01"
            folder_b.mkdir()

            (folder_b / "b.png").write_bytes(b"image B")

            # 注入数据库层的错误
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("""
                    CREATE TRIGGER fail_candidate_insert
                    BEFORE INSERT ON asset_date_candidates
                    WHEN NEW.source_directory = '2024-03-01'
                    BEGIN
                        SELECT CASE
                            WHEN (
                                SELECT size_bytes
                                FROM assets
                                WHERE path = '2024-02-29/a.jpg'
                            ) = LENGTH('updated image A')
                            AND NOT EXISTS (
                                SELECT 1
                                FROM asset_date_candidates
                                WHERE asset_id = (
                                    SELECT id FROM assets
                                    WHERE path = '2024-02-29/a.jpg'
                                )
                                AND raw_fragment = '2020-01-01'
                            )
                            THEN RAISE(
                                ABORT,
                                'expected candidate failure'
                            )
                            ELSE RAISE(
                                ABORT,
                                'previous operations not observed'
                            )
                        END;
                    END;
                """)
                conn.commit()

            # 候选插入必须抛出预期错误
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "expected candidate failure"
            ):
                index_directory(db_path, root)

            # 重新建立连接，确认全部回滚
            after = self.full_db_state(db_path)

            self.assertEqual(before, after)