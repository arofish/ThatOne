import sqlite3

from pathlib import Path
from contextlib import closing

from photoindex.scanner import scan_images

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            extension TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,

            FOREIGN KEY (source_id) REFERENCES sources(id),
            UNIQUE (source_id, path)
        )
    """)

def read_metadata(path: Path) -> tuple[str, int, int]:
    info = path.stat()

    return (
        path.suffix.lower(),
        info.st_size,
        info.st_mtime_ns,
    )

def index_directory(db_path: Path, root: Path) -> int:
    # 1. 扫描图片
    images = scan_images(root)

    # 2. 打开数据库连接
    with closing(sqlite3.connect(db_path)) as conn:

        # 3. 启用外键约束
        conn.execute("PRAGMA foreign_keys = ON")

        # 4. 初始化数据库结构
        init_db(conn)
        conn.commit()

        # 5. 开始本批索引事务
        with conn:

            # 规范化根目录路径
            root_path = str(root.resolve())

            # 确保 Source 存在
            conn.execute(
                """
                INSERT INTO sources (root_path)
                VALUES (?)
                ON CONFLICT(root_path) DO NOTHING
                """,
                (root_path,)
            )

            # 获取 Source ID
            row = conn.execute(
                "SELECT id FROM sources WHERE root_path = ?",
                (root_path,)
            ).fetchone()

            if row is None:
                raise RuntimeError("Source ID not found")

            source_id = row[0]

            # 逐张处理图片
            for image in images:

                # 获取文件元数据
                extension, size, mtime = read_metadata(image)

                # 插入或更新 Asset
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

                    ON CONFLICT(source_id, path)
                    DO UPDATE SET
                        extension = excluded.extension,
                        size_bytes = excluded.size_bytes,
                        mtime_ns = excluded.mtime_ns
                    """,
                    (
                        source_id,
                        str(image),
                        extension,
                        size,
                        mtime
                    )
                )

    # 6. 返回成功索引的图片数量
    return len(images)