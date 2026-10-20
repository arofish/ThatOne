import sqlite3
from contextlib import closing
from pathlib import Path

from photoindex.directory_dates import extract_date_candidates
from photoindex.scanner import scan_images


def init_db(conn: sqlite3.Connection) -> None:
    """初始化数据库结构，并启用外键检查。"""

    # 必须在事务开始前启用外键
    if conn.in_transaction:
        raise RuntimeError(
            "init_db requires a connection without an active transaction"
        )

    conn.execute("PRAGMA foreign_keys = ON")

    enabled = conn.execute(
        "PRAGMA foreign_keys"
    ).fetchone()

    if enabled != (1,):
        raise RuntimeError("SQLite foreign keys are not enabled")

    # 让建表操作作为一个事务
    try:
        conn.execute("BEGIN")

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
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                mtime_ns INTEGER NOT NULL,

                FOREIGN KEY (source_id) REFERENCES sources(id),
                UNIQUE (source_id, path)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_date_candidates (
                id INTEGER PRIMARY KEY,
                asset_id INTEGER NOT NULL,
                date_value TEXT NOT NULL,
                raw_fragment TEXT NOT NULL,
                source_directory TEXT NOT NULL,

                FOREIGN KEY (asset_id) REFERENCES assets(id),

                UNIQUE (
                    asset_id,
                    source_directory,
                    raw_fragment
                )
            )
        """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise


def read_metadata(path: Path) -> tuple[str, int, int]:
    """提取图片扩展名、大小和修改时间。"""

    info = path.stat()

    return (
        path.suffix.lower(),
        info.st_size,
        info.st_mtime_ns,
    )


def index_directory(db_path: Path, root: Path) -> int:
    """索引指定目录，成功后返回 Source ID。"""

    # 1. 复用扫描器
    images = scan_images(root)

    # 2. 规范化资料源根路径
    canonical_root = root.resolve()

    # 3. 连接数据库
    with closing(sqlite3.connect(db_path)) as conn:

        # 初始化，包括外键检查和建表
        init_db(conn)

        # 4. 开始本批索引事务
        with conn:

            # 确保 Source 存在
            conn.execute(
                """
                INSERT INTO sources (root_path)
                VALUES (?)
                ON CONFLICT(root_path) DO NOTHING
                """,
                (str(canonical_root),)
            )

            # 获取 Source ID
            row = conn.execute(
                """
                SELECT id
                FROM sources
                WHERE root_path = ?
                """,
                (str(canonical_root),)
            ).fetchone()

            if row is None:
                raise RuntimeError("Source ID not found")

            source_id = row[0]

            # 5. 逐张图片插入或更新
            for image in images:

                extension, size, mtime = read_metadata(image)

                # 保存相对于 Source 的路径
                relative_path = image.relative_to(
                    canonical_root
                ).as_posix()

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
                        relative_path,
                        extension,
                        size,
                        mtime,
                    )
                )
                # 1. 获取刚插入或更新的 Asset ID
                row = conn.execute(
                    """
                    SELECT id
                    FROM assets
                    WHERE source_id = ?
                    AND path = ?
                    """,
                    (source_id, relative_path)
                ).fetchone()

                if row is None:
                    raise RuntimeError("Asset ID not found")

                asset_id = row[0]

                # 2. 从 Asset 相对路径中解析候选
                candidates = extract_date_candidates(
                    Path(relative_path)
                )

                # 3. 删除该 Asset 的旧候选
                conn.execute(
                    """
                    DELETE FROM asset_date_candidates
                    WHERE asset_id = ?
                    """,
                    (asset_id,)
                )

                # 4. 插入本轮所有候选
                for candidate in candidates:
                    conn.execute(
                        """
                        INSERT INTO asset_date_candidates (
                            asset_id,
                            date_value,
                            raw_fragment,
                            source_directory
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            asset_id,
                            candidate["date_value"],
                            candidate["raw_fragment"],
                            candidate["source_directory"],
                        )
                    )

    # 6. 整个事务成功提交后返回 Source ID
    return source_id