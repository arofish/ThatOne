import re

from datetime import date
from pathlib import Path


DATE_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
)


def extract_date_candidates(
    relative_path: Path
) -> list[dict[str, str]]:

    # 1. 输入必须是资料源内的相对路径
    if relative_path.is_absolute():
        raise ValueError("Absolute paths are not allowed")

    if ".." in relative_path.parts:
        raise ValueError("Parent traversal is not allowed")

    # 2. 只取父目录部分
    parent_parts = relative_path.parent.parts

    candidates = []
    visited_parts = []

    # 3. 从外层到内层遍历父目录
    for part in parent_parts:

        # 维护当前目录的完整相对位置
        visited_parts.append(part)

        # 4. 检查精确日期格式
        if DATE_PATTERN.fullmatch(part) is None:
            continue

        # 5. 检查真实日期是否合法
        try:
            parsed_date = date.fromisoformat(part)
        except ValueError:
            continue

        # 6. 保存候选和来源
        candidates.append({
            "date_value": parsed_date.isoformat(),
            "raw_fragment": part,
            "source_directory": "/".join(visited_parts),
        })

    return candidates