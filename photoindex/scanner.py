import stat
from pathlib import Path

def scan_images(root: Path) -> list[Path]:
    """扫描直接子项中的候选图片"""
    # strict=True：不存在就抛错；同时转为绝对路径，允许根是目录链接。
    root = root.resolve(strict=True)

    # stat() 读取元数据。读取失败时让异常向调用方传播，不转成 False。
    # st_mode 包含文件类型；S_ISDIR 用它判断根是不是目录。
    if not stat.S_ISDIR(root.stat().st_mode):
        raise NotADirectoryError(root)

    allowed = {".jpg", ".jpeg", ".png"}
    pending = [root]
    images = []
    while pending:
        current = pending.pop()
        for item in current.iterdir():
            # lstat() 读取子项自身，不跟随链接；失效链接也能识别并跳过。
            mode = item.lstat().st_mode
            if stat.S_ISLNK(mode):
                # 跳过符号链接
                continue
            if stat.S_ISDIR(mode):
                pending.append(item)
            if stat.S_ISREG(mode) and item.suffix.lower() in allowed:
                # root 已规范化，因此它的直接子项已经是绝对路径。
                images.append(item)

    # 只有全部读取成功才会到这里；中途失败不会返回部分结果。
    return sorted(images, key=str)
