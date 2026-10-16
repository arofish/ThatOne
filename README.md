# ThatOne / PhotoIndex

本地优先的摄影作品索引与 AI 检索项目，面向 macOS 摄影工作流，目标是通过自然语言检索本地照片，并逐步提供原生搜索界面。

项目处于早期开发阶段。目前已实现 Python 只读候选图片扫描器；数据库索引、AI 检索和 macOS 界面尚未实现。

## 当前功能

- 递归扫描目录，识别 `.jpg`、`.jpeg`、`.png`，扩展名大小写不敏感。
- 包含隐藏目录，跳过子树中的符号链接；允许根路径本身是目录链接。
- 返回按完整路径字符串排序的绝对路径列表。
- 无效根目录或读取失败时抛出异常。
- 只读访问，不修改、移动或删除原始照片。

扫描器根据扩展名发现候选文件，不解码图片，也不验证图片内容是否有效。完整边界测试与 M0 验收仍在进行中。

## 环境准备

项目要求 Python 3.11 或更高版本，目前在 Python 3.14.0 下运行过测试。当前仅使用 Python 标准库，无需安装第三方依赖。

在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
```

## 使用

在仓库根目录启动 Python，调用扫描接口：

```python
from pathlib import Path
from photoindex.scanner import scan_images

images = scan_images(Path("/path/to/photos"))
for image in images:
    print(image)
```

当前提供 Python 函数接口，尚无命令行工具或图形界面。

## 测试

在仓库根目录、已激活虚拟环境的终端执行：

```sh
python -m unittest discover -s tests -v
```

测试使用临时目录创建样本，不依赖真实摄影库。当前用例覆盖单层筛选、多层与隐藏目录、空目录、无效根路径，以及受控读取错误；其他边界用例仍在补充。

## 项目结构

```text
photoindex/
    scanner.py         # 只读候选图片扫描器
tests/
    test_scanner.py    # 自动化测试
docs/                  # 设计、进度与开发文档
pyproject.toml         # 项目信息与 Python 版本要求
```

## 路线图

1. 完善只读扫描器及边界测试。
2. 使用 SQLite 建立可重复扫描、更新的本地 Asset 索引。
3. 验证中文图文语义检索的效果与性能。
4. 接入最小 SwiftUI 搜索界面。

后续计划包含混合检索、增量索引、RAW 与 Apple Photos 支持，以及摄影 Agent，具体范围随验证结果调整。

## 数据原则

原始照片与索引、缓存和模型文件分开保存。云端 AI 默认关闭；涉及数据外发的功能需要明确授权。私人样本、本地数据和环境文件由 `.gitignore` 排除，应保存在约定目录中。
