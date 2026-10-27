# ThatOne / PhotoIndex

本地优先的摄影作品索引与 AI 检索项目，面向 macOS 摄影工作流，目标是通过自然语言检索本地照片，并逐步提供原生搜索界面。

项目处于早期开发阶段。目前已实现 Python 只读候选图片扫描器、SQLite 本地 Asset 索引、目录日期候选，以及本地中文图文检索。50 图、12 条查询的 CPU 开发基线已通过验收，下一阶段接入本地搜索 API 与最小 SwiftUI 界面；服务和界面尚未实现。

## 当前功能

- 递归扫描目录，识别 `.jpg`、`.jpeg`、`.png`，扩展名大小写不敏感。
- 包含隐藏目录，跳过子树中的符号链接；允许根路径本身是目录链接。
- 返回按完整路径字符串排序的绝对路径列表。
- 无效根目录或读取失败时抛出异常。
- 只读访问，不修改、移动或删除原始照片。
- 按资料源与相对路径建立 SQLite 索引，重复扫描更新元数据并保留 Asset ID；失败时回滚本轮写入。
- 从资料根内完整命名为 `YYYY-MM-DD` 的目录提取合法日期候选，支持多个候选并保留来源。候选不自动等同于拍摄日期。
- 使用固定 Chinese-CLIP 模型编码中文，与缓存图片向量做精确余弦排序。本批验证范围为 50 张样本，不代表整个图库的效果；相似度不保证全部查询条件满足，也未实现无匹配拒答。

扫描器根据扩展名发现候选文件，不解码图片，也不验证图片内容是否有效；图文检索实验另行读取准备好的 PNG 衍生图。

## 环境准备

项目要求 Python 3.11 或更高版本，目前在 Python 3.14.0 下运行过测试。扫描、索引与评测指标仅使用 Python 标准库，无需安装第三方依赖；可选 AI 冒烟实验使用独立环境。

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

当前通过 Python 函数与实验脚本运行，尚无本地 HTTP 服务或图形界面。

## 可选：Chinese-CLIP 本地冒烟实验

AI 环境使用 Python 3.13 / macOS arm64。直接依赖位于 `requirements-ai.in`，确切版本与哈希位于 `requirements-ai.lock.txt`。安装示例（需要 uv 和 Python 3.13）：

```sh
uv venv --python 3.13 .venv-ai
uv pip sync --python .venv-ai/bin/python --require-hashes requirements-ai.lock.txt
.venv-ai/bin/python scripts/download_chinese_clip_http.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-ai/bin/python scripts/smoke_chinese_clip.py
```

下载器固定模型 revision，支持断点续传；已有完整 `.part` 会先核对 SHA-256，再改为正式权重文件名。缓存完整时不需要联网。不能直接将未校验的 `.part` 改名后使用。模型路径相对于项目目录解析，不依赖当前终端目录。

冒烟只使用合成图片与一句中文，在 CPU/float32 下验证两个 `(1, 512)` 向量、数值与相似度，不读取私人图库。固定使用慢速预处理器，避免默认处理方式变化。本批 Transformers 4.57.1 的 `get_text_features()` 存在空池化输出问题，冒烟使用该版本可运行的完整图文前向接口。

模型权重、环境和运行数据不提交 Git。冒烟通过只说明本机加载与推理可用，不代表真实检索质量或性能评测已通过。

## 测试

在仓库根目录、已激活虚拟环境的终端执行：

```sh
python -m unittest discover -s tests -v
```

测试使用临时目录创建样本，不依赖真实摄影库。当前用例覆盖扫描筛选、嵌套目录、链接、无效路径、读取错误，以及索引持久化、重复扫描更新、资料源隔离、约束、事务回滚与原片不变。

## 项目结构

```text
photoindex/
    scanner.py         # 只读候选图片扫描器
    indexer.py         # SQLite 资料源与 Asset 索引
    directory_dates.py # 目录日期候选解析
tests/
    test_scanner.py    # 自动化测试
    test_indexer.py    # 索引与事务测试
    test_directory_dates.py # 日期解析测试
pyproject.toml         # 项目信息与 Python 版本要求
```

## 路线图

1. 只读扫描器与 SQLite Asset 索引：已实现，可重复扫描更新。
2. 中文图文语义检索：固定小样本基线已通过质量和性能验收。
3. 下一步：提供常驻本地搜索 API，接入 SwiftUI 搜索、缩略图展示和 Finder 原片定位。

后续计划包含混合检索、增量索引、RAW 与 Apple Photos 支持，以及摄影 Agent，具体范围随验证结果调整。

## 数据原则

原始照片与索引、缓存和模型文件分开保存。云端 AI 默认关闭；涉及数据外发的功能需要明确授权。私人样本、本地数据和环境文件由 `.gitignore` 排除，应保存在约定目录中。
