# PhotoIndex

本地优先的 macOS 摄影作品索引与 AI 检索项目。采用边做边学的导师协作方式；当前处于 M0 开始前的工程初始化阶段。

## 当前结构

```text
PhotoIndex/                 # 当前 checkout 文件夹名称可以是 thatone
├── AGENTS.md               # 后续助手接手时的协作约定
├── README.md               # 运行入口与协作流程
├── pyproject.toml          # 项目信息、Python 要求与直接依赖
├── .gitignore              # 排除环境、缓存、私人样本和本地数据
├── photoindex/
│   └── __init__.py         # 业务包；下一步由用户新增 scanner.py
├── tests/
│   └── __init__.py         # 测试目录；尚无测试用例
└── docs/
    ├── PROJECT_CHARTER.md
    ├── DECISIONS.md
    ├── PROGRESS.md
    ├── LEARNING_LOG.md
    └── tasks/M0_01_SCANNER.md
```

不预建空的 AI、API、数据库或 SwiftUI 子项目。需要相应功能时再增加模块。

## Python 与依赖

当前机器已检查：Python 3.14.0、Git 2.50.1。项目声明 Python >=3.11；目前只检查了本机版本，未验证其他版本或未来 AI 库兼容性。

M0 无第三方运行或开发依赖：`pathlib`、`unittest`、`tempfile` 都来自标准库。`pyproject.toml` 的 `dependencies = []` 表示当前无需安装外部包。测试先用 `unittest`，暂不加 pytest、格式化工具或类型检查器。

首次在新 checkout 建立环境：

```sh
python3 -m venv .venv
```

每次打开新的终端，在仓库根目录激活环境：

```sh
source .venv/bin/activate
python --version
```

虚拟环境隔离这个项目后续安装的依赖，不包含在 Git 中。当前从仓库根目录直接运行，不需要 `pip install -e .`；可安装包和构建配置延期。

引入第三方库时，先说明实际用途，再记录直接依赖及可复现的版本解析；届时选择锁定工具并提交锁文件。不要把本机所有已安装包直接复制成项目需求。FastAPI/SQLite 持久化、图像库、NumPy 与模型框架在对应任务开始时评估，其中 SQLite 的 Python 接口来自标准库。

## 当前任务与验证

先阅读 [M0-01](docs/tasks/M0_01_SCANNER.md)，由用户实现单层扫描。

写好测试后，在仓库根目录运行：

```sh
python -m unittest discover -s tests -v
```

目前没有测试用例，运行显示 0 tests 不能视为扫描器通过。私人样本可放在被忽略的 `sample/`，自动化测试使用临时目录。

## Git 工作方式

当前使用本地 `main` 分支即可。按可解释的小任务提交，不必每写一行就提交，也不必等整个里程碑完成才提交。首次初始化提交只包含配置、文档与包目录。

日常步骤：

```sh
git status
git diff
# 实现和验证当前小步后，明确选择需要提交的文件
git add photoindex/scanner.py
git diff --cached
git commit -m "feat: add single-directory image discovery"
```

上面的扫描器提交命令是未来完成代码后的示例，当前文件尚不存在。提交前检查暂存区，确认没有原片、密钥、虚拟环境或本地数据库。

新功能需要隔离时再开分支；GitHub 远程仓库与 CI 后续按需求设置。进度文档记录任务与验收证据，Git 记录文件变化，两者用途不同。
