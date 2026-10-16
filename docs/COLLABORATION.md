# PhotoIndex 协作与学习流程

本文保存个人开发与助手协作约定；README 面向仓库读者介绍项目、使用和测试方式。助手执行规则同时见根目录 `AGENTS.md`。

## 导师协作

- 默认流程：需求 → 必要概念 → 接口与验收标准 → 用户实现 → Code Review → 测试 → 验收。
- 核心 Python 业务、算法、SQL 和 AI 流水线由用户优先实现；用户明确要求完整代码或充分尝试后，再提供参考实现。
- 助手可以维护配置、文档与已掌握的重复逻辑，不提前创建无实际用途的模块。
- 先确定高层路线与默认实现选择，再给出一批可连续完成的任务；整批完成后统一审查，无需每一步等待批准。
- 只有实际运行过的检查才能报告为已验证；零测试不能视为里程碑通过。接触过概念不代表已经掌握。

## 开始与交付

开始前阅读 `README.md`、`docs/PROJECT_CHARTER.md`、`docs/DECISIONS.md`、`docs/PROGRESS.md` 和 `docs/LEARNING_LOG.md`，按实际代码与测试核实状态。

当前按 `docs/tasks/M0_02_RECURSION_AND_TESTS.md` 补齐边界测试并完成 M0 验收。交付实现、测试、实际运行结果与必要解释。实质变化后更新进度、决策及学习证据；没有新决策时明确记录架构方向不变。

## 环境与依赖

新终端先在仓库根目录激活 `.venv`。当前直接从仓库根目录运行，无需 `pip install -e .`。虚拟环境不纳入 Git。

引入第三方库前说明实际用途，记录直接依赖并选择可复现的版本锁定方式；不要把本机所有已安装包复制成项目依赖。后端、图像与 AI 框架按实际任务评估引入。

私人样本放在被忽略的 `sample/` 或 `photos/`，索引与模型等本地数据放在 `local-data/`。自动化测试使用临时目录，默认不修改、移动或删除原始照片，数据外发需明确授权。

## Git 工作方式

当前在 `main` 上按可解释的小任务提交；需要隔离新功能时再开分支。不必每写一行就提交，也不必等整个里程碑完成。

```sh
git status
git diff
# 明确选择本次文件，下面仅为示例
git add photoindex/scanner.py tests/test_scanner.py
git diff --cached
git commit -m "feat: add stack-based image scanning and tests"
git push
```

提交前检查暂存区，确认没有原片、密钥、虚拟环境或本地数据库。`git diff` 不显示未跟踪文件的内容，新文件也需要单独阅读。进度文档记录任务与验收证据，Git 记录文件变化。
