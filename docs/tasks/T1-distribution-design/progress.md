# T1 进度记录 — 分发形态调研与设计

## 目标

为 govio 两种新分发形态（pi.agent 扩展包、DeepSeek Harness 插件）产出调研与设计文档，供 T2/T3 无需再调研即可实现。仅产出文档，不实现代码；只允许改动 `docs/tasks/T1-distribution-design/` 目录。

## 改动文件清单

| 文件 | 说明 |
|---|---|
| `docs/tasks/T1-distribution-design/research.md` | 调研笔记：pi 扩展机制、DSH 插件机制、govio 现状、来源与证据等级、矛盾与缺失证据 |
| `docs/tasks/T1-distribution-design/design.md` | 设计文档：共享注册契约、两种形态的目录结构/文件清单/骨架代码、打包脚本接口、安装升级卸载流程、共存矩阵、T2/T3 实现清单 |
| `docs/tasks/T1-distribution-design/progress.md` | 本文件 |
| `docs/tasks/T1-distribution-design/status.json` | 任务状态 |

另：runtime 要求的产出副本写入
`C:\Users\Administrator\.pi\agent\sessions\--D--work-gov-io-govio--\subagent-artifacts\outputs\5356ee47-f7ba-49a7-8913-fec2126fcbcc\research.md`（内容与 research.md 一致）。

**未改动**：`src/govio/**`、`skills/**`、`package_skills.py`、`pyproject.toml`、docs/specs/** 等仓库其余文件全部只读；未执行任何 git 写操作。

## 验证命令与结果

本任务为纯文档产出，且当前 runtime 无 shell 工具，无法执行 pytest/ruff/git。验证方式为工具回读校对：

| 验证项 | 方式 | 结果 |
|---|---|---|
| 4 个产出文件可写、内容完整 | write 后 read 回读关键段落 | 通过 |
| 引用的 pi 文档结论 | 逐条对照 `docs/extensions.md`/`packages.md`/`skills.md`/`configuration.md`/`examples/extensions/hello.ts` 原文 | 通过（research.md 逐条标注直接证据） |
| 引用的 DSH 结论 | 直接核对 deepseek-ai/deepseek-harness 仓库原文（README、docs/subsystems/skills.md、docs/cookbook/extension-cookbook.md、docs/cookbook/adding-a-tool.md、apps/cli/reference/README.md、packages/bundle/sdk-minimal/cordis.patch.yml） | 通过；`source_check` 自动语义评估不可用（unclear/0.30），已按预案降级为直接核对原文并在 research.md 披露 |
| govio 现状结论 | 对照 `skills/*/SKILL.md`、`package_skills.py`、`docs/install.md`、`docs/specs/README.md`、`pyproject.toml` | 通过 |

建议 T2/T3 实现后执行的验证（本任务不执行）：

```bash
uv run pytest tests/            # 新增打包脚本测试后
uv run ruff check src/ tests/
uv run package_pi.py            # T2
uv run package_dsh.py           # T3
pi install ./dist/govio-pi-<ver> && pi list        # T2 实测
dsh plugin --profile web add link:packaging/dsh    # T3 实测
```

## 遗留问题（详见 research.md「Missing evidence」与 design.md §8）

1. DSH 为 developer preview、API 有破坏性变更；第三方 npm 发布契约（peerDependencies/files）待 T3 用 `dsh plugin add link:` 实测。
2. pi git 源对仓库子目录包的支持、pi 包更新后包内 assets 的擦除行为，待 T2 实测（design.md 已给 R1/R2 双方案与兜底）。
3. govio `pyproject.toml` 中 `mcp` 依赖用途未证实（无 shell 未能 grep）；若确认可增开「形态 C：MCP server」。
4. `allowed-tools` frontmatter 在 DSH skill 加载中的行为未知（预计仅忽略/告警）。
5. `npx @deepseek-ai/dsh` 的 Node 版本要求（第三方称 22.19+）未获官方原文确认。
