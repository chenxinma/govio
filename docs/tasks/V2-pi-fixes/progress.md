# V2 进度记录 — pi 扩展 P2-d 修复 + 打包脚本测试补齐

## 目标

第 2 轮评审两项修复（任务所有权：`packaging/pi/`、`scripts/package_pi_extension.py`、`tests/test_package_pi_extension.py`、`docs/tasks/V2-pi-fixes/`，task=V2）：

1. **P2-d**：`packaging/pi/extensions/govio-tools.ts` 取消信号缺 `aborted` 预检——`signal.addEventListener("abort", ...)` 对「监听注册前已 abort」的信号永不触发，子进程会跑满超时而非立即终止。照搬 `packaging/deepseek/src/index.ts` 的 `if (signal.aborted) onAbort() else addEventListener` 范式修复，保持「尊重 pi 取消信号」契约。
2. **P2**：`scripts/package_pi_extension.py` 无自动化测试——新建 `tests/test_package_pi_extension.py`（tmp_path/monkeypatch，不依赖网络），覆盖版本注入、排除规则、重复执行幂等、产物自检失败路径。

约束遵守：未执行任何 git 写操作；Python 3.13+ 类型语法；文件读写均 `encoding="utf-8"`；注释/文档中文；其余文件只读。

## 改动文件清单

| 文件 | 说明 |
|---|---|
| `packaging/pi/extensions/govio-tools.ts` | P2-d 修复：`runGovioCli` 内补 `aborted` 预检（`if (signal.aborted) onAbort() else signal.addEventListener("abort", onAbort, { once: true })`），预 abort 信号立即 kill 子进程而非等待超时 |
| `tests/test_package_pi_extension.py` | 新建 8 个测试：版本注入（pyproject 默认 / --version 覆盖 / 非法版本拒绝 / 缺 project.version 拒绝）、排除规则（govio/assets、`__pycache__`、`.pyc`、`.DS_Store` 不入产物；普通文件与 govio-observe/assets 保留）、重复执行幂等（逐文件 SHA256 比对 + 陈旧文件清理）、产物自检失败路径（篡改产物逐类报错 + frontmatter 不一致端到端返回 1） |
| `docs/tasks/V2-pi-fixes/progress.md` | 本文件 |
| `docs/tasks/V2-pi-fixes/status.json` | 任务状态 |

**未改动**：`scripts/package_pi_extension.py`（仅补测试，行为零变更）、`skills/**`、`src/govio/**`、`docs/specs/**`、`install.md` 等均只读。文档同步规则核对：本修复恢复的是既已文档化的契约（T2 progress/install-draft 中「超时/取消收尾」），打包脚本行为未变，不产生 spec 漂移，无需更新 docs/specs（Packaging 条目指向 install.md，不在本任务所有权内且无漂移）。

## 与评审要求的对齐

1. **P2-d 范式**：与 `packaging/deepseek/src/index.ts` §runGovioCli 逐行同构（`signal.aborted` → 立即 `onAbort()`，否则注册一次性 abort 监听）；`close`/`error` 收尾里的 `removeEventListener` 对未注册情况天然无害，行为契约（超时/取消/正常三种收尾）不变。
2. **P2 测试**：通过 importlib 按文件路径加载 `scripts/package_pi_extension.py`，monkeypatch 其 `ROOT`/`PACKAGING_DIR`/`SKILLS_DIR`/`SKILLS_TARGET` 到 tmp_path 下的最小假仓库，`main()` 经 monkeypatch 过的 `sys.argv` 调用；断言侧硬编码 6 个 skill 名（独立于脚本常量，防止实现改坏时测试同盲）。

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run pytest tests/test_package_pi_extension.py -q` | 通过（8 passed in 0.61s） |
| `uv run ruff check tests/test_package_pi_extension.py scripts/package_pi_extension.py` | 通过（All checks passed!） |
| `node --check packaging/pi/extensions/govio-tools.ts` | 通过（TS 语法检查 OK） |
| `uv run scripts/package_pi_extension.py`（连续两次） | 均通过：`✓ dist/govio-pi-0.6.0（24 个文件，其中 skills 副本 21 个文件）` + `✓ 产物结构自检通过（6 个 skill、无 assets/缓存文件、版本 0.6.0 已注入）` |
| 产物幂等复核（两次运行间逐文件 SHA256 比对） | `idempotent: True | files: 24` |
| `git diff --cached --name-only` | 空（无 staged 文件；未执行任何 git 写操作） |

## 遗留问题

1. **`onAbort` 内 `child.kill()` 在 spawn 失败时抛 `EINVAL`（既有问题，非本修复引入）**：node 实测（Node 24/Windows）——若 `govio-cli` 不存在（spawn ENOENT）且信号在调用入口已 abort，`onAbort()` 里 `child.kill()` 同步抛 `kill EINVAL`，Promise 以该异常 reject，且后续 ENOENT 的 `error` 事件因处理器未注册而成为 uncaughtException。该隐患在 `packaging/deepseek/src/index.ts` 同款代码及当前代码的「abort 事件竞态失败 spawn」路径中同样存在（EventTarget 监听器内抛错即 uncaught），属共享的既有缺陷而非 P2-d 修复的回归（修复前的净新增暴露仅限「入口即 abort + spawn 失败」这一窄场景，此前该场景走 error 收尾）。建议后续任务统一加固：`onAbort` 内 `child.kill()` 包 try/catch（govio-tools.ts 与 deepseek/index.ts 同步），超出本任务所有权范围，未擅自修改。
2. `uv run pytest tests/test_package_pi_extension.py` 直接调用在本机提示 `Failed to canonicalize script path`（uv 入口脚本解析的环境问题），用等价的 `uv run python -m pytest tests/test_package_pi_extension.py -q` 全绿；pytest 自身运行不受影响。
