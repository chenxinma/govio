# J1 onAbort child.kill() 竞态加固 — 进度

## 目标

两处子进程胶水的取消收尾存在同一窄场景竞态：`onAbort` 内 `child.kill()` 在 spawn 失败（govio-cli 缺失等导致 child 处于未启动/无效状态）时抛 `EINVAL`，「入口即 abort + spawn 失败」时收尾异常。V2 已补 `signal.aborted` 预检，但 kill 调用本身未加固。目标：任何收尾路径不抛、不产生未捕获异常/未处理 rejection，两处胶水行为一致。

## 竞态点定位

两处胶水结构相同（`runGovioCli` 内 `child.kill()` 共 2 个调用点 × 2 个文件）：

| 文件 | 位置 | 触发路径 | 风险 |
|---|---|---|---|
| `packaging/pi/extensions/govio-tools.ts` | `setTimeout` 回调（超时收尾） | 超时触发时 child 处于无效状态 | timer 回调内同步抛 → 未捕获异常 |
| `packaging/pi/extensions/govio-tools.ts` | `onAbort`（取消收尾） | ①入口即 abort（`signal.aborted` 预检同步调用）+ spawn 失败；②运行中 abort + 进程已退出/重复 kill | 同步调用时 Promise executor 抛出 → promise 以原始 EINVAL 拒绝（应走 `error`/`close` 统一收尾）；事件监听器内抛出 → 未捕获异常 |
| `packaging/deepseek/src/index.ts` | `setTimeout` 回调（超时收尾） | 同上 | 同上 |
| `packaging/deepseek/src/index.ts` | `onAbort`（取消收尾） | 同上 | 同上 |

## 加固方式

两处各新增等价的防御性帮助函数（行为一致、风格随各自文件）：

```ts
/** 防御性 kill：spawn 失败/进程已退出/重复 kill 时 child.kill 可能抛异常（如 EINVAL），收尾路径一律不抛 */
function killChildQuietly(child: ChildProcess): void {
	try {
		child.kill();
	} catch {
		// 子进程处于未启动/无效状态时 kill 抛错属预期场景，忽略即可，由 close/error 事件收尾
	}
}
```

- `runGovioCli` 内 `setTimeout` 与 `onAbort` 两处 `child.kill()` 全部改经 `killChildQuietly`，覆盖 spawn 失败、进程已退出、重复 kill 三类异常点。
- kill 抛错被吞掉后，收尾仍由 `error`/`close` 事件驱动（原逻辑不变），不会双 resolve/双 reject。
- `import { spawn, type ChildProcess } from "node:child_process"`（Pi 文件双引号+分号风格；DSH 文件单引号无分号风格，各自对齐）。
- 行为差异保持原状（非本任务范围）：Pi 的 spawn 失败转为结果返回 + `toToolError`，DSH 的 spawn 失败 reject 后由 `runOrFail` 转中文错误——两者均「不抛未捕获异常」，取消/超时/正常路径语义均未变。
- 打包脚本（scripts/）与 docs/specs/ 未改（契约未变，仍是「尊重取消信号」）；README 无需补充说明（仅内部健壮性加固，对外行为与文档描述一致），未改。

## lib 同步证据

- `packaging/deepseek` 下 `npm install`（typescript ^5.6.0 等 devDependencies，node_modules 已被 .gitignore 忽略）后执行 `npx tsc -p .`，**零错误**。
- `packaging/deepseek/lib/index.js` 已同步：含 `killChildQuietly` 定义且 `setTimeout`/`onAbort` 两处调用替换（`git diff` 11 insertions, 2 deletions，与 src 改动一一对应）。
- `packaging/deepseek/lib/types/index.d.ts` 重编译后内容不变（`git hash-object` 与 index 哈希一致 `c05801b2...`；status 的 M 标记仅为 tsc 重写文件导致的 mtime 统计缓存差异，无内容差异）。

## 验证结果

| 命令 | 结果 |
|---|---|
| `node --check packaging/pi/extensions/govio-tools.ts` | 通过（Node v24.16.0 原生类型剥离，无输出） |
| `npx tsc -p .`（packaging/deepseek 内） | 通过，零错误 |
| `uv run python -m pytest tests/ -q` | **342 passed, 18 skipped**，与基线一致不降；test_package_pi_extension.py / test_package_harness_plugin.py 无受影响断言，未改动 |

### 手测（立即 abort 验证，两处胶水均实测）

方法：临时目录（系统 Temp，不入库）搭建假 govio-cli（`node` + 无扩展名 `query` 脚本，每 150ms 写 HEARTBEAT 心跳、收到 `-c DONE` 正常退出）；DSH 侧用 Node 直接驱动真实编译产物 `packaging/deepseek/lib/index.js`（mock `ctx.tools.register` 捕获 `govio_cli` 工具后调用 `execute`）；Pi 侧驱动 `govio-tools.ts` 逐字节副本（`@earendil-works/*` 打桩，`GOVIO_CLI` 环境变量指向假 CLI/不存在的 CLI）。断言三点：①promise 按预期拒绝/解析；②耗时远小于超时（≤3s，超时参数设 15s）；③心跳快照停止增长证明子进程被杀；全程监听 `uncaughtException`/`unhandledRejection`。

| 场景 | DSH（lib/index.js） | Pi（govio-tools.ts） |
|---|---|---|
| spawn 失败 + 入口即 abort（原始 EINVAL 竞态） | PASS，8ms，reject「无法启动…ENOENT」，无未捕获异常 | PASS，7ms，reject「无法启动 …ENOENT」，无未捕获异常 |
| 长驻子进程 + 入口即 abort | PASS，14ms，reject「govio-cli 调用已取消」，子进程即杀 | PASS，14ms，reject「（执行已取消）」，子进程即杀 |
| 长驻子进程 + 运行 300ms 后 abort | PASS，332ms，reject「已取消」，心跳停止 | PASS，334ms，reject「（执行已取消）」，心跳停止 |
| 超时 kill 收尾（800ms） | PASS，834ms，reject「执行超时（800 ms）」，心跳停止 | —（Pi 工具超时 120s，共用同一 `killChildQuietly`，DSH 场景已覆盖） |
| 正常完成不受影响 | PASS，82ms，resolve 正常 stdout | PASS，89ms，resolve 正常结果 |

结论：立即 abort 时子进程约 10ms 量级即终止，未跑满超时；原始竞态（spawn 失败 + 入口即 abort）不再抛 EINVAL，按既有错误路径收尾；两处胶水行为一致。

## 遗留问题

1. `docs/task-plan.json` 的 round5 条目（父代理编排产物）在工作区有修改，非本任务所有权文件，未触碰、未暂存。
2. `packaging/deepseek/lib/types/index.d.ts` 的 git status M 标记为 mtime 统计缓存差异（内容哈希与 index 一致），下次 git 索引刷新会自动消除，受「禁止 git 写操作」约束未做 `update-index --refresh`。
