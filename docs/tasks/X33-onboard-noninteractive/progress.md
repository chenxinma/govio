# X33 govio-cli onboard 非交互图后端配置 — progress.md

## 目标

给 `govio-cli onboard` 增加非交互图后端配置（`--backend {ladybug|falkordb|networkx}` + 各后端参数），
使 govio-map 的 onboard GUI（X34）能以子进程方式驱动「指定图数据库」这一步；
不破坏既有交互向导（不带 `--backend` 时行为逐字一致）。
需求规格：`docs/tasks/X33-onboard-noninteractive/FRD.md`（源自 govio-map crossCutting X33）。

## 改动清单

| 文件 | 改动 |
|---|---|
| `src/govio/cli/main.py` | `p_onboard` 新增 6 个旗标：`--backend`（choices 三后端）/`--db-path`/`--host`/`--port`/`--graph`/`--gml-path`（均在 help 中标注所属后端）；路由改为：`--backend` 或任一后端专属旗标给出 → `onboard_graph_cli`（可与数据源增删组合），否则维持原三分支（datasource 旗标 / 参数报错 / 交互向导）逐字不变 |
| `src/govio/cli/onboard.py` | 新增 `build_graph_config()`（FRD §3.3 三后端 graph 段生成 + §5 全部校验：旗标与后端不匹配「--xxx 需配合 --backend yyy」、networkx 缺 `--gml-path`、GML 不存在、端口非数字）与 `onboard_graph_cli()`（graph 段 upsert、datasources 保留、与增删数据源组合时**单次 save** 落盘、全程无 questionary）；抽出 `_apply_add_datasource`/`_apply_remove_datasource`（配置字典上操作、不落盘），`add_datasource`/`remove_datasource` 改为薄封装（公开签名与行为不变） |
| `tests/test_onboard_noninteractive.py` | 新建 26 个用例，覆盖 FRD §7 全部 7 类场景（见下） |
| `docs/specs/cli.md` | onboard 路由优先级、非交互图后端旗标表、组合语义（单次 save / upsert）、错误契约、命令示例、Key Functions 同步 |
| `pyproject.toml` + `uv.lock` | 版本 0.7.2 → 0.8.0（新增 CLI 能力走 minor）；lock 仅根包版本行 1 行 diff |
| `docs/specs/README.md` / `docs/install.md` / `packaging/{pi,deepseek}/package.json` / `packaging/{pi,deepseek}/README.md` | 版本引用同步 0.8.0（K2 的版本统一清单） |
| `docs/roadmap.md` | CLI 工具链 onboard 行补「非交互式图后端配置 `--backend`」 |

## 实现方裁决（FRD §5 遗留项）

`--backend` 与 `--remove-datasource` 同给：**允许**——先配置图后端、再删数据源、单次 `save()`。
（FRD 写「写入 CHANGES.md」，但仓库无 CHANGES.md；按仓库既有约定记入本文件与 `docs/specs/cli.md`
的 Non-interactive graph backend flags 小节。）

其余契约决策：
- 后端专属旗标与 `--backend` 不匹配（或在无 `--backend` 时出现）→ 报错「--xxx 需配合 --backend yyy」，exit 1；
- `--port` 保持字符串接收、自行校验数字（保证 `错误:` 文案 + exit 1，而非 argparse 的 exit 2）；
- 无 `--backend` 时数据源旗标路径（`onboard_datasource_cli`）与报错文案逐字保留，零回归。

## 验证命令与结果

```bash
uv run pytest tests/ -q                 # 552 passed, 18 skipped（基线 508+26 新增，全绿）
uv run ruff check .                     # All checks passed!
uv run ruff format --check .            # 105 files already formatted
env -u UV_DEFAULT_INDEX uv sync --group dev --locked   # CI 同款门禁通过（govio==0.8.0 重装）
```

新增用例（`tests/test_onboard_noninteractive.py`，26 个）：
1. 缺省值：ladybug 默认 `~/.govio/ontology.lbdb`、falkordb 默认 localhost/6379/ontology、networkx gml_path；
2. 自定义 `--db-path` 生效；
3. 幂等：切换后端 graph 段被覆盖、datasources 原样保留；重复执行结果不变；
4. 组合：图+数据源单次落盘、密码加密 + URL 脱敏、重复执行需 `--overwrite`；
5. 错误路径 8 个：缺 `--gml-path` / GML 不存在 / `--port abc` / 旗标后端不匹配 / `--add` 缺 `--url` / add+remove 冲突 / remove 带 url / 报错不落盘；
6. 向后兼容：不带 `--backend` 走 `onboard()` 交互、数据源旗标走 `onboard_datasource_cli`、`--url` 无 add 仍报原错；
7. `--help` 列出全部新旗标。

全部非交互用例用炸雷桩断言 **questionary 零调用**。

## 遗留问题

- 合并后待发版：tag + GitHub Release 四资产 + PyPI（PyPI 由 owner 手工，K7 分工；SOP 见 `docs/release.md`）。
  `docs/install.md` 的 v0.8.0 下载链接在发版前 404（K2 同款预期，发版时补齐资产）。
- `docs/release.md` 未改：其中版本号是发版命令模板与 v0.7.2 实测历史记录，发版时按当版替换。
- govio-map 侧 `MIN_GOVIO_CLI_VERSION` 提升归 X34（其 acceptance 已写明）。
- 本机注意：环境变量 `UV_DEFAULT_INDEX` 指向阿里云镜像会让裸跑 `uv lock` 把 lock registry
  整体改写成镜像（本次实测 1066 行 diff，已按 AGENTS.md 纪律回滚）；提交前务必 `git diff uv.lock`。
