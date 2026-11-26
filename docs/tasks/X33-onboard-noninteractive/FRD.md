# X33 功能需求文档：govio-cli onboard 非交互图后端配置

> 本文件是 govio 仓库（`chenxinma/govio`）改动任务的**功能需求文档（FRD）**，自包含、可直接作为 govio 侧任务的输入。
> 归属：govio-map `docs/task-plan.json` → crossCutting X33（前置，status=pending）。
> 动机背景见 X34 design.md §8。

## 1. 背景与目标

govio-map 要把「初始配置」GUI 化：前端表单背后调用 `govio-cli onboard` 完成图数据库初始化与数据源配置。

现状（govio 0.7.2）：
- **数据源**已支持非交互（`--add-datasource` / `--remove-datasource` / `--url` / `--password` / `--connect-args` / `--overwrite`），密码经 `encrypt_value` 加密落盘、URL 经 `parse_password_from_url` 脱敏。
- **图后端配置仍是 questionary 交互式**（`src/govio/cli/onboard.py` 的 `prompt_graph_config()`），没有非交互旗标。

目标：给 `govio-cli onboard` 增加**非交互图后端配置**，使 GUI 能以子进程方式驱动「指定图数据库」这一步，且**不破坏**现有交互向导。

## 2. 涉及代码位置

| 文件 | 相关函数/位置 | 说明 |
|---|---|---|
| `src/govio/cli/main.py` | `p_onboard` argparse 定义（约 26–60 行）+ `args.action == "onboard"` 路由（约 95–115 行） | 新增旗标与路由 |
| `src/govio/cli/onboard.py` | `prompt_graph_config()`（约 389–462 行）、`onboard()`（约 464 行起）、`onboard_datasource_cli()`（约 245 行起）、`add_datasource()`（约 168 行起） | 抽公共逻辑、补非交互路径 |
| `src/govio/cli/config.py` | `ConfigManager.load()/save()/exists()` | 配置读写，非交互路径复用 |

## 3. CLI 契约（新增旗标）

### 3.1 旗标定义

```
govio-cli onboard
  --backend {ladybug|falkordb|networkx}   非交互式配置图后端（给出即走非交互路径）
  --db-path PATH        ladybug 数据库文件路径（默认 ~/.govio/ontology.lbdb）
  --host HOST           falkordb 主机（默认 localhost）
  --port PORT           falkordb 端口（默认 6379，须为数字）
  --graph NAME          falkordb 图名（默认 ontology）
  --gml-path PATH       networkx GML 文件路径（networkx 必填，须已存在）
  （既有旗标保持不变：--add-datasource / --remove-datasource / --url / --password / --connect-args / --overwrite）
```

### 3.2 路由逻辑（非交互 vs 交互）

在 `args.action == "onboard"` 分支内，按以下优先级：

1. **`--backend` 给出** → 走非交互图后端配置（新函数，见 §4）。
   - 若同时给出 `--add-datasource` / `--remove-datasource` → 图后端配置与数据源增删**在同一进程内先后完成、单次 `save` 落盘**（见 §6）。
   - 若给出与 `--backend` 不匹配的后端参数（如 `--backend ladybug --host x`）→ 报错退出（见 §5）。
2. **`--backend` 未给出**：
   - 若给出 `--add-datasource` / `--remove-datasource` → 维持现状，走 `onboard_datasource_cli()`。
   - 若给出 `--url/--password/--connect-args/--overwrite` 但无 `--add-datasource` → 维持现状报错。
   - 否则 → 维持现状，走 `onboard()` 交互向导。

**关键约束：`--backend` 未给出时，行为与现有版本逐字一致（向后兼容，零回归）。**

### 3.3 各后端的 graph 段输出（与 `prompt_graph_config` 现有输出严格一致）

| 后端 | graph 段 | 参数来源 |
|---|---|---|
| ladybug | `{"backend": "ladybug", "ladybug": {"db_path": <db_path>}}` | `--db-path`，缺省 `str(Path.home() / ".govio" / "ontology.lbdb")` |
| falkordb | `{"backend": "falkordb", "falkordb": {"host": <host>, "port": <port:int>, "graph": <graph>}}` | `--host`/`--port`/`--graph`，缺省 localhost/6379/ontology |
| networkx | `{"backend": "networkx", "networkx": {"gml_path": <gml_path>}}` | `--gml-path` 必填，须 `Path(...).exists()` 为真 |

## 4. 非交互图后端配置（新函数行为）

新增 `onboard_graph_cli(...)`（命名可自定，建议与 `onboard_datasource_cli` 并列），语义：

1. 实例化 `ConfigManager`；若 `exists()`，`load()` 出既有配置；否则空 dict。
2. 按 §3.3 生成 `graph` 段，**覆盖**配置的 `graph` 键（upsert）。
3. **保留**既有 `datasources` 键不变（不读、不写、不覆盖）。
4. `save()` 落盘；打印与交互式一致的人话确认（含 `config_manager.config_path`）。
5. **非交互模式下不弹任何确认**（questionary 的 overwrite/skip 确认一律跳过——非交互就是要可脚本化、可幂等）。

## 5. 错误处理（均 `print(错误: ...)` 到 stderr + `sys.exit(1)`，风格同 `onboard_datasource_cli._fail`）

| 场景 | 行为 |
|---|---|
| `--backend networkx` 但缺 `--gml-path` | 报错：`--backend networkx 需同时提供 --gml-path` |
| `--gml-path` 指向不存在的文件 | 报错（对齐交互式 validate 的「GML 文件不存在」） |
| `--port` 非数字 | 报错：端口必须是数字 |
| `--db-path` 但 `--backend` 不是 ladybug；`--host/--port/--graph` 但非 falkordb；`--gml-path` 但非 networkx | 报错：`--xxx 需配合 --backend <对应后端>` |
| `--backend` 与 `--remove-datasource` 同给 | 允许（先配置图后端，再删数据源，见 §6），或按实现简单性选择先报错——**由实现方裁决并写入 CHANGES.md** |

## 6. 与数据源增删的组合

`--backend ... --add-datasource <name> --url ... [--password ...] [--connect-args ...] [--overwrite]`：

1. 先按 §4 生成 graph 段（若同给 `--remove-datasource`，则删数据源）。
2. 再复用 `add_datasource()` 的既有逻辑（含 `_encrypt_url_password` 加密、`parse_cli_connect_args` 解析）追加/覆盖数据源。
3. **单次 `save()` 落盘**，避免两次写文件互相覆盖。
4. 落盘后打印汇总（图后端 + 数据源结果 + `config_path`）。

`--remove-datasource` 与 `--backend` 组合时的数据源删除复用 `remove_datasource()` 既有逻辑。

## 7. 测试要求（govio 仓库 pytest）

新增用例覆盖（建议 `tests/` 下新增 onboard 相关测试，用 `tmp_path` 注入 `ConfigManager(config_path=...)`）：

1. `--backend ladybug`（缺省 db_path）→ 生成 `graph.backend=ladybug` 且 db_path 为默认值；不弹交互。
2. `--backend ladybug --db-path <tmp>` → db_path 自定义生效。
3. 幂等：先 `--backend ladybug`，再 `--backend falkordb --host h --port p --graph g` → graph 段被覆盖、datasources 保留不变。
4. 组合：`--backend ladybug --add-datasource mydb --url duckdb:///x.db` → graph 与 datasource 同落盘；重复执行带 `--overwrite` 不报「已存在」。
5. 错误路径：`--backend networkx` 缺 `--gml-path`、`--port abc`、`--backend ladybug --host x` → 均 exit 1 且 stderr 有 `错误:`。
6. 向后兼容：不带 `--backend` 时 `onboard()` 交互路径不被触碰（可用 monkeypatch `questionary` 断言未被调用，或仅靠回归现有 onboard 相关测试）。
7. `--help` 输出包含新旗标。

门禁：`ruff check` + `ruff format --check` + `pytest` 全绿（govio 仓库 CI 既有门禁）。

## 8. 验收清单（govio 侧完成定义）

- [ ] `govio-cli onboard --backend ladybug` 非交互生成 `~/.govio/config.yaml`，`graph.backend=ladybug`、db_path 默认，不弹 questionary
- [ ] `--backend ladybug --db-path <p>` 生效；重复执行幂等且已有 datasources 不丢
- [ ] `--backend ladybug --add-datasource mydb --url ... --password ...` 一次完成图+数据源，密码加密落盘、URL 脱敏
- [ ] 不带 `--backend` 时仍走交互向导（零回归）
- [ ] `--help` 列出新旗标
- [ ] ruff + pytest 门禁绿
- [ ] 升版发布新 govio 版本号（供 X34 提升 `MIN_GOVIO_CLI_VERSION`）

## 9. 明确不做（范围纪律）

- 不改数据源加密逻辑（`encrypt_value` / `parse_password_from_url`）、不改 `ConfigManager` 的文件格式/备份语义。
- 不做图后端「测试连接」（连通性探测归 govio-map X34 的「测试连接」步骤，可复用 `observe` 既有能力；本 FRD 只负责「写配置」）。
- 不新增后端类型；只覆盖现有 ladybug / falkordb / networkx 三种。
