# T7 架构审视与功能精简建议

> T7 产出。在 govio「无 LLM、可插拔任意 Agent Runtime 的 skill + CLI」定位下，
> 审视模块边界与三种分发形态的统一抽象，指出冗余、可删减功能与应停止的路线，
> 给出未来 2-3 个迭代的精简方向。**本文只写设计建议，不做大重构**；
> 本轮（T1-T6 + T7）的实际改动以各任务 progress.md 为准。

## 1. 定位与审视框架

govio 的定位（由 T1-T6 本轮定型）：

- **核心是无 LLM 的数据治理工具**：`govio-cli`（命令行）+ `skills/`（工作流指令），
  LLM 完全来自宿主 Agent Runtime（Codex、Claude Code、pi、DeepSeek harness 等）。
- **可插拔任意 Agent Runtime**：runtime 适配层是 `govio-cli` 之上的**薄壳**，
  只做「工具注册 + 子进程转发」，不绑定任何 harness 私有行为（T1 design §0）。
- **安全边界分离**：skill 决定操作意图，CLI 决定数据可见性（docs/design-thoughts.md）。

据此，审视的三条主线：

1. Python 核心（`src/govio/**`）的模块边界是否与定位一致；
2. 三种分发形态是否有统一抽象，形态差异是否收敛在单点；
3. 哪些功能/依赖/路线与定位不符或无消费方，应删减或停止。

## 2. 模块边界审视（metadata / graph / observe_data / core / cli）

| 模块 | 现状职责 | 边界评价 | 建议（未来迭代，不本轮做） |
|---|---|---|---|
| `metadata/` | 元数据抽取（TDS/DuckDB/MySQL/PostgreSQL/Trino 加载器）、标准/关系/指标加载、推荐器、node_id、CSV 导出 | 职责总体清晰，但混入两个越界点：`gen_networkx.py`（CSV→**GML 图格式**转换，产物是 graph/NetworkXGraph 的存储格式）与 `utility.py`（`make_csv` 导出编排 + `data_standard_recommend` 推荐入口两类职责） | `gen_networkx.py` 归属迁到 `graph/`（或 core）；`utility.py` 按「导出 / 推荐」拆分。均为移动级小改，需跟测试 |
| `graph/` | 三后端（NetworkX/FalkorDB/Ladybug）+ 两个 CSV bulk loader | 边界清晰 | 不动 |
| `observe_data/` | `core/`（连接、存储、比对、探查、图表）+ `tools/`（面向调用方的工具函数层） | **非冗余**（已核实：`cli/observe.py` 的实际调用层就是 `tools/`，`core/` 是其依赖）；但 `tools/` 与 `core/` 的分层意图没有文档化，易被误判为重复入口 | 在 docs/specs/observe-data.md 补一句分层说明（tools=工具入口、core=实现） |
| `core/` | GraphFactory、AssetsGenerator、sql_builder | 三者都是「图库之后」的生成逻辑，边界清晰 | 不动 |
| `cli/` | main.py 分发 + 各命令组 | `meta.py` 1240 行过大（元数据导入 7 个子命令 + URL→Loader 解析 + 导出编排）；`std_recommend.py` 是 `meta recommend` 的真实调用链（cli→utility→recommender），非死代码 | `meta.py` 按子命令域拆文件（T6 的 `_build_metadata_loader` 已是现成切点） |

跨模块重复（本轮任务所有权约束产生，属已知债）：

- **`_parse_type` 类型解析三份**：`trino_loader.py:13`、`mysql_loader.py:43`、`postgres_loader.py:72`
  同语义复制（T4 因所有权约束在模块内自带）。应收敛为一处共享实现
  （建议 `metadata/` 内共享小模块或 `MetadataLoader` 基类方法），三个 loader 引用同一实现。

## 3. 三种分发形态的统一抽象

三种形态（zip+uv tool、pi 扩展、DeepSeek Harness 插件）已经共享同一组契约
（T1 design §1，T2/T3 已按此实现）：

```
                    ┌──────────────────────────────────────────┐
   单一真源          │  skills/（6 个 skill，唯一真源，打包复制）  │
                    │  assets 契约：<skills-root>/govio/assets   │
                    │  （派生数据，meta graph 生成，打包排除）      │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
   适配层边界        │  govio-cli 契约：argv 数组子进程（无 shell） │
                    │  子命令白名单 meta/query/observe/sql/backend│
                    │  UTF-8、超时默认 120s、尊重取消信号          │
                    └───────────────────┬──────────────────────┘
                                        │
        ┌───────────────────┬───────────┴──────────┬───────────────────┐
        ▼                   ▼                      ▼                   ▼
  形态 0：zip+uv tool   形态 1：pi 扩展       形态 2：DSH 插件      （未来形态 N…）
  纯 skills 解压        package.json manifest  cordis.patch.yml
  无工具面              + TS extension         + Cordis plugin
                        govio_cli/govio_query  govio_cli/govio_query
                        + /govio 诊断命令      （同一工具面契约）
```

**评价：抽象是健康的。** 形态差异只在「宿主注册面」（manifest/插件挂载 + 一层 TS 胶水），
runtime API 漂移只影响胶水层（T3 已验证 DSH 破坏性变更只改 `src/index.ts` + `cordis.patch.yml`）。
新增形态时只需实现注册面，Python 核心与 skills 零改动。

统一抽象的**残留不统一点**（可收敛）：

1. **打包排除规则三份**：`package_skills.py`（排除 `skills/govio/assets`）、
   `scripts/package_pi_extension.py`、`scripts/package_harness_plugin.py`
   各自实现「排除 assets/`__pycache__`/`*.pyc`/`.DS_Store`」逻辑，规则漂移风险。
2. **三支打包脚本入口分散**：仓库根 `package_skills.py` + `scripts/` 下两支，
   版本注入逻辑（从 `pyproject.toml` 读版本）在两支脚本中重复。
3. **安装文档三处**：`docs/install.md`、`packaging/pi/README.md`、`packaging/deepseek/README.md`
   （T7 本轮已把 install.md 定为唯一权威，README 改为交叉引用，见 §6 迭代 1）。

## 4. 冗余与可删减项

| 项 | 证据 | 建议 |
|---|---|---|
| **`mcp` 依赖** | `src/`、`tests/` 无任何 `import mcp`（`tests/test_mcp_*.py` 实测的是 observe_data 配置/连接管理，仅历史命名）；对应 design-thoughts.md「方案一：MCP 服务」并未落地为代码 | **删除依赖**；MCP 路线取舍见 §5 |
| `_parse_type` ×3 | §2，三份同语义复制 | 收敛为共享实现 |
| 打包排除规则 ×3、版本注入 ×2 | §3 | 收敛到 `scripts/` 共享工具模块，三支脚本复用 |
| `docs/tasks/T3-harness-plugin/test_package_harness_plugin.py` | 8 个用例在任务目录，未进 `uv run pytest tests/` 常规集（T3 遗留 3） | 迁入 `tests/`（移动级小改） |
| 安装说明多处重复 | §3 | install.md 单一权威 + README 交叉引用（T7 本轮已执行） |
| `cli/meta.py` 1240 行 | §2 | 拆分属结构整理，非删减，放迭代 2 |

**保留、不建议删**（负向清单，防止误删）：

- `observe_data/tools/`：`cli/observe.py` 的真实调用层，不是重复入口；
- `cli/std_recommend.py`：`meta recommend` 的真实调用链；
- `metadata/gen_networkx.py`：NetworkX 后端的 GML 生成必需（只是模块归属错位）；
- 三图后端与 `graph/` 两支 bulk loader：分发形态对后端无偏好，全部保留。

## 5. 应停止 / 暂缓的路线

1. **MCP 托管数据集服务（design-thoughts.md 方案一）——暂缓**。
   当初为「多轮探查的中间数据集受控访问」设计，但 observe_data 本地受控执行
  （ObserveStore + CLI 数据可见性边界）已覆盖单 Agent 场景，且当前无多 Agent
   共享数据集的消费方。维持暂缓，**不作为近期方向**；连带删除 `mcp` 依赖（§4）。
   若未来出现「多个 Agent Runtime 并发共享探查中间结果」的真实需求再重启讨论。
2. **Iceberg 文件存储（方案二）——维持放弃**（访问控制成本高于收益，本轮无变化）。
3. **`meta meta --schema-alias`、`meta meta --dry-run`——维持停止**（roadmap 已评估项，
   node_id 稳定性 / 空结果守卫已覆盖诉求）。
4. **npm 发布通道（`@govio/pi-package`、`@dsh-external/govio-tools`）——暂缓**。
   本地目录/git 源分发已覆盖当前用户；npm 发布涉及外部可见性承诺，待真实外部需求再启。
5. **对 DSH developer preview 的深度适配——停止加码**。保持现有薄壳（两个工具 + skills），
   不为 DSH 做 typed wrapper 扩展或私有行为绑定；DSH 破坏性变更只修胶水层。

## 6. 未来 2-3 个迭代的精简方向

### 迭代 1：低风险清理（删冗余，半天量级，可随时做）

1. 删除 `mcp` 依赖（pyproject.toml，§4）；
2. `_parse_type` 三份收敛为共享实现（§2）；
3. 打包排除规则 + 版本注入收敛到 `scripts/` 共享模块，三支打包脚本复用（§3）；
4. `test_package_harness_plugin.py` 迁入 `tests/`；
5. README/install 重复内容收敛（T7 本轮已完成 install.md 权威化，此条收尾核对交叉引用）。

### 迭代 2：模块边界整理（移动级改动，需跟测试）

1. `cli/meta.py` 按子命令域拆文件（切点：`_build_metadata_loader` / 各 `cmd_*`）；
2. `metadata/utility.py` 拆「导出编排 / 推荐入口」两个职责；
3. `gen_networkx.py` 归属迁移到 `graph/`（或 core），同步 import 与 spec 文档；
4. observe_data 的 tools/core 分层说明补进 docs/specs/observe-data.md。

### 迭代 3：分发统一（决策驱动）

1. 统一打包入口（如 `govio package <form>` 或 scripts 统一 front-end），三种产物一键产出；
2. 壳包版本与 govio-cli 版本对齐自动化（现为打包脚本注入 + 人工约定 x.y 对应 x.y）；
3. MCP 形态 / npm 发布通道的启停决策（§5，有消费方再启）。

## 7. 本轮明确不做的事

- 不做大重构（`meta.py` 拆分、`utility.py` 拆分、`gen_networkx.py` 迁移均留待迭代 2）；
- 不动 Python 核心的对外行为（仅修 `output_result` 的 Decimal 序列化 bug 一处，见 progress.md）；
- 不新增功能、不删任何既有命令（精简只针对依赖、重复实现与路线，不针对用户可见功能）。
