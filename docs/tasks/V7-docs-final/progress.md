# V7 进度记录 — T7 文档整合 + 修复 + 全量验证

## 目标

第 2 轮评审文档整合与收尾核验（任务所有权：`docs/install.md`、`docs/roadmap.md`、`docs/specs/README.md`、`docs/tasks/V7-docs-final/`，task=V7）：

1. **P2-b**：`docs/specs/README.md` Packaging 行相对链接 `../../install.md` 失效（解析到仓库根，该处无 `install.md`），改为 `../install.md`。
2. **P2**：`docs/install.md` 示例路径语义两处：
   - cp 步骤在项目级布局下源与目标同目录会报 `cannot copy a directory into itself`，加注「assets 已在 `<skills-root>/govio/assets` 时跳过」并把示例源改为 `<assets绝对路径>`；
   - `meta graph --output ./data/meta` 与前文 `./output` 不一致且目录须先存在（`cmd_graph` 对不存在目录报 `❌ 输出目录不存在` 并退出）——统一为 `./output` 并注明前置条件。
3. **收尾核验**：检查 V2~V6 修复是否需要回写 install.md / roadmap.md / specs（trino+http 支持、wslc 真实库测试说明、新增测试文件等），保持文档与最终实现一致（AGENTS.md 文档同步规则，不得留过时描述）。
4. **全量验证**：`uv run ruff check src/ tests/` 与 `uv run pytest tests/` 全绿（integration 默认 skip，其余全绿，含本轮迁入/新增测试）。

约束遵守：无任何 git 写操作（commit/push/checkout/merge 均未执行）；Python 3.13+ 类型语法；文件读写均 `encoding="utf-8"`（本次均为 Markdown/JSON 文档）；注释与文档用中文。

## 改动文件清单

### 任务所有权内

| 文件 | 改动 |
|---|---|
| `docs/specs/README.md` | **P2-b**：Packaging 行链接 `../../install.md` → `../install.md` |
| `docs/install.md` | **P2-a**：「形态一 §2 复制 Assets」加注「assets 已在 `<skills目录>/govio/assets` 时跳过（同目录 cp 报 cannot copy a directory into itself）」，cp 示例源改为 `<assets绝对路径>`（以命令打印的绝对路径为准）；**P2-b**：「形态二」「形态三」两处 `meta graph --output ./data/meta` 统一为 `--output ./output` 并注明前置条件（须为已写出 meta CSV 的已存在目录，`cmd_graph` 对不存在目录报错退出不自动创建）；**收尾核验**：「配置数据源」URL 表下补 URL scheme 归一说明（裸 `mysql://`/`postgres://`/`postgresql://`、`trino+http(s)://` 均可用，对应 V6 的 scheme 归一修复） |
| `docs/roadmap.md` | **收尾核验**：Phase 3 待办「`test_package_harness_plugin.py` 从任务目录迁入 `tests/` 常规集」勾选为已完成（V3 已迁入）并补记打包脚本测试与真实库集成测试（`tests/test_real_mysql.py` / `test_real_postgres.py` / `test_real_observe.py`，integration 标记默认 skip、`GOVIO_REAL_DB=1` 才执行、wslc 容器验证后清理）；修正「CLI 工程化待办」中 assets 默认目录过时描述 `.s/skills/govio/assets` → `.agents/skills/govio/assets`（与 `src/govio/cli/meta.py:43` `DEFAULT_ASSETS_DIR` 一致） |
| `docs/tasks/V7-docs-final/` | 本文件与 `status.json` |

### 越界 bug 级小修（各 1 行，AGENTS.md 文档同步规则要求，见下「越界小修说明」）

| 文件 | 改动 |
|---|---|
| `docs/specs/metadata.md` | MysqlLoader 类型解析句补「数值类型的 `unsigned` / `zerofill` 后缀在解析前剥离（如 `decimal(10,2) unsigned` → (0,10,2)），`dtype` 保留原始 `COLUMN_TYPE`」——V4 修复 `_parse_type` 后该行是文档与实现唯一不一致点（V4 progress 遗留 #2 明确点名此句待同步） |
| `docs/tasks/T1-distribution-design/design.md` | §4.4 排除规则补 `package-lock.json`——V3 修复 `scripts/package_harness_plugin.py` 排除集后该清单过时（V3 progress 遗留 #1 明确点名待同步） |

## 越界小修说明

两处均属「bug 级小修」且为 AGENTS.md 文档同步规则（功能实现改动须同步对应 spec）直接要求的收尾：

1. `docs/specs/metadata.md` 一行：V4 改了 `mysql_loader._parse_type`（剥离 unsigned/zerofill），AGENTS.md 对应表指明元数据加载器变更须同步 `docs/specs/metadata.md`；不同步则该 spec 行过时（未提及后缀剥离语义）。仅补一句，无其他改动。
2. `docs/tasks/T1-distribution-design/design.md` 一行：V3 改了打包脚本排除规则，design.md §4.4 是排除规则的定稿清单，遗留 `package-lock.json` 即过时描述。仅补三个字级别的括注。

其余 V2~V6 涉及的 spec（`docs/specs/config.md`、`docs/specs/observe-data.md`、`docs/specs/cli.md`）经核对**已由 V6 同步到位**（trino+http 归一表、大小写不敏感说明、meta CLI `--db` scheme 表），无需再改；`packaging/pi/README.md`、`packaging/deepseek/README.md` 已由 V2/V3 同步，不在改动范围。

## 收尾核验结论（V2~V6 → 文档回写）

| 来源 | 修复内容 | 需回写？ | 处置 |
|---|---|---|---|
| V2 | pi 扩展 aborted 预检 + `tests/test_package_pi_extension.py` | 行为恢复既已文档化的「尊重 pi 取消信号」契约，无漂移；新测试文件回写 roadmap | roadmap 补记 ✓ |
| V3 | `test_package_harness_plugin.py` 迁入 `tests/` + 排除 `package-lock.json` | 迁入事实回写 roadmap（Phase 3 待办已过时）；排除规则回写 design.md §4.4 | roadmap 勾选 ✓、design.md 补注 ✓ |
| V4 | mysql `_parse_type` 剥离 unsigned/zerofill + 真实库测试 | metadata.md 类型解析句补后缀语义；wslc 真实库测试说明回写 roadmap | metadata.md ✓、roadmap ✓ |
| V5 | postgres `_parse_type` pd.notna + 真实库测试 | metadata.md PostgresLoader 段契约本就与修复后实现一致（修复即向文档对齐），无需改；真实库测试说明并入 roadmap 补记 | roadmap ✓ |
| V6 | trino+http 归一 + drivername 大小写不敏感 + 真实库测试 | config.md / observe-data.md / cli.md 已由 V6 同步；install.md URL 表缺归一说明 | install.md 补注 ✓ |

## 验证命令与结果

| 命令 | 结果 |
|---|---|
| `uv run ruff check src/ tests/` | 通过：**All checks passed!** |
| `uv run pytest tests/` | 本机 shell 报 `Failed to canonicalize script path`（uv 0.12.1 入口脚本解析环境问题，V2~V6 progress 均记录同一现象，非本次改动引入） |
| `uv run python -m pytest tests/`（等价命令，同一虚拟环境） | 通过：**303 passed, 18 skipped, 3 warnings**（18 skipped 全部为 integration 真实库用例默认 skip：test_real_mysql 6 + test_real_postgres 6 + test_real_observe 6；3 warnings 为 `PytestUnknownMarkWarning: integration` 未在 pyproject 注册，属既有非阻断遗留） |
| 链接修复核验：`test -f install.md`（仓库根）/ `test -f docs/install.md` | 仓库根 `install.md` 不存在（证明旧 `../../install.md` 确实失效）、`docs/install.md` 存在（新 `../install.md` 从 `docs/specs/README.md` 解析正确） |
| `meta graph` 目录语义核验：`grep` `src/govio/cli/meta.py` `cmd_graph` | `meta.py:916-918`：`if not output.exists(): print(f"❌ 输出目录不存在: {output}"); sys.exit(1)`——证实「目录须先存在、不会自动创建」的前置条件注记与实现一致 |
| `git diff --cached --name-only` | 空（无 staged 文件；全程未执行任何 git 写操作） |

## 遗留问题

1. **`pytest.mark.integration` 未在 `pyproject.toml` 注册**（V4/V5/V6 同款遗留）：`uv run pytest tests/` 有 3 条 `PytestUnknownMarkWarning`（非阻断，测试全绿）。`pyproject.toml` 不在本任务所有权内，未动。建议后续在 `[tool.pytest.ini_options] markers` 注册 `integration: 真实库集成测试`。
2. **`uv run pytest tests/` 直接调用报 `Failed to canonicalize script path`**（本机 uv 环境问题，V2~V6 均记录），全量验证用等价的 `uv run python -m pytest tests/` 通过；`uv run ruff check src/ tests/` 直接调用正常。
3. **pi git 子目录源、DSH `link:` 安装回路、R2 包内 assets 更新存续**仍为 T1 标注的待实测缺口，install.md 已如实保留「待实测」标注，非本轮问题。
