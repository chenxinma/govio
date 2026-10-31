// govio 新工作模式编排：批次1（T1/T4/T5 并行）→ R1 记录 → 批次2（T2/T3/T6 并行）→ R2 记录 → 批次3（T7）→ R3 记录
const REPO = '仓库 D:/work/gov-io/govio，git 分支 experiment/new-work-mode 已检出；所有改动只落在工作树，禁止执行任何 git 写操作（commit/push/checkout/merge 等）';
const RULES = '通用约束：Python 3.13+ 类型语法；文件读写必须 encoding="utf-8"；注释与文档用中文；遵守仓库 AGENTS.md（import 三组分离、用 logging 不用 print、命名规范、不用裸 except）；只允许修改你任务所有权清单内的文件，其余文件只读；不做与任务无关的重构。完成前必须在你的任务目录下写 progress.md（目标/改动文件清单/验证命令与结果/遗留问题）和 status.json（JSON 字段：task、status 取 done|partial|failed、summary 一句话、files 数组）。若被阻塞则标 partial 或 failed 并说明原因，不要硬凑实现。';

const batch1 = await runs.all([
  {
    key: 't1',
    label: '调研两种分发形态并出设计',
    agent: 'researcher',
    task: '【T1 分发形态调研与设计】' + REPO + '。目标：为 govio 的两种新分发形态（pi.agent 扩展、DeepSeek harness 插件）产出调研与设计文档，供 T2/T3 直接照做。' + RULES + ' 任务所有权：仅 docs/tasks/T1-distribution-design/ 目录（status.json 中 task 写 "T1"）。调研内容：(1) pi.agent 扩展机制：阅读 C:/Users/Administrator/.pi/agent/install/releases/1.4.0/node_modules/@earendil-works/pi-coding-agent/docs/extensions.md 以及该包 examples/extensions/ 目录下的示例，弄清 pi 扩展的目录结构、注册方式（custom tools / MCP / skills 安装）、安装与更新方式；(2) DeepSeek harness 插件机制：用 web_search 调研 DeepSeek 开源 agent harness/CLI 的插件或技能扩展形态，用 source_check 保留来源证据；若不存在官方标准，明确列出假设并给出通用适配层方案；(3) 结合 govio 现状（可读 skills/ 目录 6 个 skill、package_skills.py、docs/install.md、pyproject.toml）设计两种分发形态的打包产物结构、安装/升级/卸载流程、skills 与 govio-cli 在目标 runtime 中的注册契约、与现有 dist/govio-skills.zip + uv tool 流程的共存关系。产出两个文件：docs/tasks/T1-distribution-design/research.md（调研笔记，附来源链接与关键结论）和 docs/tasks/T1-distribution-design/design.md（设计文档，具体到目录结构、文件清单、打包脚本接口，让 T2/T3 无需再调研即可实现）。除任务目录外不得修改任何仓库文件。'
  },
  {
    key: 't4',
    label: '实现 MySQL 元数据加载器',
    agent: 'worker',
    task: '【T4 MySQL 元数据加载器】' + REPO + '。目标：新增 MysqlLoader，从 MySQL 抽取表/列元数据。' + RULES + ' 任务所有权：src/govio/metadata/mysql_loader.py、tests/test_mysql_loader.py、docs/tasks/T4-mysql-loader/ 目录（status.json 中 task 写 "T4"）。实现前先读参照代码：src/govio/metadata/trino_loader.py（同类参照实现）、src/govio/metadata/database.py 的 MetadataLoader 基类、src/govio/metadata/duckdb_loader.py、tests/test_trino_loader.py 的打桩测试风格、src/govio/metadata/datasource.py 的 qualify。要求：MysqlLoader(MetadataLoader)，用 pymysql（已是依赖）读 information_schema.tables/columns 抽取表与列（含表注释、列注释，类型解析出 size/precision/scale），输出 DataFrame 列结构与 TrinoLoader/DuckDBLoader 完全一致，data_entity_type 用 MYSQL_TABLE / MYSQL_COLUMN，支持 datasource_name 全限定（qualify）与 schema 过滤。测试用 monkeypatch 打桩连接，不依赖真实数据库，覆盖：表清单、列清单、注释为空时回退列名、类型解析（decimal(10,2)/varchar(64)/bigint）、order_no 递增、空 schema、多 schema。CLI 接入不在本任务范围（T6 负责），不要改 cli/meta.py，也不要改 docs/specs/ 下文档（spec 草稿写到你的任务目录即可）。'
  },
  {
    key: 't5',
    label: '实现 PostgreSQL 元数据加载器',
    agent: 'worker',
    task: '【T5 PostgreSQL 元数据加载器】' + REPO + '。目标：新增 PostgresLoader，从 PostgreSQL 抽取表/列元数据。' + RULES + ' 任务所有权：src/govio/metadata/postgres_loader.py、tests/test_postgres_loader.py、pyproject.toml、docs/tasks/T5-postgres-loader/ 目录（status.json 中 task 写 "T5"）。实现前先读参照代码：src/govio/metadata/trino_loader.py、src/govio/metadata/database.py 的 MetadataLoader 基类、src/govio/metadata/duckdb_loader.py、tests/test_trino_loader.py、src/govio/metadata/datasource.py 的 qualify。要求：PostgresLoader(MetadataLoader)，用 information_schema.tables/columns + pg_catalog（col_description 取列注释、obj_description 取表注释）抽取表与列（含类型解析 size/precision/scale），输出 DataFrame 列结构与 TrinoLoader/DuckDBLoader 完全一致，data_entity_type 用 POSTGRES_TABLE / POSTGRES_COLUMN，支持 datasource_name 全限定（qualify）与 schema 过滤。驱动选型倾向 psycopg[binary]（psycopg3），加入 pyproject.toml dependencies 并在 progress.md 记录选型理由（若改用其他驱动须说明）。测试用 monkeypatch 打桩连接，不依赖真实数据库，覆盖：表清单、列清单、注释回退、类型解析、order_no 递增、空 schema、多 schema。CLI 接入不在本任务范围（T6 负责），不要改 cli/meta.py 与 docs/specs/ 下文档。'
  }
]);

const rec1 = await runs.run('rec1', {
  label: '记录批次1进展到计划 JSON',
  agent: 'delegate',
  task: '【记录员 R1·批次1】' + REPO + '。你只允许修改 docs/task-plan.json 这一个文件。读取 docs/tasks/T1-distribution-design/status.json、docs/tasks/T4-mysql-loader/status.json、docs/tasks/T5-postgres-loader/status.json（若缺 status.json 则读同目录 progress.md 推断状态；都没有则标 failed）。把 docs/task-plan.json 中 "id" 为 T1、T4、T5 的任务对象更新四个字段：status（done/partial/failed）、summary（取 status.json 的 summary）、files（取 status.json 的 files）、completedAt（ISO 日期字符串）。其余字段、其他任务对象与 JSON 结构保持原样。不要改其他任何文件，不要运行 git。'
});

const batch2 = await runs.all([
  {
    key: 't2',
    label: '实现 pi.agent 扩展分发',
    agent: 'worker',
    task: '【T2 pi.agent 扩展分发实现】' + REPO + '。目标：按 T1 的设计实现 govio 的 pi.agent 扩展分发形态。' + RULES + ' 任务所有权：packaging/pi/（新建目录）、scripts/package_pi_extension.py（新建）、docs/tasks/T2-pi-extension/ 目录（status.json 中 task 写 "T2"）。先读 docs/tasks/T1-distribution-design/design.md 与 research.md（T1 已完成），严格按其中 pi 扩展的设计实现；若设计缺失关键决策，按最简可用原则自行决策并在 progress.md 记录。实现内容：扩展包目录结构（按设计组织 skills 与 govio-cli 的安装/注册接线）、可重复执行的打包脚本（产物放 dist/ 下，命名不与 dist/govio-skills.zip 冲突）、产物结构自检（脚本内置校验或在 progress.md 记录产物清单与验证结果）。硬约束：不得修改 docs/install.md、docs/roadmap.md、docs/specs/、package_skills.py、pyproject.toml、src/、skills/ 等共享文件；安装文档草稿写到你的任务目录，T7 会整合。'
  },
  {
    key: 't3',
    label: '实现 DeepSeek harness 插件分发',
    agent: 'worker',
    task: '【T3 DeepSeek harness 插件分发实现】' + REPO + '。目标：按 T1 的设计实现 govio 的 DeepSeek harness 插件分发形态。' + RULES + ' 任务所有权：packaging/deepseek/（新建目录）、scripts/package_harness_plugin.py（新建）、docs/tasks/T3-harness-plugin/ 目录（status.json 中 task 写 "T3"）。先读 docs/tasks/T1-distribution-design/design.md 与 research.md（T1 已完成），严格按其中 DeepSeek harness 插件的设计实现；若 T1 对 DeepSeek harness 只给出假设性适配层，按该假设实现并在 progress.md 明确标注假设与风险。实现内容：插件包目录结构、可重复执行的打包脚本（产物放 dist/ 下，命名不冲突）、产物结构自检。硬约束：不得修改 docs/install.md、docs/roadmap.md、docs/specs/、package_skills.py、pyproject.toml、src/、skills/ 等共享文件；安装文档草稿写到你的任务目录，T7 会整合。'
  },
  {
    key: 't6',
    label: '接入多数据库 CLI 与查询执行',
    agent: 'worker',
    task: '【T6 多数据库 CLI 接入与查询执行统一】' + REPO + '。目标：把 mysql / postgres / trino 接入元数据导入与查询执行链路。' + RULES + ' 任务所有权：src/govio/cli/meta.py、src/govio/cli/main.py、src/govio/observe_data/、tests/test_meta_cli_sources.py（新建）、tests/test_observe_multi_db.py（新建）、docs/specs/metadata.md、docs/specs/cli.md、docs/specs/observe-data.md、docs/tasks/T6-query-cli-integration/ 目录（status.json 中 task 写 "T6"）。背景：T4 已实现 src/govio/metadata/mysql_loader.py、T5 已实现 src/govio/metadata/postgres_loader.py（先读这两个文件确认构造参数与输出契约）；src/govio/metadata/trino_loader.py 早已存在但未接入 CLI。实现内容：(1) govio-cli meta meta --source 增加 mysql / postgres / trino 取值（现状只有 tds/duckdb，见 src/govio/cli/meta.py 的 cmd_meta/step_meta_export 与 src/govio/cli/main.py 的 argparse 定义），参数校验与报错风格向现有 duckdb 分支看齐，注意 --db 在 duckdb 下是文件路径，mysql/postgres 下应是连接 URL（或按你的判断给出一致且在 spec 中文档化的参数语义）；(2) observe 查询执行链路（src/govio/observe_data/core/database.py 的 DatabaseManager 与 tools/load_dataframe.py）支持 postgresql 与 trino 的查询执行，并确认 mysql 查询执行可用（duckdb 原生、mysql 走 sqlalchemy 的现状先核实；postgres 可走 sqlalchemy+psycopg，trino 可走 trino 包自带 sqlalchemy dialect 或 dbapi，自行选型并在 progress.md 记录理由）；(3) 新增测试打桩不依赖真实数据库，覆盖 CLI 参数分支与连接参数传递；(4) docs/specs/metadata.md、cli.md、observe-data.md 与实现严格同步（AGENTS.md 文档同步规则，不得留过时类名/参数描述）。验证：uv run pytest tests/test_meta_cli_sources.py tests/test_observe_multi_db.py tests/test_trino_loader.py -q 与 uv run ruff check src/ tests/，并抽查既有测试 tests/test_meta_cli_guard.py、tests/test_meta_export_ids.py 未被破坏。'
  }
]);

const rec2 = await runs.run('rec2', {
  label: '记录批次2进展到计划 JSON',
  agent: 'delegate',
  task: '【记录员 R2·批次2】' + REPO + '。你只允许修改 docs/task-plan.json 这一个文件。读取 docs/tasks/T2-pi-extension/status.json、docs/tasks/T3-harness-plugin/status.json、docs/tasks/T6-query-cli-integration/status.json（若缺 status.json 则读同目录 progress.md 推断状态；都没有则标 failed）。把 docs/task-plan.json 中 "id" 为 T2、T3、T6 的任务对象更新四个字段：status（done/partial/failed）、summary、files、completedAt（ISO 日期字符串）。其余字段、其他任务对象与 JSON 结构保持原样。不要改其他任何文件，不要运行 git。'
});

const t7 = await runs.run('t7', {
  label: '架构审视、路线图修正与全量验证',
  agent: 'worker',
  task: '【T7 架构审视、功能精简与路线图修正】' + REPO + '。目标：作为整合收尾任务，完成架构审视、路线图修正、安装文档整合和全量验证。' + RULES + ' 任务所有权：docs/roadmap.md、docs/install.md、docs/specs/README.md、docs/tasks/T7-architecture-roadmap/ 目录（status.json 中 task 写 "T7"；若验证后需 bug 级小修可超出所有权清单，但仅限小改且必须在 progress.md 列明）。前置：T1~T6 已完成，先读 docs/task-plan.json、docs/tasks/ 下各任务的 progress.md 与 status.json 了解本轮全部产出，再读 AGENTS.md、docs/roadmap.md、docs/install.md、docs/specs/README.md、docs/design-thoughts.md。产出：(1) docs/tasks/T7-architecture-roadmap/design.md——架构审视与功能精简建议：在 govio 作为"无 LLM、可插拔任意 Agent Runtime 的 skill+CLI"定位下，审视模块边界（metadata/graph/observe_data/core/cli）与三种分发形态（zip+uv tool、pi 扩展、DeepSeek harness 插件）的统一抽象，指出冗余、可删减功能与应停止的路线，给出未来 2-3 个迭代的精简方向（只写设计建议，不做大重构）；(2) docs/roadmap.md 修正——把本轮新能力（mysql/postgres loader、查询执行统一、三种分发形态）纳入，重排 Phase 与优先级，删除或标注过时项，与实现保持一致；(3) docs/install.md 整合三种分发形态的安装/升级文档（T2/T3 的安装草稿在 docs/tasks/T2-pi-extension/ 与 docs/tasks/T3-harness-plugin/ 下）；(4) docs/specs/README.md 同步依赖与模块变更（T5 新增驱动依赖、新增 mysql_loader/postgres_loader 模块、新增 packaging/ 与 scripts/ 打包脚本）。验证：uv run ruff check src/ tests/ 与 uv run pytest tests/ 全量必须通过；失败时小问题直接修，大问题写入 status.json 的 summary 并标 partial。禁止重构性改动。'
});

const rec3 = await runs.run('rec3', {
  label: '记录批次3进展到计划 JSON',
  agent: 'delegate',
  task: '【记录员 R3·批次3】' + REPO + '。你只允许修改 docs/task-plan.json 这一个文件。读取 docs/tasks/T7-architecture-roadmap/status.json（若缺则读 progress.md 推断；都没有则标 failed）。把 docs/task-plan.json 中 "id" 为 T7 的任务对象更新四个字段：status、summary、files、completedAt（ISO 日期字符串）。保持 JSON 结构与其余内容原样。不要改其他任何文件，不要运行 git。'
});

return {
  batch1: batch1.map(r => r.output),
  rec1: rec1.output,
  batch2: batch2.map(r => r.output),
  rec2: rec2.output,
  t7: t7.output,
  rec3: rec3.output
};
