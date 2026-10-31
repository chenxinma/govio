// review-loop 第 1 轮：3 个并行 fresh-context reviewer（正确性 / 测试 / 契约与文档）
const COMMON = '你是 fresh-context 独立代码评审员。仓库：D:/work/gov-io/govio（分支 experiment/new-work-mode）；评审目标 = 该分支相对 main 的 9 个提交（53 文件，+5845/-61），diff 已物化供阅读：代码 diff（src/ tests/ pyproject.toml）在 /tmp/claude-1000/review/diff-code.patch，文档与打包 diff（docs/ packaging/ scripts/ .gitignore）在 /tmp/claude-1000/review/diff-docs-packaging.patch，提交清单在 /tmp/claude-1000/review/commits.txt。你只有只读工具（read/grep/find/ls），不得修改任何文件；用 read 读取物化 diff 与仓库源文件/契约文件逐条核对。报告规则：只报告由该 diff 引入或使其可达的具体现存问题，每条须附证据（文件:行号、可复现测试/命令、或契约矛盾），按 P0（阻塞合并）/P1（发布前应修）/P2（仅报告）分级；不报风格好恶、不报与 diff 无关的既有问题、不报无契约无复现的臆测风险。报告结尾单独一行输出 Merge verdict: BLOCK 或 Merge verdict: OK 或 Merge verdict: OK with notes。';

const reviews = await runs.all([
  {
    key: 'r-correctness',
    label: '评审正确性与回归',
    agent: 'reviewer',
    task: '【评审角度：正确性与回归（Python 运行时行为）】' + COMMON + ' 重点文件：src/govio/metadata/mysql_loader.py、src/govio/metadata/postgres_loader.py（两个新 loader）、src/govio/cli/meta.py（--source 扩展与 _build_metadata_loader）、src/govio/observe_data/core/database.py（_resolve_sqlalchemy_target）、src/govio/cli/query.py（Decimal 序列化修复）。契约参照（同仓库）：src/govio/metadata/database.py 的 MetadataLoader 基类、src/govio/metadata/duckdb_loader.py 与 src/govio/metadata/trino_loader.py 的输出列契约、src/govio/metadata/datasource.py 的 qualify。核查点（示例，不限于）：两个新 loader 的输出 DataFrame 列名/顺序/语义与既有 loader 是否逐列一致；SQL 标识符拼接的注入面与参数化过滤的正确性；URL 解析与 scheme 校验边界（缺 host、带密码、scheme 大小写、驱动后缀、trino+https）；注释与类型解析对 None/NaN/空串的边界；--source 扩展对既有 duckdb/tds 路径的回归；query.py 用 df.map 将 Decimal 转 str 是否改变其他值语义或引入大结果集性能问题。'
  },
  {
    key: 'r-tests',
    label: '评审测试与验证质量',
    agent: 'reviewer',
    task: '【评审角度：测试与验证质量】' + COMMON + ' 重点文件：tests/test_mysql_loader.py、tests/test_postgres_loader.py、tests/test_meta_cli_sources.py、tests/test_observe_multi_db.py、docs/tasks/T3-harness-plugin/test_package_harness_plugin.py。核查点：打桩对真实驱动语义的保真度（pymysql/psycopg 返回行结构、information_schema/pg_catalog 查询结果形状）——打桩是否可能掩盖真实连接下的失败；断言强度（是否存在 false-green：只断言不抛异常、只核列名）；关键缺失场景（URL 校验失败分支、注释为 NULL、schema 为空、BASE TABLE 过滤、连接关闭）；测试与实现同源错误（照抄实现的错误假设）；docs/tasks/T3-harness-plugin/test_package_harness_plugin.py 放在 docs/tasks 下会被 uv run pytest tests/ 遗漏的覆盖缺口评估。你没有 bash 无法运行测试，通过逐条对照测试代码与被测代码判断。'
  },
  {
    key: 'r-contracts-docs',
    label: '评审契约一致性与文档同步',
    agent: 'reviewer',
    task: '【评审角度：契约一致性与文档同步】' + COMMON + ' 项目规则：AGENTS.md（仓库根）的文档同步规则——修改功能必须同步 docs/specs/ 对应文档，不得留过时类名/方法签名/参数描述。重点文件：packaging/pi/package.json、packaging/pi/extensions/govio-tools.ts、packaging/pi/README.md、packaging/deepseek/package.json、packaging/deepseek/cordis.patch.yml、packaging/deepseek/src/index.ts 与 packaging/deepseek/lib/index.js（编译产物与源是否同步）、packaging/deepseek/tsconfig.json、scripts/package_pi_extension.py、scripts/package_harness_plugin.py、docs/specs/metadata.md、docs/specs/cli.md、docs/specs/observe-data.md、docs/specs/README.md、docs/install.md、docs/roadmap.md。核查点：spec/安装文档与实现逐条对照（参数名、默认值、URL scheme 归一表、产物结构、版本号占位）找不一致；pi 工具注册契约对照本机 pi 文档 C:/Users/Administrator/.pi/agent/install/releases/1.4.0/node_modules/@earendil-works/pi-coding-agent/docs/extensions.md 与 examples/extensions/ 核实 defineTool/registerTool 用法；打包脚本幂等/排除规则的声明与代码是否一致；lib/index.js 与 src/index.ts 是否同步（找源码已改而产物未更新的痕迹）；安装步骤的可执行性（命令、路径、前后置条件）。'
  }
]);

return reviews.map(r => r.output);
