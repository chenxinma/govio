"""meta 命令组 — 知识图库维护

提供元数据导入、JSON Schema 输出、数据标准推荐等功能。

各导入子命令（meta / std / compliance / rel / metric）完全独立运行，
所有输入通过 CLI 参数显式传入，不依赖配置文件。通过 output 目录的 CSV 文件
作为共享状态，支持增量合并（幂等）。

推荐执行顺序：meta → std → compliance → rel → metric → graph

原单文件 `cli/meta.py`（1294 行）按子命令域拆分（Phase 3 工程精简第 4 项），
模块职责与依赖方向（parser → commands → steps / graph_ops → merge / common）：

- `constants` — `DEFAULT_ASSETS_DIR` 与 `--source` 相关映射（展示名、schema 提示、URL scheme/默认值）
- `parser`    — `meta()` argparse 装配，唯一入口
- `commands`  — 各 `cmd_*`：参数校验、退出码（失败 1 / 成功含空结果 0）、步骤编排
- `steps`     — 各 `step_*_export`：读源 → 赋 node_id → 幂等写 CSV
- `merge`     — 节点/边 CSV 增量合并与 import-delta 计数
- `graph_ops` — 图后端装载（update/clear）、assets 生成、导入校验报告
- `common`    — 元数据加载器构造（`_build_metadata_loader`）与数据源 schema 解析

本文件只 re-export 函数以保持 `from govio.cli.meta import meta` 等既有导入不变；
加载器 / 配置类不在此 re-export —— 测试打桩请直接 patch 所属子模块
（如 `govio.cli.meta.common.MysqlLoader`、`govio.cli.meta.graph_ops.ConfigManager`）。
"""

from .commands import (
    cmd_compliance,
    cmd_graph,
    cmd_import_schema,
    cmd_meta,
    cmd_metric,
    cmd_recommend,
    cmd_rel,
    cmd_schema,
    cmd_std,
)
from .parser import meta
from .steps import (
    step_compliance_export,
    step_meta_export,
    step_metric_export,
    step_rel_export,
    step_std_export,
)

__all__ = [
    "cmd_compliance",
    "cmd_graph",
    "cmd_import_schema",
    "cmd_meta",
    "cmd_metric",
    "cmd_recommend",
    "cmd_rel",
    "cmd_schema",
    "cmd_std",
    "meta",
    "step_compliance_export",
    "step_meta_export",
    "step_metric_export",
    "step_rel_export",
    "step_std_export",
]
