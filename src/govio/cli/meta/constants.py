"""meta 命令组的常量：assets 默认目录与 --source 相关映射

拆自原 cli/meta.py 顶部常量区。`SOURCE_URL_SCHEMES` / `SOURCE_URL_DEFAULTS`
约定 `--db` 连接 URL 的合法形态，见 docs/specs/cli.md。
"""

from pathlib import Path

DEFAULT_ASSETS_DIR = Path(".agents/skills/govio/assets")

# --source 取值与展示名（tds 之外的来源都通过 --db 定位数据源）
SOURCE_LABELS = {
    "tds": "TDS",
    "duckdb": "DuckDB",
    "mysql": "MySQL",
    "postgres": "PostgreSQL",
    "trino": "Trino",
}

# --schemas 缺省时的来源相关提示
SCHEMA_HINTS = {
    "duckdb": "DuckDB 文件默认 schema 为 main",
    "mysql": "MySQL 中 schema 即库名",
    "postgres": "PostgreSQL 常用 schema 为 public",
    "trino": "Trino schema 为 catalog 下的命名空间",
}

# --source 对应的 --db 连接 URL scheme（含驱动限定写法），见 docs/specs/cli.md
SOURCE_URL_SCHEMES: dict[str, set[str]] = {
    "mysql": {"mysql", "mysql+pymysql"},
    "postgres": {"postgres", "postgresql", "postgresql+psycopg"},
    "trino": {"trino", "trino+http", "trino+https"},
}

# 连接 URL 未给出 user / port 时的默认值，与各加载器构造参数默认值一致
SOURCE_URL_DEFAULTS: dict[str, tuple[str, int]] = {
    "mysql": ("root", 3306),
    "postgres": ("postgres", 5432),
    "trino": ("trino", 8080),
}
