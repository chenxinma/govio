"""meta 的公共辅助 —— 元数据加载器构造与来源参数解析

`_build_metadata_loader` 是各导入子命令（`meta meta` / `meta import-schema`）
构造加载器的唯一入口：`--source` 决定加载器类型，`--db` 的 URL 形态必须与之匹配。
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import make_url
from trino.auth import BasicAuthentication

from govio.metadata.database import MetadataLoader
from govio.metadata.datasource import DatasourceLoader
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.mysql_loader import MysqlLoader
from govio.metadata.postgres_loader import PostgresLoader
from govio.metadata.trino_loader import TrinoLoader

from .constants import SOURCE_URL_DEFAULTS, SOURCE_URL_SCHEMES


def _describe_duckdb_schemas(db_path: str) -> str:
    """只读列出 DuckDB 文件中可导入的 schema，用于空结果时的错误提示。"""
    try:
        pairs = DuckDBLoader(db_path, []).list_schemas()
    except Exception as e:
        return f"（读取 schema 列表失败: {e}）"
    if not pairs:
        return "（该文件没有可导入的 schema）"
    return "、".join(f"{name}({count} 张表)" for name, count in pairs)


def _build_metadata_loader(
    source: str, db: str, schemas: list[str], datasource_name: str
) -> MetadataLoader:
    """按 --source 构造元数据加载器。

    duckdb 的 db 为文件路径；mysql / postgres / trino 的 db 为连接 URL，
    scheme 必须与 --source 匹配（格式见 docs/specs/cli.md）。

    Args:
        source: 信息来源（duckdb / mysql / postgres / trino）
        db: DuckDB 文件路径，或 mysql / postgres / trino 连接 URL
        schemas: 导入的 schema 列表（非 TDS 导入的实际过滤范围）
        datasource_name: 数据源名（全限定前缀与 Datasource / OWNS 归属）

    Returns:
        MetadataLoader: 对应来源的元数据加载器

    Raises:
        ValueError: URL 不合法、scheme 与 --source 不匹配或缺少必要 URL 段
    """
    if source == "duckdb":
        return DuckDBLoader(db, schemas, datasource_name)
    if source not in SOURCE_URL_SCHEMES:
        raise ValueError(f"不支持的信息来源: {source}")

    try:
        url = make_url(db)
    except Exception as e:
        raise ValueError(f"--db 不是合法的连接 URL: {db}（{e}）") from e

    # scheme 匹配大小写不敏感（与 observe 的 scheme.lower() 归一行为一致）
    if url.drivername.lower() not in SOURCE_URL_SCHEMES[source]:
        expected = " / ".join(sorted(SOURCE_URL_SCHEMES[source]))
        raise ValueError(
            f"--db 的 URL scheme '{url.drivername}' 与 --source {source} 不匹配"
            f"（期望: {expected}）"
        )
    if not url.host:
        raise ValueError(f"--db 的 URL 缺少 host: {db}")

    default_user, default_port = SOURCE_URL_DEFAULTS[source]
    user = url.username or default_user
    password = url.password or ""
    port = url.port or default_port

    if source == "mysql":
        return MysqlLoader(
            host=url.host,
            schemas=schemas,
            port=port,
            user=user,
            password=password,
            database=url.database or "",
            datasource_name=datasource_name,
        )
    if source == "postgres":
        return PostgresLoader(
            host=url.host,
            schemas=schemas,
            dbname=url.database or "postgres",
            port=port,
            user=user,
            password=password,
            datasource_name=datasource_name,
        )

    # trino：路径恰好一段为 catalog，schema 由 --schemas 指定
    catalog = url.database or ""
    if not catalog or "/" in catalog:
        raise ValueError(
            "trino URL 路径必须恰好为 catalog 一段"
            "（如 trino://user@host:8080/catalog），schema 由 --schemas 指定"
        )
    if password and not url.username:
        raise ValueError("trino URL 指定 password 时必须同时指定 user")
    auth = BasicAuthentication(user, password) if password else None
    return TrinoLoader(
        host=url.host,
        catalog=catalog,
        schemas=schemas,
        port=port,
        user=user,
        http_scheme="https" if url.drivername.lower() == "trino+https" else "http",
        auth=auth,
        datasource_name=datasource_name,
    )


def _resolve_datasource_schemas(
    datasource_name: str, datasource_file: str, csv_dir: Path
) -> list[str]:
    """解析 --datasource 的 filter.schemas：声明文件优先，其次 csv-dir/Datasource.csv"""
    if datasource_file:
        try:
            return DatasourceLoader(datasource_file).schemas_for(datasource_name)
        except Exception as e:
            print(f"❌ 无法加载数据源声明文件: {e}", file=sys.stderr)
            return []
    csv_path = csv_dir / "Datasource.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        rows = df[df["datasource_name"] == datasource_name]
        if not rows.empty:
            flt = json.loads(rows.iloc[0].get("filter") or "{}")
            return [s for s in flt.get("schemas", []) if s]
    return []
