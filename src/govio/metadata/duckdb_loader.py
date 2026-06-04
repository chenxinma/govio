import duckdb
import pandas as pd

from .database import MetadataLoader


class DuckDBLoader(MetadataLoader):
    """Load metadata from a local DuckDB file.

    Reads table and column metadata directly from a .duckdb file using the
    duckdb Python library, without SQLAlchemy.  Produces DataFrames with the
    same column schema as TDSLoader but uses ``DUCKDB_TABLE`` /
    ``DUCKDB_COLUMN`` as ``data_entity_type``.
    """

    def __init__(self, db_path: str, schemas: list[str]) -> None:
        self.db_path = db_path
        self.schemas = schemas

    def load_tables(self) -> pd.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            df = conn.execute(
                """
                SELECT schema_name || '.' || table_name AS full_table_name,
                       schema_name AS "schema",
                       table_name AS table_name,
                       COALESCE(comment, table_name) AS "name",
                       'DUCKDB_TABLE' AS data_entity_type,
                       '' AS database_name
                FROM duckdb_tables()
                WHERE schema_name IN (SELECT unnest(?))
                ORDER BY schema_name, table_name
                """,
                [self.schemas],
            ).fetchdf()
        finally:
            conn.close()
        return df

    def load_columns(self) -> pd.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            df = conn.execute(
                """
                SELECT c.table_schema || '.' || c.table_name || '.' || c.column_name AS "column",
                       c.column_name AS column_name,
                       COALESCE(dc.comment, c.column_name) AS "name",
                       c.table_schema || '.' || c.table_name AS full_table_name,
                       'DUCKDB_COLUMN' AS data_entity_type,
                       c.data_type AS dtype,
                       0 AS "size",
                       0 AS "precision",
                       0 AS "scale",
                       c.ordinal_position AS order_no,
                       c.data_type AS data_type
                FROM information_schema.columns c
                LEFT JOIN duckdb_columns() dc
                    ON dc.schema_name = c.table_schema
                    AND dc.table_name = c.table_name
                    AND dc.column_name = c.column_name
                WHERE c.table_schema IN (SELECT unnest(?))
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
                """,
                [self.schemas],
            ).fetchdf()
        finally:
            conn.close()
        return df
