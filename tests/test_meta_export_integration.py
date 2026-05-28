import pandas as pd
import duckdb
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.cli.meta_export import merge_metadata


def test_full_merge_pipeline(tmp_path):
    """End-to-end: DuckDB load -> merge with mock TDS -> correct output."""
    # Setup DuckDB
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE SCHEMA IF NOT EXISTS s1")
    conn.execute("CREATE TABLE s1.t1 (id INTEGER, name VARCHAR)")
    conn.execute("CREATE TABLE s1.t2 (val DECIMAL(10,2))")
    conn.close()

    # Load DuckDB
    loader = DuckDBLoader(db_path, ["s1"])
    df_duck_tables = loader.PhysicalTable
    df_duck_cols = loader.Col

    # Mock TDS data (s1.t1 exists in TDS too, s1.t3 is TDS-only)
    df_tds_tables = pd.DataFrame({
        "full_table_name": ["s1.t1", "s1.t3"],
        "schema": ["s1", "s1"],
        "table_name": ["t1", "t3"],
        "name": ["TDS Table 1", "TDS Table 3"],
        "data_entity_type": ["MYSQL_TABLE", "MYSQL_TABLE"],
        "database_name": ["db1", "db1"],
    })
    df_tds_cols = pd.DataFrame({
        "column": ["s1.t1.id", "s1.t3.col1"],
        "column_name": ["id", "col1"],
        "name": ["TDS ID", "TDS Col1"],
        "full_table_name": ["s1.t1", "s1.t3"],
        "data_entity_type": ["MYSQL_COLUMN", "MYSQL_COLUMN"],
        "dtype": ["int", "varchar"],
        "size": [0, 255],
        "precision": [0, 0],
        "scale": [0, 0],
        "order_no": [1, 1],
        "data_type": ["int", "varchar(255)"],
    })

    # Merge
    df_tables = merge_metadata(df_tds_tables, df_duck_tables, "full_table_name")
    df_cols = merge_metadata(df_tds_cols, df_duck_cols, "column")

    # Assertions: 3 tables (t1 from DuckDB wins, t2 DuckDB-only, t3 TDS-only)
    assert len(df_tables) == 3
    t1_row = df_tables[df_tables["table_name"] == "t1"].iloc[0]
    assert t1_row["data_entity_type"] == "DUCKDB_TABLE"  # DuckDB wins

    # Columns: s1.t1.id from DuckDB wins, s1.t1.name DuckDB-only,
    # s1.t2.* DuckDB-only, s1.t3.col1 TDS-only
    assert "s1.t3.col1" in df_cols["column"].values  # TDS-only column preserved
    id_row = df_cols[df_cols["column"] == "s1.t1.id"].iloc[0]
    assert id_row["data_entity_type"] == "DUCKDB_COLUMN"  # DuckDB wins
