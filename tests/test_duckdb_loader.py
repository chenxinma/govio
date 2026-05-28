import pytest
import pandas as pd
import duckdb
from pathlib import Path
from govio.metadata.duckdb_loader import DuckDBLoader
from govio.metadata.database import MetadataLoader


@pytest.fixture
def sample_duckdb(tmp_path):
    """Create a sample DuckDB with test tables and columns."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
    conn.execute("""
        CREATE TABLE test_schema.users (
            id INTEGER,
            name VARCHAR
        )
    """)
    conn.execute("COMMENT ON TABLE test_schema.users IS '用户表'")
    conn.execute("COMMENT ON COLUMN test_schema.users.id IS '用户ID'")
    conn.execute("COMMENT ON COLUMN test_schema.users.name IS '用户名'")
    conn.execute("""
        CREATE TABLE test_schema.orders (
            order_id INTEGER,
            amount DECIMAL(10,2)
        )
    """)
    conn.close()
    return db_path


def test_duckdb_loader_is_metadata_loader(sample_duckdb):
    """DuckDBLoader is a subclass of MetadataLoader."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    assert isinstance(loader, MetadataLoader)


def test_duckdb_loader_load_tables(sample_duckdb):
    """DuckDBLoader loads tables with expected columns."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_tables()
    assert len(df) == 2
    assert "full_table_name" in df.columns
    assert "schema" in df.columns
    assert "table_name" in df.columns
    assert "name" in df.columns
    assert "data_entity_type" in df.columns
    assert all(df["data_entity_type"] == "DUCKDB_TABLE")


def test_duckdb_loader_load_columns(sample_duckdb):
    """DuckDBLoader loads columns with expected structure."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_columns()
    assert len(df) == 4
    assert "column" in df.columns
    assert "column_name" in df.columns
    assert "name" in df.columns
    assert "full_table_name" in df.columns
    assert "data_entity_type" in df.columns
    assert "dtype" in df.columns
    assert "data_type" in df.columns
    assert all(df["data_entity_type"] == "DUCKDB_COLUMN")


def test_duckdb_loader_properties(sample_duckdb):
    """PhysicalTable and Col properties work."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    assert len(loader.PhysicalTable) == 2
    assert len(loader.Col) == 4


def test_duckdb_loader_table_comments(sample_duckdb):
    """Table comments are loaded correctly."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_tables()
    users_row = df[df["table_name"] == "users"].iloc[0]
    assert users_row["name"] == "用户表"


def test_duckdb_loader_column_comments(sample_duckdb):
    """Column comments are loaded correctly."""
    loader = DuckDBLoader(sample_duckdb, ["test_schema"])
    df = loader.load_columns()
    id_row = df[df["column_name"] == "id"].iloc[0]
    assert id_row["name"] == "用户ID"
