import pytest
import pandas as pd
from govio.cli.meta_export import merge_metadata


def test_merge_metadata_duckdb_wins():
    """DuckDB rows take precedence over TDS rows on conflict."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1", "s1.t2"],
        "name": ["TDS Table 1", "TDS Table 2"],
    })
    df_duck = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["DuckDB Table 1"],
    })
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 2
    t1_row = result[result["full_table_name"] == "s1.t1"].iloc[0]
    assert t1_row["name"] == "DuckDB Table 1"


def test_merge_metadata_adds_new():
    """DuckDB-only rows are included in merge result."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["TDS Table 1"],
    })
    df_duck = pd.DataFrame({
        "full_table_name": ["s1.t2"],
        "name": ["DuckDB Table 2"],
    })
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 2
    assert set(result["full_table_name"]) == {"s1.t1", "s1.t2"}


def test_merge_metadata_tds_only():
    """Works when DuckDB has no rows."""
    df_tds = pd.DataFrame({
        "full_table_name": ["s1.t1"],
        "name": ["TDS Table 1"],
    })
    df_duck = pd.DataFrame(columns=["full_table_name", "name"])
    result = merge_metadata(df_tds, df_duck, "full_table_name")
    assert len(result) == 1
