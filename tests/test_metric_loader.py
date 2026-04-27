import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from govio.metadata.metric import MetricLoader, load_metrics


@pytest.fixture
def sample_tables():
    return pd.DataFrame(
        {
            "full_table_name": ["db.schema.t_bill", "db.schema.t_risk", "db.schema.t_leads"],
            "name": ["t_bill", "t_risk", "t_leads"],
            "schema": ["schema", "schema", "schema"],
        }
    )


@pytest.fixture
def sample_columns():
    return pd.DataFrame(
        {
            "column": [
                "db.schema.t_bill.bill_income_amt",
                "db.schema.t_bill.ym",
                "db.schema.t_risk.risk_amt",
                "db.schema.t_leads.leads_forecast_amt",
            ],
            "column_name": ["bill_income_amt", "ym", "risk_amt", "leads_forecast_amt"],
            "full_table_name": [
                "db.schema.t_bill",
                "db.schema.t_bill",
                "db.schema.t_risk",
                "db.schema.t_leads",
            ],
        }
    )


@pytest.fixture
def valid_metric_data():
    return {
        "version": "1.0",
        "shared_dimensions": [
            {"code": "ym", "name": "年月", "granularity": "月"},
            {"code": "business_unit", "name": "事业部"},
        ],
        "metrics": [
            {
                "code": "bill_income_amt",
                "name": "当月账单收入",
                "business_definition": "当月实际确认的账单收入金额",
                "type": "atomic",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DWS",
                "source_tables": [
                    {
                        "full_table_name": "db.schema.t_bill",
                        "columns": [
                            {"column_name": "bill_income_amt", "role": "measure"},
                            {"column_name": "ym", "role": "dimension_ref"},
                        ],
                    }
                ],
                "dimensions": [
                    {"code": "ym", "usage_type": "group"},
                    {"code": "business_unit", "usage_type": "slice"},
                ],
            },
            {
                "code": "risk_amt",
                "name": "危机金额",
                "business_definition": "当月危机流失金额",
                "type": "atomic",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DWS",
                "source_tables": [
                    {
                        "full_table_name": "db.schema.t_risk",
                        "columns": [
                            {"column_name": "risk_amt", "role": "measure"},
                        ],
                    }
                ],
                "dimensions": [
                    {"code": "ym", "usage_type": "group"},
                ],
            },
            {
                "code": "burndown_amt",
                "name": "存量消耗额",
                "business_definition": "预计当月产生的账单收入减去当月危机金额",
                "type": "derived",
                "formula": "forecast_income_amt - risk_amt",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DM",
                "derived_from": ["bill_income_amt", "risk_amt"],
                "dimensions": [
                    {"code": "ym", "usage_type": "group"},
                    {"code": "business_unit", "usage_type": "slice"},
                ],
            },
        ],
    }


def _write_json(data, tmpdir):
    path = Path(tmpdir) / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(path)


def test_load_success(sample_tables, sample_columns, valid_metric_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        assert len(loader.Metric) == 3
        assert len(loader.Dimension) == 2
        assert loader.Metric.iloc[0]["code"] == "bill_income_amt"
        assert loader.Metric.iloc[2]["type"] == "derived"
        assert loader.Metric.iloc[2]["formula"] == "forecast_income_amt - risk_amt"


def test_atomic_metric_generates_uses_table_edges(
    sample_tables, sample_columns, valid_metric_data
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        assert len(loader.uses_table_edges) == 2
        assert ":START_ID(Metric)" in loader.uses_table_edges.columns
        assert ":END_ID(PhysicalTable)" in loader.uses_table_edges.columns


def test_atomic_metric_generates_refers_column_edges(
    sample_tables, sample_columns, valid_metric_data
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        assert len(loader.refers_column_edges) == 3
        roles = set(loader.refers_column_edges["role"].values)
        assert "measure" in roles
        assert "dimension_ref" in roles


def test_derived_metric_generates_derived_from_edges(
    sample_tables, sample_columns, valid_metric_data
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        assert len(loader.derived_from_edges) == 2
        # burndown_amt depends on bill_income_amt and risk_amt
        start_ids = loader.derived_from_edges[":START_ID(Metric)"].values
        assert all(sid == 2 for sid in start_ids)


def test_dimension_used_edges(sample_tables, sample_columns, valid_metric_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        total_dims = sum(len(m.get("dimensions", [])) for m in valid_metric_data["metrics"])
        assert len(loader.dimension_used_edges) == total_dims
        assert "usage_type" in loader.dimension_used_edges.columns


def test_supersedes_edges_empty(sample_tables, sample_columns, valid_metric_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = MetricLoader(path, sample_tables, sample_columns)

        assert len(loader.supersedes_edges) == 0
        assert ":START_ID(Metric)" in loader.supersedes_edges.columns


def test_invalid_source_table_raises_error(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "shared_dimensions": [{"code": "ym", "name": "年月"}],
        "metrics": [
            {
                "code": "bad_metric",
                "name": "不存在的表",
                "business_definition": "测试",
                "type": "atomic",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DWS",
                "source_tables": [
                    {"full_table_name": "db.schema.nonexistent_table"}
                ],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(data, tmpdir)
        with pytest.raises(ValueError, match="不存在于 PhysicalTable"):
            MetricLoader(path, sample_tables, sample_columns)


def test_invalid_derived_from_raises_error(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "shared_dimensions": [{"code": "ym", "name": "年月"}],
        "metrics": [
            {
                "code": "bad_derived",
                "name": "不存在的依赖",
                "business_definition": "测试",
                "type": "derived",
                "formula": "nonexistent * 2",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DM",
                "derived_from": ["nonexistent_metric"],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(data, tmpdir)
        with pytest.raises(ValueError, match="不在本文件定义的指标中"):
            MetricLoader(path, sample_tables, sample_columns)


def test_circular_dependency_raises_error(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "shared_dimensions": [{"code": "ym", "name": "年月"}],
        "metrics": [
            {
                "code": "metric_a",
                "name": "指标A",
                "business_definition": "测试",
                "type": "derived",
                "formula": "metric_b + 1",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DM",
                "derived_from": ["metric_b"],
            },
            {
                "code": "metric_b",
                "name": "指标B",
                "business_definition": "测试",
                "type": "derived",
                "formula": "metric_a + 1",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DM",
                "derived_from": ["metric_a"],
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(data, tmpdir)
        with pytest.raises(ValueError, match="循环依赖"):
            MetricLoader(path, sample_tables, sample_columns)


def test_file_not_found_raises_error(sample_tables, sample_columns):
    with pytest.raises(FileNotFoundError):
        MetricLoader("/nonexistent/metrics.json", sample_tables, sample_columns)


def test_invalid_dimension_ref_raises_error(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "shared_dimensions": [{"code": "ym", "name": "年月"}],
        "metrics": [
            {
                "code": "bad_dim",
                "name": "无效维度引用",
                "business_definition": "测试",
                "type": "atomic",
                "unit": "万元",
                "data_type": "decimal(18,2)",
                "source_layer": "DWS",
                "source_tables": [
                    {"full_table_name": "db.schema.t_bill"}
                ],
                "dimensions": [
                    {"code": "nonexistent_dim", "usage_type": "group"},
                ],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(data, tmpdir)
        with pytest.raises(ValueError, match="不在 shared_dimensions"):
            MetricLoader(path, sample_tables, sample_columns)


def test_load_metrics_convenience_function(
    sample_tables, sample_columns, valid_metric_data
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_json(valid_metric_data, tmpdir)
        loader = load_metrics(path, sample_tables, sample_columns)
        assert len(loader.Metric) == 3
