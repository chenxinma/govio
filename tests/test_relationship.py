import json
import tempfile

import pandas as pd
import pytest

from govio.metadata.relationship import RelationshipLoader, load_relationships


@pytest.fixture
def sample_tables():
    return pd.DataFrame(
        {
            "full_table_name": ["db.schema.table1", "db.schema.table2"],
            "name": ["table1", "table2"],
            "schema": ["schema", "schema"],
        }
    )


@pytest.fixture
def sample_columns():
    return pd.DataFrame(
        {
            "column": [
                "db.schema.table1.col1",
                "db.schema.table1.col2",
                "db.schema.table2.col_a",
                "db.schema.table2.col_b",
            ]
        }
    )


@pytest.fixture
def valid_json_data():
    return {
        "version": "1.0",
        "relationships": [
            {
                "description": "外键关联",
                "source": {"PhysicalTable": "db.schema.table1", "Cols": ["col1"]},
                "target": {"PhysicalTable": "db.schema.table2", "Cols": ["col_a"]},
                "relationship_type": "many_to_one",
            }
        ],
    }


def test_load_json_success(sample_tables, sample_columns, valid_json_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_json_data, f)
        f.flush()

        loader = RelationshipLoader(f.name, sample_tables, sample_columns)
        data = loader.load_json()

        assert data["version"] == "1.0"
        assert len(data["relationships"]) == 1


def test_validate_relationship_valid(sample_tables, sample_columns):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"version": "1.0", "relationships": []}, f)
        f.flush()

        loader = RelationshipLoader(f.name, sample_tables, sample_columns)

        rel = {
            "source": {"PhysicalTable": "t1", "Cols": ["c1"]},
            "target": {"PhysicalTable": "t2", "Cols": ["c2"]},
            "relationship_type": "many_to_one",
        }

        assert loader.validate_relationship(rel, 0) is True


def test_validate_relationship_invalid_type(sample_tables, sample_columns):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"version": "1.0", "relationships": []}, f)
        f.flush()

        loader = RelationshipLoader(f.name, sample_tables, sample_columns)

        rel = {
            "source": {"PhysicalTable": "t1", "Cols": ["c1"]},
            "target": {"PhysicalTable": "t2", "Cols": ["c2"]},
            "relationship_type": "invalid_type",
        }

        assert loader.validate_relationship(rel, 0) is False


def test_load_relationships_success(sample_tables, sample_columns, valid_json_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_json_data, f)
        f.flush()

        df = load_relationships(f.name, sample_tables, sample_columns)

        assert len(df) == 1
        assert df.iloc[0]["source"] == "db.schema.table1"
        assert df.iloc[0]["target"] == "db.schema.table2"
        assert df.iloc[0]["relationship_type"] == "many_to_one"
        assert df.iloc[0]["source_columns"] == "col1"
        assert df.iloc[0]["target_columns"] == "col_a"


def test_load_relationships_invalid_table(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "relationships": [
            {
                "source": {"PhysicalTable": "nonexistent", "Cols": ["c1"]},
                "target": {"PhysicalTable": "db.schema.table2", "Cols": ["c2"]},
                "relationship_type": "many_to_one",
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()

        df = load_relationships(f.name, sample_tables, sample_columns)

        assert len(df) == 0


def test_composite_key_relationship(sample_tables, sample_columns):
    data = {
        "version": "1.0",
        "relationships": [
            {
                "description": "复合键关联",
                "source": {
                    "PhysicalTable": "db.schema.table1",
                    "Cols": ["col1", "col2"],
                },
                "target": {
                    "PhysicalTable": "db.schema.table2",
                    "Cols": ["col_a", "col_b"],
                },
                "relationship_type": "many_to_many",
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()

        df = load_relationships(f.name, sample_tables, sample_columns)

        assert len(df) == 1
        assert df.iloc[0]["source_columns"] == "col1,col2"
        assert df.iloc[0]["target_columns"] == "col_a,col_b"
