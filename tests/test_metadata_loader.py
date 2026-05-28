import pytest
import pandas as pd
from govio.metadata.database import MetadataLoader


def test_metadata_loader_is_abstract():
    """MetadataLoader cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MetadataLoader()


def test_metadata_loader_requires_load_tables():
    """Subclass must implement load_tables."""

    class IncompleteLoader(MetadataLoader):
        def load_columns(self):
            return pd.DataFrame()

    with pytest.raises(TypeError):
        IncompleteLoader()


def test_metadata_loader_requires_load_columns():
    """Subclass must implement load_columns."""

    class IncompleteLoader(MetadataLoader):
        def load_tables(self):
            return pd.DataFrame()

    with pytest.raises(TypeError):
        IncompleteLoader()


def test_metadata_loader_properties_delegate():
    """PhysicalTable and Col properties delegate to load_tables/load_columns."""

    class MockLoader(MetadataLoader):
        def load_tables(self):
            return pd.DataFrame({"full_table_name": ["a.b"]})

        def load_columns(self):
            return pd.DataFrame({"column": ["a.b.c"]})

    loader = MockLoader()
    assert list(loader.PhysicalTable.columns) == ["full_table_name"]
    assert list(loader.Col.columns) == ["column"]
    assert len(loader.PhysicalTable) == 1
    assert len(loader.Col) == 1
