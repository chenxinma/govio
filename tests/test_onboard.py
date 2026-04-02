from pathlib import Path
import tempfile


def create_test_csv_files(csv_dir: Path):
    """创建测试用的 CSV 文件"""
    csv_dir.mkdir(parents=True, exist_ok=True)

    (csv_dir / "PhysicalTable.csv").write_text(
        """:ID(PhysicalTable),name,full_table_name
table1,表1,SCHEMA.TABLE1
""",
        encoding="utf-8",
    )

    (csv_dir / "Col.csv").write_text(
        """:ID(Col),name,column_name,full_table_name
col1,字段1,COL1,SCHEMA.TABLE1
""",
        encoding="utf-8",
    )

    (csv_dir / "Application.csv").write_text(
        """:ID(Application),name,app_name_en
app1,应用1,APP1
""",
        encoding="utf-8",
    )

    (csv_dir / "Standard.csv").write_text(
        """:ID(Standard),name
std1,标准1
""",
        encoding="utf-8",
    )

    (csv_dir / "HAS_COLUMN.csv").write_text(
        """:START_ID(PhysicalTable),:END_ID(Col)
table1,col1
""",
        encoding="utf-8",
    )

    (csv_dir / "USE.csv").write_text(
        """:START_ID(Application),:END_ID(PhysicalTable)
app1,table1
""",
        encoding="utf-8",
    )


def test_onboard_networkx_with_csv(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        output_dir = Path(tmpdir) / "assets"

        create_test_csv_files(csv_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inputs = ["networkx", "yes", str(csv_dir), str(output_dir)]

        input_iter = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _: next(input_iter))

        from govio.core.assets_generator import AssetsGenerator

        from govio.metadata.gen_networkx import build_graph

        gml_path = output_dir / "ontology.gml"
        build_graph(str(csv_dir), str(gml_path))

        from govio import NetworkXGraph

        graph = NetworkXGraph(gml_path)

        generator = AssetsGenerator(graph, output_dir)
        generator.generate_all()

        assert gml_path.exists()
        assert (output_dir / "schema.md").exists()
        assert (output_dir / "names" / "node_names.md").exists()


def test_validate_csv_directory():
    from govio.cli.onboard import validate_csv_directory

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = Path(tmpdir) / "csv"
        csv_dir.mkdir()

        (csv_dir / "PhysicalTable.csv").write_text(
            ":ID(PhysicalTable),name\n", encoding="utf-8"
        )

        assert validate_csv_directory(csv_dir) is True

        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir()

        assert validate_csv_directory(empty_dir) is False
