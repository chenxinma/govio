"""observe chart 测试"""

from pathlib import Path

import pandas as pd
import pytest

from govio.observe_data.core.chart import render_chart


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """带中文列名和中文分类的样本 DataFrame"""
    return pd.DataFrame(
        {
            "地区": ["华北", "华东", "华南", "西南"],
            "销售额": [100, 250, 180, 90],
        }
    )


def test_render_bar_chart_creates_png(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """柱状图: 生成 PNG 文件"""
    output = tmp_path / "bar.png"
    result = render_chart(
        df=sample_df,
        chart_type="bar",
        x_col="地区",
        y_col="销售额",
        output_path=str(output),
    )
    assert result["success"] is True
    assert output.exists()
    assert output.stat().st_size > 0


def test_render_line_chart_creates_png(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """折线图: 生成 PNG 文件"""
    output = tmp_path / "line.png"
    result = render_chart(
        df=sample_df,
        chart_type="line",
        x_col="地区",
        y_col="销售额",
        output_path=str(output),
    )
    assert result["success"] is True
    assert output.exists()
    assert output.stat().st_size > 0


def test_render_chart_missing_x_col(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """X 列不存在: 返回错误"""
    output = tmp_path / "err.png"
    result = render_chart(
        df=sample_df,
        chart_type="bar",
        x_col="not_exist",
        y_col="销售额",
        output_path=str(output),
    )
    assert result["success"] is False
    assert "not_exist" in result["error"]
    assert not output.exists()


def test_render_chart_missing_y_col(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Y 列不存在: 返回错误"""
    output = tmp_path / "err.png"
    result = render_chart(
        df=sample_df,
        chart_type="bar",
        x_col="地区",
        y_col="not_exist",
        output_path=str(output),
    )
    assert result["success"] is False
    assert "not_exist" in result["error"]
    assert not output.exists()
