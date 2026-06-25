"""DataFrame 图表生成"""

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # 非交互后端,无 DISPLAY 也能出图

import matplotlib.pyplot as plt


_FONT_FALLBACK = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
]

_font_configured = False


def _setup_font() -> None:
    """配置中文字体回退链和负号显示,模块级只执行一次"""
    global _font_configured
    if _font_configured:
        return
    plt.rcParams["font.sans-serif"] = _FONT_FALLBACK
    plt.rcParams["axes.unicode_minus"] = False
    _font_configured = True


def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    output_path: str,
) -> dict[str, Any]:
    """渲染图表到 PNG

    Args:
        df: 数据源
        chart_type: "bar" 或 "line"
        x_col: X 轴列名
        y_col: Y 轴列名
        output_path: 输出 PNG 路径

    Returns:
        {"success": bool, "error": str (失败时), "output": str (成功时)}
    """
    if x_col not in df.columns:
        return {"success": False, "error": f"列 '{x_col}' 不存在"}
    if y_col not in df.columns:
        return {"success": False, "error": f"列 '{y_col}' 不存在"}

    _setup_font()

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "bar":
        ax.bar(df[x_col], df[y_col])
    elif chart_type == "line":
        ax.plot(df[x_col], df[y_col], marker="o")
    else:
        plt.close(fig)
        return {"success": False, "error": f"不支持的图表类型: '{chart_type}'"}

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return {"success": True, "output": str(output.resolve())}
