# observe chart 子命令实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `observe` CLI 下新增 `chart` 子命令,从已加载的 DataFrame 生成 PNG 柱状图/折线图。

**Architecture:** 新增 `observe_data/core/chart.py` 模块封装 matplotlib 画图逻辑(含中文字体回退链 + 负号修复),`cli/observe.py` 新增 `chart` 子命令分发。复用 `ObserveStore` 取 DataFrame,与 `compare`/`explore` 命令族风格一致。

**Tech Stack:** Python 3.13+, matplotlib (新增依赖), pandas, argparse

---

## 文件结构

- **新增** `src/govio/observe_data/core/chart.py` — 画图核心模块,含字体设置和 `render_chart` 函数
- **修改** `src/govio/cli/observe.py` — 新增 `cmd_chart` 函数和 `chart` 子命令注册
- **新增** `tests/observe_data/test_chart.py` — 测试 `render_chart`
- **修改** `pyproject.toml` — 加 `matplotlib>=3.8` 主依赖

---

## Task 1: 加 matplotlib 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 编辑 pyproject.toml 加依赖**

在 `dependencies` 列表中,在 `"pandas>=2.3.3",` 后面加一行:

```toml
    "matplotlib>=3.8",
```

- [ ] **Step 2: 同步依赖**

Run: `uv sync`
Expected: 成功安装 matplotlib

- [ ] **Step 3: 验证 import**

Run: `uv run python -c "import matplotlib; print(matplotlib.__version__)"`
Expected: 打印版本号,无报错

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add matplotlib for observe chart"
```

---

## Task 2: 写 render_chart 的失败测试

**Files:**
- Create: `tests/observe_data/test_chart.py`
- Create: `tests/observe_data/__init__.py` (空文件)

- [ ] **Step 1: 创建 tests/observe_data/ 目录和 __init__.py**

```bash
mkdir -p tests/observe_data
touch tests/observe_data/__init__.py
```

- [ ] **Step 2: 写测试文件**

创建 `tests/observe_data/test_chart.py`:

```python
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
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/observe_data/test_chart.py -v`
Expected: 4 个测试全部 FAIL,报错 `ModuleNotFoundError: No module named 'govio.observe_data.core.chart'`

- [ ] **Step 4: Commit**

```bash
git add tests/observe_data/__init__.py tests/observe_data/test_chart.py
git commit -m "test: add failing tests for observe chart"
```

---

## Task 3: 实现 render_chart

**Files:**
- Create: `src/govio/observe_data/core/chart.py`

- [ ] **Step 1: 创建 chart.py**

创建 `src/govio/observe_data/core/chart.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认通过**

Run: `uv run pytest tests/observe_data/test_chart.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 3: Commit**

```bash
git add src/govio/observe_data/core/chart.py
git commit -m "feat(observe): add chart rendering module"
```

---

## Task 4: 接入 observe CLI

**Files:**
- Modify: `src/govio/cli/observe.py`

- [ ] **Step 1: 在 observe.py 顶部 import 处加 render_chart 引用**

在 `from ..observe_data.tools.visualize_relations import visualize_relations` 这一行后面加:

```python
from ..observe_data.core.chart import render_chart
```

- [ ] **Step 2: 在 cmd_visualize 函数后面加 cmd_chart 函数**

在 `cmd_visualize` 函数定义之后(在 `def observe():` 之前)插入:

```python
def cmd_chart(
    config: dict,
    name: str,
    chart_type: str,
    x: str,
    y: str,
    output: str,
) -> None:
    """生成图表 PNG"""
    store = ObserveStore()
    df = store.get(name)
    if df is None:
        print(json.dumps({"success": False, "error": f"DataFrame '{name}' 不存在"}))
        return

    result = render_chart(
        df=df,
        chart_type=chart_type,
        x_col=x,
        y_col=y,
        output_path=output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **Step 3: 在 argparse 子命令注册区加 chart**

在 `# visualize-relations --relations <json>` 块之后(在 `args = parser.parse_args(sys.argv[1:])` 之前)插入:

```python
    # chart --name --type --x --y -o
    p = sub.add_parser("chart", help="从 DataFrame 生成图表 PNG")
    p.add_argument("--name", required=True, help="DataFrame 名称")
    p.add_argument("--type", required=True, choices=["bar", "line"], help="图表类型")
    p.add_argument("--x", required=True, help="X 轴列名")
    p.add_argument("--y", required=True, help="Y 轴列名")
    p.add_argument("-o", "--output", required=True, help="输出 PNG 路径")
```

- [ ] **Step 4: 在 match-case 分发区加 chart 分支**

在 `case "visualize-relations":` 之后加:

```python
        case "chart":
            cmd_chart(config, args.name, args.type, args.x, args.y, args.output)
```

- [ ] **Step 5: 验证 CLI 注册**

Run: `uv run govio-cli observe chart --help`
Expected: 打印 chart 子命令帮助,显示 `--name`、`--type`、`--x`、`--y`、`-o/--output` 参数

- [ ] **Step 6: 端到端冒烟测试**

准备一个测试 DataFrame 并画图(用 python 直接调):

Run:
```bash
uv run python -c "
import pandas as pd
from govio.observe_data.core.observe_store import ObserveStore
df = pd.DataFrame({'地区': ['华北', '华东', '华南'], '销售额': [100, 250, 180]})
store = ObserveStore()
store.store(name='smoke_test', df=df, datasource='memory', sql='manual')
" && uv run govio-cli observe chart --name smoke_test --type bar --x 地区 --y 销售额 -o /tmp/smoke_bar.png && ls -la /tmp/smoke_bar.png
```
Expected: 输出 `{"success": true, "output": "/tmp/smoke_bar.png"}` 且文件存在、大小 > 0

- [ ] **Step 7: 清理冒烟测试数据**

Run: `uv run govio-cli observe release --name smoke_test`
Expected: `{"success": true, ...}`

- [ ] **Step 8: Commit**

```bash
git add src/govio/cli/observe.py
git commit -m "feat(observe): add chart subcommand to observe CLI"
```

---

## Task 5: 回归测试

**Files:** 无新增

- [ ] **Step 1: 跑全部测试**

Run: `uv run pytest tests/ -v`
Expected: 全部 PASS(含原有测试 + 新增 4 个 chart 测试)

- [ ] **Step 2: 如有失败,修复后重新跑**

如失败,定位是 chart 模块影响还是其他回归,修复后重跑直到全绿。
