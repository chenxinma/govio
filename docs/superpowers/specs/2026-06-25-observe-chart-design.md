# observe chart 子命令设计

日期: 2026-06-25
状态: 已确认

## 目标

在 `observe` CLI 下新增 `chart` 子命令,从已加载的 DataFrame 生成 PNG 图表。第一版用最简单的方案打通链路。

## 范围

### 支持

- 柱状图 (`bar`) 和折线图 (`line`) 两种类型
- 单系列,不支持分组
- 从 ObserveStore 取已加载的 DataFrame
- 输出 PNG 到指定路径

### 不支持 (YAGNI)

- 分组 / 多系列
- 直方图、散点图、饼图等其他图表类型
- 配置文件覆盖字体
- 字体缺失告警
- 从外部 CSV/JSON 文件读 DataFrame

## CLI 接口

```
govio observe chart --name <df_name> --type {bar|line} --x <col> --y <col> -o <output.png>
```

参数全部必填:

- `--name`: ObserveStore 中已加载的 DataFrame 名称
- `--type`: `bar` 或 `line`
- `--x`: X 轴列名 (分类轴 / 时序轴)
- `--y`: Y 轴列名 (数值轴,单列)
- `-o/--output`: 输出 PNG 路径

### 行为

1. 从 `ObserveStore` 取 DataFrame;不存在则输出 `{"success": false, "error": "DataFrame '...' 不存在"}`
2. 校验 `--x` / `--y` 列存在;不存在则返回错误 JSON
3. 调用 matplotlib 画图并 `savefig` 到指定路径
4. 成功时 stdout 打印 `{"success": true, "output": "<abs_path>"}`,保持 observe 命令族 JSON 输出风格

## 模块结构

### 新增 `src/govio/observe_data/core/chart.py`

- `_setup_font()`: 模块级调用一次,设 `rcParams['font.sans-serif']` 为回退链,设 `rcParams['axes.unicode_minus'] = False`
- `render_chart(df, chart_type, x_col, y_col, output_path) -> dict`: 校验列、按类型调 matplotlib、`savefig`,返回结果字典

### 字体回退链

```python
['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
```

matplotlib 会自动找系统里第一个可用的;都找不到时仍会出图但中文显示为豆腐块,不在第一版处理。

### 负号显示

`rcParams['axes.unicode_minus'] = False`,用 ASCII 减号代替,避免中文字体下负号渲染为方块。

## CLI 接入

`src/govio/cli/observe.py`:

- 新增 `cmd_chart(config, name, chart_type, x, y, output)` 函数,模式与 `cmd_compare` / `cmd_explore` 一致
- `observe()` 的 argparse 注册 `chart` 子命令
- match-case 分发新增一支 `case "chart"`

## 依赖

`pyproject.toml` 主依赖加 `matplotlib>=3.8`。

## 测试

`tests/observe_data/test_chart.py`:

- 用一个小 DataFrame 测 `bar` / `line` 两种类型
- 校验文件生成、返回 `success: true`
- 不校验图像内容
- 校验列不存在时返回错误

## 错误处理

- DataFrame 不存在: `{"success": false, "error": "DataFrame '<name>' 不存在"}`
- 列不存在: `{"success": false, "error": "列 '<col>' 不存在"}`
- 图表类型非法: argparse `choices=['bar', 'line']` 在解析阶段拒绝
