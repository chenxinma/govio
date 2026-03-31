# 统一查询入口设计

## 背景

当前项目有两个独立的查询脚本：
- `query.py`: 使用 NetworkXGraph 执行 Python 代码查询
- `query_cypher.py`: 使用 FalkorDBGraph 执行 Cypher 查询

用户需要根据不同场景选择不同的脚本，希望合并为统一入口。

## 目标

- 提供统一的命令行入口
- 通过子命令区分不同图后端
- 保持现有功能不变

## 设计方案

### 命令格式

```bash
# NetworkX 模式
uv run query networkx --gml-path path/to/ontology.gml --code "result = [...]"

# FalkorDB 模式  
uv run query falkordb --graph-name ontology --host localhost --port 6379 --cypher "MATCH ..."
```

### 子命令设计

#### networkx 子命令

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gml-path` | str | `assets/ontology.gml` | GML 文件路径 |
| `--code` | str | 必填 | Python 代码字符串 |

#### falkordb 子命令

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--graph-name` | str | `ontology` | 图名称 |
| `--host` | str | `localhost` 或 `FALKORDB_HOST` 环境变量 | 数据库主机 |
| `--port` | int | `6379` 或 `FALKORDB_PORT` 环境变量 | 数据库端口 |
| `--cypher` | str | 必填 | Cypher 查询语句 |

### 共同逻辑

1. **结果输出**
   - ≤10 条结果：打印 JSON 到控制台
   - >10 条结果：保存到 `assets/output-{timestamp}.json` 文件

2. **日志记录**
   - 日志文件：`logs/query_{YYYYMMDD}.log`
   - 记录查询语句和结果数量

3. **Schema 检查**
   - NetworkX 模式：检查 assets 目录和 GML 文件是否存在
   - 不存在时提示用户先运行初始化脚本

### 文件变更

- 合并 `skills/govio/scripts/query.py` 和 `skills/govio/scripts/query_cypher.py`
- 新的 `query.py` 使用 argparse subcommands 实现
- 保留 `query_cypher.py` 作为备份（后续可删除）

## 实现要点

1. 使用 `argparse.add_subparsers()` 创建子命令
2. 为每个子命令设置独立的参数解析器
3. 根据子命令调用对应的图后端
4. 统一结果处理和日志记录逻辑
