# 物理表关系JSON读取器设计

## 概述

创建一个读取 `schema_of_relationships.json` 文件的程序，将其补充到本体模型图数据生成中。

## 需求

### 输入
- `schema_of_relationships.json` 文件，定义物理表之间的字段关联关系
- 格式：
  ```json
  {
    "version": "1.0",
    "relationships": [
      {
        "description": "关系描述",
        "source": {
          "PhysicalTable": "表A",
          "Cols": ["列1"]
        },
        "target": {
          "PhysicalTable": "表B",
          "Cols": ["列B"]
        },
        "relationship_type": "many_to_one"
      }
    ]
  }
  ```

### 输出
- 生成 `RELATES_TO.csv` 文件
- 集成到现有的图数据生成流程中

## 设计方案

### 方案选择
**方案A：简单边类型**（已确认）

- 使用单一边类型 `RELATES_TO`
- 边属性包含 `relationship_type`, `description`, `source_columns`, `target_columns`
- 简单实用，与现有架构一致

### 模块结构

#### 1. 新增文件：`src/govio/metadata/relationship.py`

**类**：
- `RelationshipLoader`：读取和验证 JSON 文件

**函数**：
- `load_relationships(json_path: str, df_tables: DataFrame, df_columns: DataFrame) -> DataFrame`：生成 RELATES_TO 边数据

**验证逻辑**：
- 检查 JSON schema 是否符合格式
- 验证 source/target 表名是否存在于 PhysicalTable.csv
- 验证列名是否存在于 Col.csv
- 支持 single key 和 composite key（复合键）

#### 2. 修改文件：`src/govio/metadata/gen_networkx.py`

**修改点**：
- 在 `load_edges()` 函数的 `edge_files` 列表中添加 `"RELATES_TO.csv"`
- CSV 格式：
  ```
  :START_ID(PhysicalTable),:END_ID(PhysicalTable),relationship_type,description,source_columns,target_columns
  table1,table2,many_to_one,外键关联,"col1,col2",col_b
  ```

**边属性**：
- `relationship_type`: 关系类型
- `description`: 关系描述
- `source_columns`: 源表关联列（复合键用逗号分隔）
- `target_columns`: 目标表关联列

#### 3. 修改文件：`src/govio/metadata/utility.py`

**修改点**：
- 在 `make_csv()` 函数中添加参数 `relationship_file: str | None = None`
- 调用 `relationship.py` 的 `load_relationships()` 生成 RELATES_TO.csv
- 添加命令行参数 `--relationship` 支持 JSON 文件路径

**命令行用法**：
```bash
python -m govio.metadata.utility --kundb <db_url> --relationship <json_file> -o <output>
```

### 预定义关系类型
- `one_to_one`
- `one_to_many`
- `many_to_one`
- `many_to_many`

### 错误处理

**JSON 验证错误**：
- 缺少必需字段（source, target, relationship_type）
- 无效的 relationship_type 值
- 表名或列名不存在于现有数据中

**处理方式**：
- 记录警告日志并跳过无效关系
- 不中断整个处理流程
- 提供 `--strict` 参数可选择严格模式（遇到错误则中断）

**日志示例**：
```
WARNING: 表 'unknown_table' 不存在于 PhysicalTable.csv，跳过关系
WARNING: 列 'table1.col_x' 不存在于 Col.csv，跳过关系
```

### 数据流

1. 用户提供 `schema_of_relationships.json` 文件
2. `relationship.py` 验证并解析 JSON
3. 生成 `RELATES_TO.csv`（包含 source_columns 和 target_columns）
4. `gen_networkx.py` 加载到图中，边属性包含完整关系信息

## 文件清单

**新增文件**：
- `src/govio/metadata/relationship.py`

**修改文件**：
- `src/govio/metadata/gen_networkx.py`
- `src/govio/metadata/utility.py`

## 测试要点

1. 正常 JSON 文件加载和验证
2. 复合键（多列）支持
3. 表名或列名不存在的错误处理
4. 无效 relationship_type 的错误处理
5. 集成到现有 CSV 生成流程
6. 图生成流程中的边加载