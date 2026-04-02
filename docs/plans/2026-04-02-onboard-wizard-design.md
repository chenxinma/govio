# Govio Onboard 向导设计文档

**日期**: 2026-04-02  
**状态**: 已批准

## 概述

将 `./skills/govio/scripts/load*.py` 的实现内化到 govio 模块本身，创建一个 `govio onboard` 向导命令，用于首次安装时通过命令行确定 graph 使用 FalkorDB 还是 NetworkX，并在 `skills/govio/assets` 下创建合适的 schema.md 和 names。

## 需求

1. **运行时机**: 首次安装后手动运行（如 `govio onboard` 命令）
2. **配置存储**: 全局配置文件（`~/.govio/config.yaml`）
3. **功能范围**: 全面向导（选择 graph backend + 创建配置 + 生成 schema.md + 生成 names 文件）
4. **NetworkX 数据源**: 从 CSV 文件生成 GML
5. **输出位置**: 固定路径 `skills/govio/assets`

## 整体架构

```
govio onboard 向导架构
├── CLI 层 (src/govio/cli/)
│   ├── __init__.py
│   ├── onboard.py          # 主向导命令
│   └── config.py           # 配置管理
│
├── 核心功能层 (src/govio/core/)
│   ├── __init__.py
│   ├── assets_generator.py # 资产生成器（schema + names）
│   └── graph_factory.py    # 图对象工厂
│
├── 配置层
│   └── ~/.govio/
│       └── config.yaml     # 全局配置文件
│
└── 输出层
    └── skills/govio/assets/
        ├── schema.md
        ├── ontology.gml (NetworkX)
        └── names/
            └── *.md
```

**关键设计决策:**
1. 提取 load 脚本逻辑到 `core/assets_generator.py`
2. 使用工厂模式创建不同 backend 的图对象
3. 配置使用 YAML 格式，存储在 `~/.govio/config.yaml`

## 核心组件设计

### 1. CLI 命令 (`govio onboard`)

```python
# src/govio/cli/onboard.py

def onboard():
    """Onboard向导主函数"""
    # 1. 欢迎信息
    # 2. 选择图数据库后端 (networkx/falkordb)
    # 3. 根据选择收集配置信息
    # 4. 保存配置到 ~/.govio/config.yaml
    # 5. 生成 assets (schema.md + names/)
    # 6. 显示完成信息
```

### 2. 配置管理

```yaml
# ~/.govio/config.yaml 示例
backend: networkx  # 或 falkordb

# NetworkX 配置
networkx:
  gml_path: skills/govio/assets/ontology.gml

# FalkorDB 配置
falkordb:
  host: localhost
  port: 6379
  graph: ontology

# Assets 输出路径
assets:
  output_dir: skills/govio/assets
```

### 3. 资产生成器 (`core/assets_generator.py`)

```python
class AssetsGenerator:
    """统一的资产生成器"""
    
    def __init__(self, graph, output_dir: Path):
        self.graph = graph
        self.output_dir = output_dir
    
    def generate_schema(self):
        """生成 schema.md"""
        
    def generate_names(self):
        """生成 names/ 目录"""
        
    def generate_all(self):
        """生成所有资产"""
        self.generate_schema()
        self.generate_names()
```

### 4. 图对象工厂 (`core/graph_factory.py`)

```python
class GraphFactory:
    """创建图对象的工厂"""
    
    @staticmethod
    def create(config: dict):
        """根据配置创建图对象"""
        backend = config['backend']
        if backend == 'networkx':
            return NetworkXGraph(config['networkx']['gml_path'])
        elif backend == 'falkordb':
            return FalkorDBGraph(**config['falkordb'])
```

### 5. Onboard 向导流程

**NetworkX 流程:**
```
1. 询问是否需要生成新的 GML 文件？
   - 是 → 输入 CSV 目录路径 → 调用 gen_networkx.build_graph() 生成 GML
   - 否 → 输入已有 GML 文件路径
2. 保存配置
3. 加载图对象
4. 生成 schema.md
5. 生成 names/
```

**FalkorDB 流程:**
```
1. 输入 FalkorDB 连接信息（host, port, graph）
2. 保存配置
3. 连接图数据库
4. 生成 schema.md
5. 生成 names/
```

### 6. 核心组件调用关系

```python
# src/govio/cli/onboard.py

def onboard():
    config_manager = ConfigManager()
    
    # 选择 backend
    backend = prompt_backend_choice()
    
    if backend == 'networkx':
        # 询问是否生成 GML
        if prompt_generate_gml():
            csv_dir = prompt_csv_dir()
            output_gml = SKILLS_ASSETS_DIR / "ontology.gml"
            build_graph(csv_dir, output_gml)  # 复用 gen_networkx
            gml_path = output_gml
        else:
            gml_path = prompt_gml_path()
        
        config = {
            'backend': 'networkx',
            'networkx': {'gml_path': gml_path}
        }
    else:  # falkordb
        host, port, graph = prompt_falkordb_config()
        config = {
            'backend': 'falkordb',
            'falkordb': {'host': host, 'port': port, 'graph': graph}
        }
    
    # 保存配置
    config_manager.save(config)
    
    # 生成 assets
    graph_obj = GraphFactory.create(config)
    generator = AssetsGenerator(graph_obj, SKILLS_ASSETS_DIR)
    generator.generate_all()
```

## 数据流设计

### NetworkX 数据流

```
用户输入 CSV 目录
    ↓
gen_networkx.build_graph(csv_dir, gml_path)
    ↓
生成 ontology.gml (skills/govio/assets/ontology.gml)
    ↓
NetworkXGraph(gml_path)
    ↓
AssetsGenerator.generate_all()
    ↓
生成：
  - schema.md (节点类型、边类型、关系定义)
  - names/node_names.md (所有节点名称索引)
```

### FalkorDB 数据流

```
用户输入连接信息（host, port, graph）
    ↓
FalkorDBGraph(host, port, graph)
    ↓
AssetsGenerator.generate_all()
    ↓
生成：
  - schema.md (节点属性、关系模式)
  - names/{name}_{app_name_en}.md (按应用分组)
```

### 配置文件生命周期

```
首次运行 onboard
    ↓
检测 ~/.govio/config.yaml 是否存在
    ↓ (不存在)
创建配置文件
    ↓
后续使用时读取配置
    ↓
如需修改，重新运行 onboard (会覆盖配置)
```

### Assets 输出结构

```
skills/govio/assets/
├── schema.md              # 图结构定义
├── ontology.gml           # NetworkX 图数据文件（仅在 NetworkX 模式）
└── names/
    ├── node_names.md      # NetworkX: 所有节点名称
    └── {name}_{app}.md    # FalkorDB: 按应用分组的名称
```

## 错误处理与验证

### 输入验证

```python
# CSV 目录验证
- 检查目录是否存在
- 检查是否包含必需的 CSV 文件
- 验证 CSV 文件格式（第一列符合 :ID(NodeType) 格式）

# GML 文件验证
- 检查文件是否存在
- 尝试加载 GML，捕获格式错误

# FalkorDB 连接验证
- 尝试连接数据库
- 验证指定的 graph 是否存在
- 捕获连接超时、认证失败等错误
```

### 错误处理策略

```python
class OnboardError(Exception):
    """Onboard 相关错误基类"""

class CSVValidationError(OnboardError):
    """CSV 文件验证失败"""

class GraphConnectionError(OnboardError):
    """图数据库连接失败"""

class ConfigSaveError(OnboardError):
    """配置保存失败"""
```

### 边界情况处理

1. **配置文件已存在**
   - 提示用户是否覆盖
   - 显示当前配置内容

2. **assets 目录已存在**
   - 提示用户是否重新生成
   - 备份旧文件（可选）

3. **CSV 目录缺少部分文件**
   - 显示缺失文件列表
   - 询问是否继续（某些 CSV 可选）

4. **GML 文件格式错误**
   - 显示错误详情
   - 建议重新生成 GML

5. **FalkorDB 连接失败**
   - 显示错误原因
   - 允许重试或修改配置

### 友好的错误提示

```python
# 示例：CSV 目录验证失败
"❌ CSV 目录验证失败：
  - 缺少必需文件：PhysicalTable.csv
  - Col.csv 格式错误：第一列应为 ':ID(Col)'
  
请检查 CSV 文件格式是否符合要求。"
```

## 测试策略

### 单元测试

```python
# tests/test_assets_generator.py
- test_generate_schema_networkx()
- test_generate_schema_falkordb()
- test_generate_names_networkx()
- test_generate_names_falkordb()

# tests/test_config_manager.py
- test_save_config()
- test_load_config()
- test_config_validation()

# tests/test_graph_factory.py
- test_create_networkx_graph()
- test_create_falkordb_graph()
```

### 集成测试

```python
# tests/test_onboard.py
- test_onboard_networkx_with_csv()
- test_onboard_networkx_with_existing_gml()
- test_onboard_falkordb()
- test_onboard_overwrite_config()
```

### 测试数据

```
tests/fixtures/
├── sample_csv/
│   ├── PhysicalTable.csv
│   ├── Col.csv
│   ├── Application.csv
│   ├── Standard.csv
│   ├── HAS_COLUMN.csv
│   ├── USE.csv
│   └── COMPLIES_WITH.csv
└── sample_gml/
    └── test_ontology.gml
```

## 实现计划

### 实现步骤

1. **创建配置管理模块**
   - `src/govio/cli/config.py`
   - 配置文件的读写、验证

2. **提取资产生成逻辑**
   - `src/govio/core/assets_generator.py`
   - 从 `load_*.py` 脚本提取逻辑

3. **创建图对象工厂**
   - `src/govio/core/graph_factory.py`

4. **实现 onboard CLI 命令**
   - `src/govio/cli/onboard.py`
   - 交互式向导逻辑

5. **注册 CLI 入口**
   - 在 `pyproject.toml` 添加 `onboard` 命令

6. **编写测试**
   - 单元测试和集成测试

7. **更新文档**
   - README 中添加 onboard 使用说明

### 文件变更清单

```
新增文件：
- src/govio/cli/__init__.py
- src/govio/cli/onboard.py
- src/govio/cli/config.py
- src/govio/core/__init__.py
- src/govio/core/assets_generator.py
- src/govio/core/graph_factory.py
- tests/test_onboard.py
- tests/test_assets_generator.py

修改文件：
- pyproject.toml (添加 onboard 入口点)
- src/govio/__init__.py (导出新模块)

保留文件：
- skills/govio/scripts/load_*.py (兼容性考虑)
```

## 依赖

### 现有依赖
- `networkx>=3.6.1` - NetworkX 图库
- `falkordb>=1.4.0` - FalkorDB 客户端
- `pandas>=2.3.3` - CSV 数据处理
- `pyyaml` - YAML 配置文件（需要添加）

### 新增依赖
- `pyyaml` - 用于读写 YAML 配置文件

## 风险与缓解

### 风险 1: 配置文件路径权限问题
**缓解**: 使用 `pathlib.Path.home()` 获取用户目录，确保跨平台兼容

### 风险 2: CSV 文件格式不一致
**缓解**: 提供详细的错误提示，指导用户修正格式

### 风险 3: FalkorDB 连接超时
**缓解**: 设置合理的超时时间，提供重试机制

## 后续优化

1. **配置加密**: 对敏感信息（如数据库密码）加密存储
2. **多环境支持**: 支持多个环境配置（dev/staging/prod）
3. **自动更新检测**: 检测 assets 是否需要更新
4. **配置验证命令**: 添加 `govio config validate` 命令
