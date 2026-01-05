"""
环境初始化, 获得层级结构的名称索引

生成的名称索引按应用存储在 `{name}_{app_name_en}.md` 的文件中
`{application_name}.md` 文件格式：

```markdown
# {full_table_name} {name}
- {column_name} {name}

```

---

输出样例：
assets
   └── names
        ├── app1.md
        ├── app2.md
        └── app3.md

app1.md
```markdown
# MDM.COLUMN_DEFINE 字段定义表
- ACC_YM 结算期间
- ADDRESS 地址
...
```
"""
import argparse
from pathlib import Path
from govio import FalkorDBGraph

ASSETS_NAMES_DIR = Path(__file__).parent.parent / "assets/names"

def generate_names(g:FalkorDBGraph):
    """生成按应用分组的名称索引文件"""

    if not ASSETS_NAMES_DIR.exists():
        ASSETS_NAMES_DIR.mkdir()
    
    # 查询所有应用
    apps_query = """
    MATCH (app:Application)
    RETURN app.app_name_en AS app_name_en, app.name AS name
    ORDER BY app.app_name_en
    """
    apps = g.query(apps_query)
    
    # 按应用逐次处理
    for app_row in apps:
        app_name_en, name = app_row
        
        # 查询该应用使用的所有物理表
        tables_query = """
        MATCH (app:Application {app_name_en: $app_name_en})-[:USE]->(table:PhysicalTable)
        RETURN table.full_table_name, table.name AS table_name
        ORDER BY table.full_table_name
        """
        tables = g.query(tables_query, {'app_name_en': app_name_en})
        
        md_content = []
        
        # 按物理表逐次处理
        for table_row in tables:
            full_table_name, table_name = table_row

            if not table_name or table_name == "None":
                table_name = ""
            
            md_content.append(f"# {full_table_name} {table_name}")
            
            # 查询该物理表的所有字段
            cols_query = """
            MATCH (table:PhysicalTable {full_table_name: $full_table_name})-[:HAS_COLUMN]->(col:Col)
            RETURN col.column_name, col.name AS col_name
            ORDER BY col.order_no
            """
            cols = g.query(cols_query, {'full_table_name': full_table_name})
            
            for col_row in cols:
                column_name, col_name = col_row
                if not col_name or col_name == "None":
                    col_name = ""
                md_content.append(f"- {column_name} {col_name}")
            
            md_content.append("")  # 空行分隔
        
        # 写入文件
        file_path = ASSETS_NAMES_DIR / f"{name}_{app_name_en}.md"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='图数据库名称', default="ontology")

    # 解析命令行参数
    args = parser.parse_args()
    g = FalkorDBGraph(graph = args.graph)

    generate_names(g)
