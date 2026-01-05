import argparse
from enum import Enum
import os
from pathlib import Path
import sys
import textwrap
from dotenv import load_dotenv
import pandas as pd

from .application import AppInfoLoader
from .database import DatabseLoader
from .standard import StandardLoader
from .recommender import create_recommender, StandardRecommender

app_db_map_data = [
        ['ihrodb', '外包项目管理系统'],
        ['IHRO_BILL', '外包项目管理系统'],
        ['ihrmdb', '外包雇员管理系统'],
        ['fsghr', '人力资源服务订单'],
        ['pdm', '产品价格中心'],
        ['sqc', '报价单中心'],
        # ['fsgcontract', '集团业务合同管理系统（合同中心）'],
        ['SSOP_USER', '企业法定福利服务系统'],
        ['podb', '产品订单系统'],
        ['CDPS_USER', '财务管理平台收付费管理'],
        ['ITS_USER', '财务管理平台发票管理'],
        ['MDM', '客户及供应商中心主数据系统'],
        ['ioms', '外服内部机构管理系统'],
        ['hps_health', '健康管理生产系统'],
        ['paypro', '薪税生产系统'],
        ['AEP_USER', '会计引擎'],
        ['sprtdb', '销售及订单管理平台（销售门户）'],
        ['BILL_USER', '客户账单管理'],
        ['NHRS_USER', '人事服务订单（调派订单）'],
    ]
df_app_db_map = pd.DataFrame(app_db_map_data, columns=['schema', 'name'])

class Mode(Enum):
    CSV = "csv"
    RECOMMEND = "recommend"

    def __str__(self):
        return self.value


def reorder_index(dfs: list[pd.DataFrame]):
    base_index:int = 1

    for df in dfs:
        _end_index = base_index + df.shape[0]
        df["index"] = [ i for i in range(base_index, _end_index)]
        df.set_index("index", drop=True, inplace=True)
        base_index = _end_index

def run():
    """
    1.从数据治理平台和应用清单获取基础元数据生成CSV
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f'''\
        从元数据管理的数据库中提取元数据信息生成用于沟通图数据库的csv。
        从应用清单中获得应用信息生成用于沟通图数据库的csv。
        ''')
    )
    parser.add_argument('--kundb', type=str, help='元数据库URL')
    parser.add_argument('--app-list', type=str, help="应用清单")
    parser.add_argument('-m', "--mode", type=Mode, choices=list(Mode), default=Mode.CSV)
    parser.add_argument('-o', '--output', type=str, default=".", help="输出目录")
    # 解析命令行参数
    args = parser.parse_args()

    load_dotenv()
    db = os.getenv("KUNDB_URL", "")
    if args.kundb:
        db = args.kundb
    
    app_list = os.getenv("APP_LIST_FILE", "")
    if args.app_list:
        app_list = args.app_list
    
    workspace_uuid = '82ee37374b314a938bf28170ab4db7cf'

    if len(db) == 0:
        print("元数据管理库未设置")
        sys.exit()
    
    if not os.path.exists(args.output):
        print("输出目录未找到")
        sys.exit()
    
    output = Path(args.output)
    
    if args.mode == Mode.CSV:
        make_csv(output, db, workspace_uuid, app_list)
    elif args.mode == Mode.RECOMMEND:
        data_standard_recommend(output, db, workspace_uuid)

    
def make_csv(output:Path, db:str, workspace_uuid:str, app_list_file: str):
    db_loader = DatabseLoader(db, workspace_uuid, df_app_db_map["schema"].to_list())
    app_loader = AppInfoLoader(app_list_file, df_app_db_map["name"].to_list())
    std_loader = StandardLoader(db, workspace_uuid)

    df_tables = db_loader.PhysicalTable
    df_columns = db_loader.Col
    df_apps = app_loader.Application
    df_stds = std_loader.Standard

    reorder_index([df_tables, df_columns, df_apps, df_stds])

    files = []

    df_tables.to_csv(output / "PhysicalTable.csv", index_label=":ID(PhysicalTable)")
    files.append("-n " + str(output/ "PhysicalTable.csv"))

    df_columns.to_csv(output / "Col.csv", index_label=":ID(Col)")
    files.append("-n " + str(output/ "Col.csv"))

    df_apps.to_csv(output / "Application.csv", index_label=":ID(Application)")
    files.append("-n " + str(output/ "Application.csv"))

    df_stds.to_csv(output / "Standard.csv", index_label=":ID(Standard)")
    files.append("-n " + str(output/ "Standard.csv"))

    df_has_column = pd.merge(
        df_tables[["full_table_name"]].reset_index().rename(columns={"index":":START_ID(PhysicalTable)"}), 
        df_columns[["full_table_name"]].reset_index().rename(columns={"index":":END_ID(Col)"}), 
        on="full_table_name", 
        how="inner") [[":START_ID(PhysicalTable)", ":END_ID(Col)"]]
    df_has_column.to_csv(output / "HAS_COLUMN.csv", index=False)
    files.append("-r " + str(output/ "HAS_COLUMN.csv"))

    df_app_table = pd.merge(df_app_db_map, 
                      df_tables[["schema"]].reset_index().rename(columns={"index": ":END_ID(PhysicalTable)"}), 
                    on="schema", how="inner")
    df_use = pd.merge(df_apps[["name"]].reset_index().rename(columns={"index": ":START_ID(Applicatin)"}),
                      df_app_table,
                      on="name", how="inner")[[":START_ID(Applicatin)", ":END_ID(PhysicalTable)"]]
    
    df_use.to_csv(output / "USE.csv", index=False)
    files.append("-r " + str(output/ "USE.csv"))

    s = f"falkordb-bulk-insert {{GRAPH}} {"  ".join(files)}"
    print("Bulk insert usage:")
    print(s)
    
def data_standard_recommend(output:Path, db:str, workspace_uuid:str):
    db_loader = DatabseLoader(db, workspace_uuid, ['ihrodb', 'ihrmdb', 'MDM'])
    std_loader = StandardLoader(db, workspace_uuid)
    # 加载数据
    std_compliance = std_loader.StdCompliance  # 已贯标列
    all_columns = db_loader.Col  # 所有列

    # 创建推荐器
    recommender = create_recommender(
        std_compliance=std_compliance,
        k_neighbors=5,  # 使用5个最近邻
        top_n=3  # 返回Top 3推荐
    )

    # 批量推荐
    recommendations = recommender.batch_recommend(all_columns)

    # 保存结果
    recommendations.to_csv(output / 'recommendations.csv', index=False)
