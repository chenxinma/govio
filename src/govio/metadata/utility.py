import argparse
import os
from pathlib import Path
import sys
import textwrap
from dotenv import load_dotenv
import pandas as pd

from .application import AppInfoLoader
from .database import DatabseLoader


def reorder_index(dfs: list[pd.DataFrame]):
    base_index:int = 1

    for df in dfs:
        _end_index = base_index + df.shape[0]
        df["index"] = [ i for i in range(base_index, _end_index)]
        df.set_index("index", drop=True, inplace=True)
        base_index = _end_index


def make_metadata_csv():
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

    db_loader = DatabseLoader(db, workspace_uuid, df_app_db_map["schema"].to_list())
    app_loader = AppInfoLoader(app_list, df_app_db_map["name"].to_list())

    df_tables = db_loader.PhysicalTable
    df_columns = db_loader.Col
    df_apps = app_loader.Application

    reorder_index([df_tables, df_columns, df_apps])

    files = []

    df_tables.to_csv(output / "PhysicalTable.csv", index_label=":ID(PhysicalTable)")
    files.append("-n " + str(output/ "PhysicalTable.csv"))

    df_columns.to_csv(output / "Col.csv", index_label=":ID(Col)")
    files.append("-n " + str(output/ "Col.csv"))

    df_apps.to_csv(output / "Application.csv", index_label=":ID(Application)")
    files.append("-n " + str(output/ "Application.csv"))

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
    