import textwrap

import pandas as pd
from sqlalchemy import create_engine

from .datasource import qualify, resolve_datasource


class StandardLoader:
    def __init__(self, db: str, workspace_uuid: str,
                       datasource_map: dict[str, str] | None = None) -> None:
        self.engine = create_engine(db)
        self.workspace_uuid = workspace_uuid
        self.datasource_map = datasource_map
    
    def load_standard_connects(self) -> pd.DataFrame:
        """从数据库加载数据标准关联的元数据列

        Returns:
            pd.DataFrame: 包含数据标准-列关系的DataFrame
        """
        sql = textwrap.dedent(f"""
                select
                    bis.code standard_id,
                    nv.name standard_name,
                    d2.name  database_name,
                    concat(d.name, ".", t.name) as "full_table_name",
                    c.name "column_name",
                    IFNULL(c.comment, c.name) as "name",
                    c.data_entity_type,
                    c.type as "dtype",
                    IFNULL(c.length,0) as "size",
                    IFNULL(c.`precision`, 0) as "precision",
                    IFNULL(c.scale , 0) as "scale"
                from
                    connector_foundation1.database_table_column c
                inner join connector_foundation1.database_table t
                on
                    c.database_table_id = t.id
                inner join connector_foundation1.`database` d
                on
                    t.database_id = d.id
                inner join connector_foundation1.datasource d2 
                on
                    d.service_id = d2.connection_uuid
                inner join standard_tdsgovernor1.standard_conn scn
                on
                    scn.database_name = d.name
                and scn.table_name = t.name
                and scn.column_name = c.name
                inner join navigator_foundation1.standard_basic bis
                on scn.standard_uuid = bis.uuid
                inner join navigator_foundation1.navigation nv
                on bis.uuid = nv.uuid
                where
                    scn.category= 'METADATA'
                    and
                    (
                    t.data_entity_type <> 'ORACLE_TABLE'
                        or 
                    (t.data_entity_type = 'ORACLE_TABLE'
                            and d.owner = d.name )
                    )
                    and t.is_deleted = 0
                    and c.is_deleted = 0
                    and d2.tenant = 'TDH'
                    and d2.workspace_uuid = '{self.workspace_uuid}'
                """)
        df_std_col = pd.read_sql(sql, self.engine, 
                                 dtype={"size":"int", 
                                        "precision":"int", 
                                        "scale":"int"})
        self._qualify_columns(df_std_col)
        return df_std_col

    def _qualify_columns(self, df_std_col: pd.DataFrame) -> None:
        """就地生成全限定 column 标识，并同步改写 full_table_name

        未提供 datasource_map 时保持 `schema.table[.column]` 原始标识。
        """
        ftn_list = []
        col_list = []
        for ftn, col_name in zip(df_std_col["full_table_name"], df_std_col["column_name"], strict=False):
            raw_col = f"{ftn}.{col_name}"
            if self.datasource_map is None:
                ftn_list.append(ftn)
                col_list.append(raw_col)
                continue
            schema = str(ftn).split(".", 1)[0]
            ds_name = resolve_datasource(schema, self.datasource_map)
            ftn_list.append(qualify(ds_name, str(ftn)))
            col_list.append(qualify(ds_name, raw_col))
        df_std_col["full_table_name"] = ftn_list
        df_std_col["column"] = col_list

    def load_standards(self) -> pd.DataFrame:
        """从数据库加载数据标准

        Returns:
            pd.DataFrame: 包含数据标准的DataFrame
        """

        sql = textwrap.dedent(f"""
                with std_value as (
                    SELECT 
                        std.uuid,
                        info.key_uuid,
                        bis.code,
                        nv.name,
                        bis.publish_status,
                        std.category, 
                        info.id, 
                        info.value,
                        ROW_NUMBER() OVER (PARTITION BY std.uuid ORDER BY info.id ASC) AS seq
                    FROM standard_tdsgovernor1.standard_info info
                    inner join navigator_foundation1.navigation nv
                    on info.standard_uuid = nv.uuid
                    inner join standard_tdsgovernor1.standard std
                    on info.standard_uuid = std.uuid
                    inner join navigator_foundation1.standard_basic bis
                    on info.standard_uuid = bis.uuid
                    where 
                    nv.workspace_uuid = '{self.workspace_uuid}'
                    and nv.node_type = 'FILE'
                    and info.is_deleted = 0
                    and std.is_deleted = 0
                    and bis.publish_status = 1
                )
                select
                    std_value.code standard_id,
                    std_value.name,
                    lower(substr(attr.code,7)) attrbute,
                    std_value.value
                from std_value
                inner join standard_tdsgovernor1.META_ATTR attr
                on std_value.key_uuid = attr.uuid
                order by 1,2
                """)
        df_standards_kv = pd.read_sql(sql, self.engine)
        return df_standards_kv.pivot(index=["standard_id", "name"], 
                                            columns="attrbute", 
                                            values="value") \
                                    .reset_index().rename(columns={"code": "ref_code_define"})

    @property
    def Standard(self):
        return self.load_standards()

    @property
    def StdCompliance(self):
        return self.load_standard_connects()
