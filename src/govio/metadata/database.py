from pathlib import Path
import textwrap
import pandas as pd
from sqlalchemy import create_engine

class DatabaseLoader:
    def __init__(self, db: str, workspace_uuid: str, schema_limits: list[str] | None = None) -> None:
        self.engine = create_engine(db)
        self.workspace_uuid = workspace_uuid
        if schema_limits:
            self.schema_str = "'" +  "','".join(schema_limits) + "'"
        else:
            self.schema_str = None
    
    def _convert_data_type(self, row: pd.Series) -> str:
        """根据数据库类型和字段属性转换数据类型
        
        针对Oracle列类型进行特殊处理，将其转换为标准SQL数据类型格式
        
        Args:
            row: 包含列信息的Pandas Series
        
        Returns:
            str: 转换后的标准数据类型字符串
        """
        # 非Oracle列直接返回原始类型
        if "ORACLE_COLUMN" != row['data_entity_type']:
            return row['dtype']  # pyright: ignore[reportReturnType]
        
        _dtype = row['dtype']
        # 处理字符串类型
        if _dtype in ['NVARCHAR2', 'VARCHAR2', 'VARCHAR']:
            return f"varchar({row['size']})"
        if _dtype == 'CHAR': # type: ignore
            return f"char({row['size']})"
        
        # 处理数值类型
        if _dtype == 'NUMBER': # type: ignore
            if row['scale'] > 0: # type: ignore
                return f"decimal({row['precision']}, {row['scale']})"
            if row['precision'] == 0: # type: ignore
                return f"decimal(38,20)"
            return f"decimal({row['precision']})"
        
        # 其他类型转为小写
        return str(_dtype).lower()

    def load_columns(self) -> pd.DataFrame:
        """从数据库加载列元数据
        
        Returns:
            pd.DataFrame: 包含列信息、数据类型的DataFrame
        """
        sql = textwrap.dedent(f"""
            select
                concat(d.name, ".", t.name, ".", c.name) as "column",
                c.name as "column_name",
                c.comment as "name",
                concat(d.name, ".", t.name) as "full_table_name",
                c.data_entity_type,
                c.type as "dtype",
                c.length as "size",
                c.`precision`,
                c.scale ,
                c.`order` as "order_no"
            from connector_foundation1.database_table_column c
            inner join connector_foundation1.database_table t
            on c.database_table_id = t.id
            inner join connector_foundation1.`database` d
            on t.database_id = d.id
            inner join connector_foundation1.datasource d2 
            on	d.service_id = d2.connection_uuid
            where 
                d.name in ({self.schema_str})
            and (
                t.data_entity_type <> 'ORACLE_TABLE' or 
                (t.data_entity_type = 'ORACLE_TABLE' and d.owner=d.name )
                )
            and t.is_deleted = 0
            and c.is_deleted = 0
            and d2.tenant = 'TDH'
            and d2.workspace_uuid ='{self.workspace_uuid}'
            """)
        
        df_columns = pd.read_sql(sql, self.engine).fillna(0) \
                        .astype(dtype={'size': 'int', 'precision': 'int', 'scale': 'int', 'order_no': 'int'})
        # 转换数据类型
        df_columns['data_type'] = df_columns.apply(self._convert_data_type, axis=1)
        
        return df_columns
    
    def load_tables(self) -> pd.DataFrame:
        """从数据库加载表元数据

        Returns:
            pd.DataFrame: 包含表信息的DataFrame
        """
        sql = textwrap.dedent(f"""
                select 
                    concat(d.name, ".", t.name) full_table_name,
                    d.name "schema",
                    t.name table_name,
                    t.comment name,
                    t.data_entity_type,
                    d2.name database_name
                from connector_foundation1.database_table t
                inner join connector_foundation1.`database` d
                on t.database_id = d.id
                inner join connector_foundation1.datasource d2 
                on	d.service_id = d2.connection_uuid
                where 
                    d.name in ({self.schema_str})
                and (
                    t.data_entity_type <> 'ORACLE_TABLE' or 
                    (t.data_entity_type = 'ORACLE_TABLE' and d.owner=d.name )
                    )
                and t.is_deleted = 0
                and d2.tenant = 'TDH'
                and d2.workspace_uuid ='{self.workspace_uuid}'
            """)
        df_tables = pd.read_sql(sql, self.engine)
        return df_tables
    
    @property
    def PhysicalTable(self):
        return self.load_tables()

    @property
    def Col(self):
        return self.load_columns()
