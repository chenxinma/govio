"""DataFrame 内存存储管理"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataFrameInfo:
    """DataFrame 信息"""

    name: str
    rows: int
    columns: int
    column_info: list[dict] = field(default_factory=list)


class DataFrameStore:
    """DataFrame 内存存储"""

    _instance = None
    _dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __new__(cls) -> "DataFrameStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dataframes = {}
        return cls._instance

    def store(self, name: str, df: pd.DataFrame) -> DataFrameInfo:
        """存储 DataFrame"""
        self._dataframes[name] = df

        column_info = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]

        return DataFrameInfo(
            name=name, rows=len(df), columns=len(df.columns), column_info=column_info
        )

    def get(self, name: str) -> pd.DataFrame | None:
        """获取 DataFrame"""
        return self._dataframes.get(name)

    def list(self) -> list[DataFrameInfo]:
        """列出所有 DataFrame"""
        infos = []
        for name, df in self._dataframes.items():
            column_info = [
                {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
            ]
            infos.append(
                DataFrameInfo(
                    name=name,
                    rows=len(df),
                    columns=len(df.columns),
                    column_info=column_info,
                )
            )
        return infos

    def release(self, name: str) -> bool:
        """释放 DataFrame"""
        if name in self._dataframes:
            del self._dataframes[name]
            return True
        return False
