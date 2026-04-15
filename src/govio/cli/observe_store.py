"""observe DataFrame 文件持久化存储

DataFrame 存储在 .govio/observe/dataframes/ 目录下的 parquet 文件中。
清单保存在 .govio/observe/manifest.json。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


OBSERVE_DIR = Path(".govio/observe")
DATAFRAMES_DIR = OBSERVE_DIR / "dataframes"
MANIFEST_FILE = OBSERVE_DIR / "manifest.json"


@dataclass
class DataFrameInfo:
    """DataFrame 信息"""

    name: str
    datasource: str
    sql: str
    file: str
    loaded_at: str
    rows: int
    columns: int
    column_info: list[dict] = field(default_factory=list)


@dataclass
class Manifest:
    """回放清单"""

    version: str = "1.0"
    dataframes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Manifest":
        if not MANIFEST_FILE.exists():
            return cls()
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            version=data.get("version", "1.0"),
            dataframes=data.get("dataframes", {}),
        )

    def save(self) -> None:
        OBSERVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"version": self.version, "dataframes": self.dataframes},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add(self, info: DataFrameInfo) -> None:
        self.dataframes[info.name] = {
            "datasource": info.datasource,
            "sql": info.sql,
            "file": info.file,
            "loaded_at": info.loaded_at,
            "rows": info.rows,
            "columns": info.columns,
            "column_info": info.column_info,
        }

    def remove(self, name: str) -> bool:
        if name in self.dataframes:
            del self.dataframes[name]
            return True
        return False


class ObserveStore:
    """基于文件的 DataFrame 存储"""

    def __init__(self) -> None:
        self._manifest = Manifest.load()

    def _ensure_dataframes_dir(self) -> None:
        DATAFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        name: str,
        df: pd.DataFrame,
        datasource: str,
        sql: str,
    ) -> DataFrameInfo:
        """存储 DataFrame 到 parquet 文件"""
        self._ensure_dataframes_dir()

        file_path = DATAFRAMES_DIR / f"{name}.parquet"
        df.to_parquet(file_path, index=False)

        column_info = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]

        info = DataFrameInfo(
            name=name,
            datasource=datasource,
            sql=sql,
            file=str(file_path),
            loaded_at=datetime.now(timezone.utc).isoformat(),
            rows=len(df),
            columns=len(df.columns),
            column_info=column_info,
        )

        self._manifest.add(info)
        self._manifest.save()

        return info

    def get(self, name: str) -> pd.DataFrame | None:
        """加载 DataFrame 到内存"""
        if name not in self._manifest.dataframes:
            return None

        file_path = Path(self._manifest.dataframes[name]["file"])
        if not file_path.exists():
            # 文件不存在，清理清单
            self._manifest.remove(name)
            self._manifest.save()
            return None

        return pd.read_parquet(file_path)

    def list(self) -> list[DataFrameInfo]:
        """列出所有 DataFrame"""
        infos = []
        for name, data in self._manifest.dataframes.items():
            column_info = data.get("column_info", [])
            if not column_info:
                # 尝试从文件读取
                file_path = Path(data["file"])
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    column_info = [
                        {"name": col, "dtype": str(df[col].dtype)}
                        for col in df.columns
                    ]

            infos.append(
                DataFrameInfo(
                    name=name,
                    datasource=data["datasource"],
                    sql=data["sql"],
                    file=data["file"],
                    loaded_at=data["loaded_at"],
                    rows=data["rows"],
                    columns=data["columns"],
                    column_info=column_info,
                )
            )
        return infos

    def release(self, name: str) -> bool:
        """释放 DataFrame — 删除文件并从清单移除"""
        if name not in self._manifest.dataframes:
            return False

        file_path = Path(self._manifest.dataframes[name]["file"])
        if file_path.exists():
            file_path.unlink()

        self._manifest.remove(name)
        self._manifest.save()
        return True

    def exists(self, name: str) -> bool:
        """检查 DataFrame 是否存在"""
        return name in self._manifest.dataframes
