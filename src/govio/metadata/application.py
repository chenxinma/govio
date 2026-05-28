from pathlib import Path
import pandas as pd


class AppInfoLoader:
    def __init__(self, app_list_file: str | Path, app_limits: list[str] | None = None):
        self.app_list_file = app_list_file
        self.app_limits = app_limits
    
    def load_apps(self):
        _df_apps_excel = pd.read_excel(self.app_list_file, sheet_name="应用清单", dtype={"APPID": str})[[
                                        "APPID", 
                                        "应用系统\n中文名称", 
                                        "应用系统\n英文简称", 
                                        "系统类型", 
                                        "业务分类", 
                                        "系统管理部门", 
                                        "所属网络区域", 
                                        "系统分级",
                                        "外部供应商"
                                    ]].rename(columns={ 
                                        "APPID": "app_id",
                                        "应用系统\n中文名称": "name",
                                        "应用系统\n英文简称": "app_name_en",
                                        "系统类型": "app_type",
                                        "业务分类": "business_domain",
                                        "系统管理部门": "manager",
                                        "所属网络区域": "network_area",
                                        "系统分级": "maintenance_level",
                                        "外部供应商": "external_vendor"
                                    }) 
        _df_app_fix = pd.DataFrame([{"app_id": "0000", 
                                    "name": "分析数据集", 
                                    "app_name_en": "analyze dataset", 
                                    "app_type": "数据集",
                                    "business_domain": "数据",
                                    "manager": "",
                                    "network_area": "本地", 
                                    "maintenance_level": "C",
                                    "external_vendor": "无"}])
        df_apps = pd.concat([_df_apps_excel, _df_app_fix])
        
        if self.app_limits:
            return df_apps[df_apps["name"].isin(self.app_limits)].copy()
        else:
            return df_apps


    @property
    def Application(self):
        return self.load_apps()
