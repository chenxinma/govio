import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

import pandas as pd

from govio import NetworkXGraph


ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOCAL_DIR = Path(__file__).parent
log_dir = os.path.join(LOCAL_DIR / "../", 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'query_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过执行Python脚本进行查询，存在g是NetworkX的图对象，通过result=...获得返回结果")
    parser.add_argument('--code', type=str, help='NetworkX Query Pyhton Code', required=True)

    # 解析命令行参数
    args = parser.parse_args()
    gf = NetworkXGraph(ASSETS_DIR / "ontology.gml")
    g = gf.G
    
    if not ASSETS_DIR.exists():
        from load_schema import load_schema
        load_schema(gf)
        from load_names import generate_names
        generate_names(g)
        print("Please read [schema.md](assets/schema.md) before make python code.")
        exit(1)

    _code = args.code
    logger.info("Code: "+ _code)

    local_vars = {"g": g}
    exec(_code, locals=local_vars)
    data = local_vars.get("result")
    if isinstance(data, list):
        _size = len(data)
        if _size > 0:
            if _size > 10:
                fname = ASSETS_DIR / f"output-{datetime.now().strftime("%Y%m%d%H%M%s")}.json"
                df = pd.DataFrame(data)
                df.to_json(fname, index=False, orient="records", lines=True, force_ascii=False)
                print("Result output:", fname, "rows:", _size)
                logger.info("result file: %s", str(fname))
            else:
                print(json.dumps(data, ensure_ascii=False, default=lambda obj: obj.__dict__))
            logger.info("result rows: %s", str(_size))
        else:
            print("Data not found.")
            logger.info("data not found.")
    else:
        print(json.dumps(data, ensure_ascii=False, default=lambda obj: obj.__dict__))
