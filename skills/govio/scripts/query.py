"""
执行Cypher查询
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import json

from govio import FalkorDBGraph

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
ASSETS_DIR = Path(__file__).parent.parent / "assets"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='Graph name', default="ontology")
    parser.add_argument('--cypher', type=str, help='Cypher', required=True)

    # 解析命令行参数
    args = parser.parse_args()
    g = FalkorDBGraph(graph = args.graph)

    if not ASSETS_DIR.exists():
        from load_schema import load_schema
        load_schema(g)
        from load_names import generate_names
        generate_names(g)
        print("Please read [schema.md](assets/schema.md) before make cypher.")
        exit(1)

    _cypher = args.cypher
    if not _cypher.upper().startswith('MATCH'):
        print('Please write a MATCH query.') 
        sys.exit(1)
    
    logger.info("cypher: "+ _cypher)

    data = g.query(args.cypher)
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
