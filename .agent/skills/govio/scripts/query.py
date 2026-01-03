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
    parser.add_argument('--graph', type=str, help='图数据库名称', default="ontology")
    parser.add_argument('--cypher', type=str, help='Cypher', required=True)

    # 解析命令行参数
    args = parser.parse_args()
    g = FalkorDBGraph(graph = args.graph)

    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir()

    _cypher = args.cypher
    if not _cypher.upper().startswith('MATCH'):
        print('请编写一个MATCH的查询。') 
        sys.exit(1)
    
    logger.info("cypher: "+ _cypher)

    df = pd.DataFrame(g.query(args.cypher))
    if df.shape[0] > 0:
        fname = ASSETS_DIR / f"output-{datetime.now().strftime("%Y%m%d%H%M%s")}.csv"
        df.to_csv(ASSETS_DIR / fname, index=False)
        print("Result output:", fname)
        logger.info("result rows: %s", str(df.shape[0]))
        logger.info("result file: %s", str(fname))
    else:
        print("Data not found.")
        logger.info("data not found.")
