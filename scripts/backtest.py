"""回测入口：在全局时间切分上评估方案 C。

用法：
    python scripts/backtest.py --split 2024-09-01
    python scripts/backtest.py --split 2024-09-01 --set w_cf=0.5 --set alpha_pop=0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SPLIT_MAIN
from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT_MAIN)
    ap.add_argument("--set", action="append", default=[], help="覆盖超参，如 w_cf=0.5")
    args = ap.parse_args()

    params = {}
    for kv in args.set:
        k, v = kv.split("=")
        try:
            params[k] = float(v) if "." in v else int(v)
        except ValueError:
            params[k] = v

    task = build_backtest(args.split)
    hist_map = {
        rid: g["call_no"].tolist() for rid, g in task["hist_by_reader"].items()
    }
    readers = list(task["truth"].keys())

    model = HybridRecommender(params).fit(task["hist"], task["newbooks"])
    pred = model.recommend_all(
        readers, task["profile"], hist_map, task["utype"], k=10
    )
    res = evaluate(pred, task["truth"], {r: task["utype"][r] for r in readers})
    return res


if __name__ == "__main__":
    main()
