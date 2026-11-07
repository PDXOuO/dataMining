"""参数网格搜索：一次 fit，多次评估，避免重复构建索引。"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SPLIT_MAIN
from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest

GRID = {
    "top_n": [100, 300, 1000],
    "beta_pop": [0.3, 0.5, 0.7],
    "gamma_cls": [0.4, 0.8, 1.2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT_MAIN)
    ap.add_argument("--w_self", type=float, nargs="+", default=[0.6, 0.8])
    args = ap.parse_args()

    task = build_backtest(args.split)
    hist_map = {rid: g["call_no"].tolist() for rid, g in task["hist_by_reader"].items()}
    readers = list(task["truth"].keys())
    utype = {r: task["utype"][r] for r in readers}

    model = HybridRecommender().fit(task["hist"], task["newbooks"])
    rows = []
    for w_self in args.w_self:
        model.p["w_self"] = w_self
        dists = model.prepare_dists(readers, task["profile"], hist_map)
        for top_n, beta, gamma in itertools.product(
            GRID["top_n"], GRID["beta_pop"], GRID["gamma_cls"]
        ):
            model.p.update({"top_n": top_n, "beta_pop": beta, "gamma_cls": gamma})
            pred = model.recommend_all(
                readers, task["profile"], hist_map, utype, k=10, dists=dists
            )
            res = evaluate(pred, task["truth"], utype, verbose=False)
            rows.append(
                (res["total"], w_self, top_n, beta, gamma, res["new"]["hit"], res["old"]["hit"])
            )

    rows.sort(reverse=True)
    print(f"{'total':>6} {'w_self':>6} {'top_n':>6} {'beta':>5} {'gamma':>5} {'new_hit':>8} {'old_hit':>8}")
    for r in rows[:12]:
        print(f"{r[0]:6.2f} {r[1]:6.1f} {r[2]:6d} {r[3]:5.2f} {r[4]:5.2f} {r[5]:8.4f} {r[6]:8.4f}")


if __name__ == "__main__":
    main()
