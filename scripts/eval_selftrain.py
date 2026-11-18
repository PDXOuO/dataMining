"""决赛数据自监督验证：把老人的历史按时间切成「可见」与「待预测」。

为什么需要它：初赛四折能验证模型结构，但那是另一批读者、另一套候选池；
决赛答案不公开，唯一贴近决赛的信号就是「用同一批读者的前半段历史，
预测他们的后半段历史」。伪答案是读者自己后段借的书，天然在候选池内、
且已剔除可见部分，与决赛「推荐他没读过的新书」结构一致。

局限：伪答案必是同一个人借过的书，模型只要押中他惯常的类目就能得分，
因此绝对分数会高于决赛真实分；只用于变体之间的相对比较。

用法：
    python scripts/eval_selftrain.py
    python scripts/eval_selftrain.py --set k_cls1=0.8 --set max_new=3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.final_data import Bench, load_bench, load_train_base  # noqa: E402
from src.final_model import FinalRecommender  # noqa: E402
from src.metrics import per_user  # noqa: E402


def make_bench(bench: Bench, hold_ratio: float = 0.3, min_hist: int = 6):
    """把每个足够长的历史切成可见段与伪答案段，构造一个新的 Bench + 答案。"""
    hist, truth = {}, {}
    for rid, seq in bench.hist.items():
        if len(seq) < min_hist:
            hist[rid] = seq
            continue
        k = max(1, int(round(len(seq) * hold_ratio)))
        k = min(k, 5)                       # 伪答案最多 5 本，贴近真实答案规模
        vis, held = seq[:-k], seq[-k:]
        hist[rid] = vis
        truth[rid] = held

    # 只保留有伪答案的读者做评估，其余读者不进名单
    keep = set(truth)
    readers = bench.readers[bench.readers["读者ID"].isin(keep)].reset_index(drop=True)
    return Bench(bench.name, readers, bench.whitelist, bench.calls,
                 bench.call_cls1, bench.call_cls2, bench.call_new, hist), truth


def run(bench_name: str, params: dict, hold_ratio: float, min_hist: int, verbose=True):
    base = load_bench(bench_name)
    bench, truth = make_bench(base, hold_ratio, min_hist)
    train = load_train_base()
    model = FinalRecommender(params).fit(bench, train)
    pred = model.recommend_all()

    rows = [per_user(pred.get(rid, []), g) for rid, g in truth.items()]
    n = len(rows)
    hit = sum(r[0] for r in rows) / n
    rec = sum(r[1] for r in rows) / n
    nd = sum(r[2] for r in rows) / n
    # 老人组公式（自监督只覆盖老人）
    score = (0.4 * hit + 0.3 * rec + 0.3 * nd) * 100
    if verbose:
        print(f"  [{bench_name}] n={n:3d}  Hit={hit:.3f} Recall={rec:.3f} "
              f"NDCG={nd:.3f}  老人组 {score:5.2f}")
    return {"n": n, "hit": hit, "recall": rec, "ndcg": nd, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hidden")
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--hold", type=float, default=0.3)
    ap.add_argument("--min-hist", type=int, default=6)
    args = ap.parse_args()

    params = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v or "e" in v.lower() else int(v)

    print(f"自监督留出：hold={args.hold} min_hist={args.min_hist} params={params or '默认'}")
    run(args.bench, params, args.hold, args.min_hist)


if __name__ == "__main__":
    main()
