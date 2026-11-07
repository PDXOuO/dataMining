"""消融与基线对照实验，产出技术报告第 5 节的全部数字。

所有变体在同一个回测任务上运行，共享同一候选池与切分点，保证可比。
用法：
    python scripts/ablation.py --split 2024-09-01
    python scripts/ablation.py --split 2024-03-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest

FULL = {
    "top_n": 1000,
    "beta_pop": 0.40,
    "gamma_cls": 0.80,
    "w_self": 0.70,
    "max_per_cls_new": 6,
    "max_per_cls_old": 4,
}


# ---------------- 基线 ----------------
def baseline_global(model, readers, hist_map, dept_map, by_dept=False, k=10):
    out = {}
    for rid in readers:
        seen = {model.item_idx[c] for c in (hist_map.get(rid) or []) if c in model.item_idx}
        pop = model.pop_recent
        if by_dept and dept_map.get(rid) in model.dept_vec:
            # 学院热度：用该学院的物品偏好向量作为排序依据
            score = model.dept_vec[dept_map[rid]]
        else:
            score = pop
        order = np.argsort(-score)
        picked = [model.items[j] for j in order if j not in seen][:k]
        out[rid] = picked
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-09-01")
    args = ap.parse_args()

    task = build_backtest(args.split)
    readers = [r for r in task["truth"]]
    utype = task["utype"]
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}
    profile = task["profile"]
    dept_map = {r: profile.get(r, {}).get("dept", "其他") for r in readers}

    model = HybridRecommender(FULL).fit(task["hist"], task["newbooks"])

    runs: list[tuple[str, dict]] = []

    # 1. 两个官方口径基线
    runs.append(("全局热门 Top10", baseline_global(model, readers, hist_map, dept_map)))
    runs.append(
        ("同学院热门 Top10", baseline_global(model, readers, hist_map, dept_map, by_dept=True))
    )

    # 2. 我们的方案：逐项开启
    def run(tag, **over):
        cfg = {**FULL, **over}
        model.p.update(cfg)
        dists = model.prepare_dists(readers, profile, hist_map)
        runs.append((tag, model.recommend_all(readers, profile, hist_map, utype, 10, dists)))
        for k, v in FULL.items():
            model.p[k] = v

    run("仅热门候选池（β=1,γ=0,无类目上限）",
        beta_pop=1.0, gamma_cls=0.0, max_per_cls_new=10, max_per_cls_old=10)
    run("+ 类目偏好重排（β=.4,γ=.8）",
        max_per_cls_new=10, max_per_cls_old=10)
    run("+ 类目上限（完整方案）")

    # 3. 关键组件消融
    run("消融：关闭新书时间加权", use_new_weight=False)
    run("消融：关闭学院/专业先验", w_self=1.0)
    run("消融：仅用群体先验（w_self=0）", w_self=0.0)
    run("消融：候选池缩小至 300", top_n=300)
    run("消融：候选池扩大至 3000", top_n=3000)

    rows = []
    print(f"\n回测切分点 {args.split}｜候选池 {len(task['cand'])} 册\n")
    print(f"{'变体':<34}{'新人Hit':>8}{'新人Rec':>8}{'老人Hit':>8}{'老人Rec':>8}{'老人NDCG':>9}{'总分':>7}")
    print("-" * 84)
    for tag, pred in runs:
        r = evaluate(pred, task["truth"], utype, verbose=False)
        rows.append((tag, r))
        print(
            f"{tag:<34}{r['new']['hit']:8.4f}{r['new']['recall']:8.4f}"
            f"{r['old']['hit']:8.4f}{r['old']['recall']:8.4f}{r['old']['ndcg']:9.4f}{r['total']:7.2f}"
        )
    return rows


if __name__ == "__main__":
    main()
