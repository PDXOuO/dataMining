"""打分器变体快速对比：同一模型实例上切换打分函数，避免重复 fit。

用法：python scripts/exp_scorer.py --split 2024-09-01
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest

EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-09-01")
    args = ap.parse_args()

    task = build_backtest(args.split)
    readers = list(task["truth"])
    utype = task["utype"]
    profile = task["profile"]
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}

    m = HybridRecommender({}).fit(task["hist"], task["newbooks"])
    n_items = len(m.items)
    items = m.items
    pop_g = m.pop_recent / (m.pop_recent.sum() + EPS)      # 近期全局份额
    pop_ga = m.pop_all / (m.pop_all.sum() + EPS)
    pop_max = pop_g.max()

    # 类目 → 成员
    cls_members = defaultdict(list)
    for j, c in enumerate(items):
        cls_members[m.cls_of.get(c, "?")].append(j)
    cls_members = {k: np.asarray(v) for k, v in cls_members.items()}

    # 每位读者的三类向量
    def vec_of(vec_dict, key):
        return vec_dict.get(key)

    rows = []
    for rid in readers:
        prof = profile.get(rid, {})
        calls = hist_map.get(rid) or []
        seen = np.asarray([m.item_idx[c] for c in calls if c in m.item_idx], dtype=int)
        seen_mask = np.zeros(n_items, dtype=bool)
        if len(seen):
            seen_mask[seen] = True

        dv = m.dept_vec.get(prof.get("dept", "其他"))
        mv = m.major_vec.get(prof.get("major", "未知"))
        if dv is None:
            dv = np.zeros(n_items, dtype=np.float32)
        if mv is None:
            mv = np.zeros(n_items, dtype=np.float32)

        # 历史：类目二级命中集合 + 一级类目分布
        hist_cls2 = {m.cls2_of.get(c, "?") for c in calls if c in m.item_idx}
        hit2 = np.zeros(n_items, dtype=bool)
        if hist_cls2:
            hit2 = np.asarray([m.cls2_of.get(c, "?") in hist_cls2 for c in items])
        h1 = defaultdict(float)
        for c in calls:
            h1[m.cls_of.get(c, "?")] += 1.0
        s1 = sum(h1.values()) or 1.0
        h1 = {k: v / s1 for k, v in h1.items()}
        hist_cls1 = np.asarray([h1.get(m.cls_of.get(c, "?"), 0.0) for c in items], dtype=np.float32)

        rows.append(dict(rid=rid, seen=seen_mask, dv=dv, mv=mv, hit2=hit2, h1=hist_cls1, calls=calls))

    def run(tag, fn):
        pred = {}
        for r in rows:
            s = fn(r).copy()
            s[r["seen"]] = -1e9
            order = np.argsort(-s)
            pred[r["rid"]] = [items[j] for j in order[:10] if s[j] > 0]
        res = evaluate(pred, task["truth"], utype, verbose=False)
        print(f"{tag:<44}{res['new']['hit']:8.4f}{res['old']['hit']:8.4f}"
              f"{res['old']['ndcg']:8.4f}{res['total']:8.2f}")
        return res

    print(f"\n切分点 {args.split}")
    print(f"{'变体':<40}{'新人Hit':>10}{'老人Hit':>10}{'老人NDCG':>10}{'总分':>8}")
    print("-" * 78)

    # V0 全局热门
    run("V0 全局近期热门", lambda r: pop_g.copy())

    # V1 纯学院物品热度（复现基线）
    run("V1 学院物品热度", lambda r: r["dv"].copy())

    # V2 学院 × 全局 几何融合
    for w in (0.5, 0.7, 1.0):
        run(f"V2 学院^{w} × 全局^{1-w:.1f}",
            lambda r, w=w: (r["dv"] / (r["dv"].max() + EPS) + 0.01) ** w
            * (pop_g / pop_max + 0.01) ** (1 - w))

    # V3 学院 + 专业
    for w in (0.3, 0.5):
        run(f"V3 学院^{1-w:.1f} × 专业^{w:.1f} × 全局",
            lambda r, w=w: ((1 - w) * r["dv"] / (r["dv"].max() + EPS)
                            + w * r["mv"] / (r["mv"].max() + EPS) + 0.01)
            * (pop_g / pop_max + 0.05))

    # V4 学院 + 历史二级类目提权
    for k in (0.5, 1.0, 2.0):
        run(f"V4 学院 × (1+{k}·同类目) 仅老人",
            lambda r, k=k: r["dv"] * (1.0 + k * r["hit2"]) if r["calls"] else r["dv"])

    # V5 学院 + 历史一级类目分布
    for k in (0.5, 1.0):
        run(f"V5 学院 × (1+{k}·历史一级类目) 仅老人",
            lambda r, k=k: r["dv"] * (1.0 + k * r["h1"]) if r["calls"] else r["dv"])


if __name__ == "__main__":
    main()
