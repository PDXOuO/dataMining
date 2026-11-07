"""第二轮实验：在「学院物品热度」骨架上叠加各类细化信号。

用法：python scripts/exp2.py --split 2024-09-01
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest

EPS = 1e-9


def cap_pick(score, items, cap):
    """类目上限选取；cap=None 表示不限制。"""
    order = np.argsort(-score)
    if not cap:
        return [items[j] for j in order[:10] if score[j] > 0]
    cnt, out = defaultdict(int), []
    for j in order:
        if len(out) >= 10:
            break
        if score[j] <= 0:
            continue
        c = items[j]
        cl = c.split("/")[0][:1] if "/" in c else "?"
        if cnt[cl] >= cap:
            continue
        out.append(c)
        cnt[cl] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-09-01")
    args = ap.parse_args()

    task = build_backtest(args.split)
    readers = list(task["truth"])
    utype, profile = task["utype"], task["profile"]
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}

    m = HybridRecommender({}).fit(task["hist"], task["newbooks"])
    items = m.items
    n = len(items)
    idx_map = m.item_idx

    # 不同时间窗口的学院累积向量
    hist = task["hist"]
    end = pd.Timestamp(args.split)
    dept_windows = {}
    for days, tag in ((365, "1y"), (730, "2y"), (9999, "all")):
        sub = hist[hist["ts"] >= end - pd.Timedelta(days=days)]
        acc = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        jj = sub["call_no"].map(idx_map)
        ok = jj.notna().values
        for d, j in zip(sub["dept"].values[ok], jj[ok].astype(int).values):
            acc[d][j] += 1.0
        dept_windows[tag] = {k: v / (v.sum() + EPS) for k, v in acc.items()}

    # 学院 × 读者级别
    acc2 = defaultdict(lambda: np.zeros(n, dtype=np.float32))
    jj = hist["call_no"].map(idx_map)
    ok = jj.notna().values
    for d, lv, j in zip(hist["dept"].values[ok], hist["level"].values[ok], jj[ok].astype(int).values):
        acc2[(d, lv)][j] += 1.0
    dept_level = {k: v / (v.sum() + EPS) for k, v in acc2.items()}

    rows = []
    for rid in readers:
        prof = profile.get(rid, {})
        calls = hist_map.get(rid) or []
        seen = np.zeros(n, dtype=bool)
        si = [idx_map[c] for c in calls if c in idx_map]
        if si:
            seen[si] = True
        d = prof.get("dept", "其他")
        lv = prof.get("level", "未知")
        hit2 = np.zeros(n, dtype=bool)
        if calls:
            c2 = {m.cls2_of.get(c, "?") for c in calls if c in idx_map}
            hit2 = np.asarray([m.cls2_of.get(c, "?") in c2 for c in items])
        rows.append(dict(rid=rid, seen=seen, hit2=hit2, calls=calls, dept=d, level=lv,
                         dv={t: dept_windows[t].get(d) for t in dept_windows},
                         dl=dept_level.get((d, lv))))

    def run(tag, fn, cap_new=None, cap_old=None):
        pred = {}
        for r in rows:
            s = fn(r).copy()
            s[r["seen"]] = -1e9
            pred[r["rid"]] = cap_pick(s, items, cap_old if r["calls"] else cap_new)
        res = evaluate(pred, task["truth"], utype, verbose=False)
        print(f"{tag:<42}{res['new']['hit']:8.4f}{res['new']['recall']:8.4f}"
              f"{res['old']['hit']:8.4f}{res['old']['ndcg']:8.4f}{res['total']:8.2f}")
        return res

    print(f"\n切分点 {args.split}")
    print(f"{'变体':<38}{'新人Hit':>9}{'新人Rec':>9}{'老人Hit':>9}{'老人NDCG':>9}{'总分':>8}")
    print("-" * 82)

    def base(r):
        v = r["dv"]["all"]
        if v is None:
            return np.zeros(n, dtype=np.float32)
        s = v.astype(np.float32)
        if r["calls"]:
            s = s * (1.0 + 1.0 * r["hit2"])
        return s

    run("V4 基线（学院全期 + 同类目提权）", base)

    # 时间窗口
    for t in ("1y", "2y", "all"):
        run(f"窗口 {t}",
            lambda r, t=t: (r["dv"][t] if r["dv"][t] is not None else np.zeros(n, np.float32))
            * (1.0 + (1.0 * r["hit2"] if r["calls"] else 0)))

    # 学院 × 级别
    run("学院×级别（稀疏时回退学院）",
        lambda r: ((r["dl"] if r["dl"] is not None else r["dv"]["all"]) if r["dv"]["all"] is not None
                   else np.zeros(n, np.float32)) * (1.0 + (1.0 * r["hit2"] if r["calls"] else 0)))

    # 学院 + 级别融合
    for w in (0.3, 0.5):
        run(f"学院^{1-w:.1f} × 学院级别^{w:.1f}",
            lambda r, w=w: (((1 - w) * r["dv"]["all"] + w * r["dl"]) if r["dl"] is not None
                            else r["dv"]["all"]) * (1.0 + (1.0 * r["hit2"] if r["calls"] else 0)))

    # 类目上限
    for capn, capo in ((None, None), (2, 3), (3, 4), (4, 5), (5, 10)):
        run(f"类目上限 新人{capn} / 老人{capo}", base, cap_new=capn, cap_old=capo)

    # 老人同类目提权强度
    for k in (0.5, 1.0, 1.5, 2.5):
        run(f"老人同类目提权 k={k}",
            lambda r, k=k: (r["dv"]["all"] * (1.0 + k * r["hit2"])) if r["calls"] else r["dv"]["all"])


if __name__ == "__main__":
    main()
