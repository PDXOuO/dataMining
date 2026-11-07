"""第四轮：在「学院计数 + 贝叶斯平滑」骨架上做精细网格，并在两个切分点上交叉验证。

用法：python scripts/exp4.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate
from src.recommender import HybridRecommender
from src.splits import build_backtest

EPS = 1e-9


def prepare(split):
    task = build_backtest(split)
    m = HybridRecommender({}).fit(task["hist"], task["newbooks"])
    items, n = m.items, len(m.items)
    idx_map = m.item_idx
    h = task["hist"]
    jj = h["call_no"].map(idx_map)
    ok = jj.notna().values
    jarr = jj[ok].astype(int).values
    depts = sorted(set(h["dept"].values))
    dcode = {d: i for i, d in enumerate(depts)}
    darr = np.asarray([dcode[d] for d in h["dept"].values[ok]])
    d_cnt = np.zeros((len(depts), n), dtype=np.float64)
    np.add.at(d_cnt, (darr, jarr), 1.0)
    d_n = d_cnt.sum(axis=1)
    g_cnt = d_cnt.sum(axis=0)
    g_share = g_cnt / g_cnt.sum()

    rows = []
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}
    for rid in task["truth"]:
        prof = task["profile"].get(rid, {})
        calls = hist_map.get(rid) or []
        seen = np.zeros(n, dtype=bool)
        si = [idx_map[c] for c in calls if c in idx_map]
        if si:
            seen[si] = True
        c1 = {m.cls_of.get(c, "?") for c in calls if c in idx_map}
        c2 = {m.cls2_of.get(c, "?") for c in calls if c in idx_map}
        au = {m.auth_of.get(c, "") for c in calls if c in idx_map} - {""} if hasattr(m, "auth_of") else set()
        hit1 = np.asarray([m.cls_of.get(c, "?") in c1 for c in items]) if c1 else np.zeros(n, bool)
        hit2 = np.asarray([m.cls2_of.get(c, "?") in c2 for c in items]) if c2 else np.zeros(n, bool)
        rows.append(dict(rid=rid, seen=seen, calls=calls, hit1=hit1, hit2=hit2,
                         d=dcode.get(prof.get("dept", "其他"), -1)))
    return task, m, items, n, d_cnt, d_n, g_share, rows


def evaluate_cfg(task, items, d_cnt, d_n, g_share, rows, lam, k2, k1):
    pred = {}
    for r in rows:
        d = r["d"]
        if d < 0:
            s = g_share.copy()
        else:
            cnt = d_cnt[d] + lam * d_n[d] * g_share
            s = cnt / (cnt.sum() + EPS)
        if r["calls"]:
            s = s * (1.0 + k2 * r["hit2"] + k1 * r["hit1"])
        s = s.copy()
        s[r["seen"]] = -1e9
        order = np.argsort(-s)
        pred[r["rid"]] = [items[j] for j in order[:10] if s[j] > 0]
    return evaluate(pred, task["truth"], task["utype"], verbose=False)


def main():
    splits = ["2024-09-01", "2024-03-01"]
    data = {s: prepare(s) for s in splits}

    grids = []
    for lam in (0.2, 0.35, 0.5, 0.7):
        for k2 in (0.8, 1.0, 1.3):
            for k1 in (0.0, 0.3, 0.6):
                res = {}
                for s in splits:
                    task, m, items, n, d_cnt, d_n, g_share, rows = data[s]
                    res[s] = evaluate_cfg(task, items, d_cnt, d_n, g_share, rows, lam, k2, k1)
                grids.append((lam, k2, k1, res))

    grids.sort(key=lambda x: -(x[3][splits[0]]["total"] + x[3][splits[1]]["total"]))
    print(f"{'lam':>5}{'k2':>5}{'k1':>5} | {'09总分':>8}{'09新人':>8}{'09老人':>8} | "
          f"{'03总分':>8}{'03新人':>8}{'03老人':>8} | {'均值':>7}")
    print("-" * 88)
    for lam, k2, k1, res in grids[:14]:
        a, b = res["2024-09-01"], res["2024-03-01"]
        avg = (a["total"] + b["total"]) / 2
        print(f"{lam:5.2f}{k2:5.2f}{k1:5.2f} | {a['total']:8.2f}{a['new']['hit']:8.4f}"
              f"{a['old']['hit']:8.4f} | {b['total']:8.2f}{b['new']['hit']:8.4f}"
              f"{b['old']['hit']:8.4f} | {avg:7.2f}")


if __name__ == "__main__":
    main()
