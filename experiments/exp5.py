"""第五轮：分层平滑（物品→二级类目→全局）与老人侧补充信号。

用法：python scripts/exp5.py
"""
from __future__ import annotations

import sys
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

    # 二级类目成员
    cls2_codes = {}
    cls2_of = []
    for c in items:
        k = m.cls2_of.get(c, "?")
        if k not in cls2_codes:
            cls2_codes[k] = len(cls2_codes)
        cls2_of.append(cls2_codes[k])
    cls2_of = np.asarray(cls2_of)
    n_c2 = len(cls2_codes)
    member = np.zeros((n, n_c2), dtype=np.float64)
    member[np.arange(n), cls2_of] = 1.0

    d_c2 = d_cnt @ member                      # dept × cls2
    d_c2_n = d_c2.sum(axis=1, keepdims=True) + EPS
    d_c2_share = d_c2 / d_c2_n
    g_c2 = g_cnt @ member
    g_c2_share = g_c2 / (g_c2.sum() + EPS)
    within = g_cnt / (np.asarray(g_c2 @ member.T).ravel() + EPS)  # 类目内全局份额
    within = np.where(member.sum(axis=0)[cls2_of] > 0, within, 0.0)

    rows = []
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}
    for rid in task["truth"]:
        prof = task["profile"].get(rid, {})
        calls = hist_map.get(rid) or []
        seen = np.zeros(n, dtype=bool)
        si = [idx_map[c] for c in calls if c in idx_map]
        if si:
            seen[si] = True
        c2 = {m.cls2_of.get(c, "?") for c in calls if c in idx_map}
        hit2 = np.asarray([m.cls2_of.get(c, "?") in c2 for c in items]) if c2 else np.zeros(n, bool)
        rows.append(dict(rid=rid, seen=seen, calls=calls, hit2=hit2,
                         d=dcode.get(prof.get("dept", "其他"), -1)))
    return (task, m, items, n, d_cnt, d_n, g_share, d_c2_share, within, rows)


def run(task, items, rows, score_fn):
    pred = {}
    for r in rows:
        s = np.asarray(score_fn(r), dtype=np.float64).copy()
        s[r["seen"]] = -1e9
        order = np.argsort(-s)
        pred[r["rid"]] = [items[j] for j in order[:10] if s[j] > 0]
    return evaluate(pred, task["truth"], task["utype"], verbose=False)


def main():
    splits = ["2024-09-01", "2024-03-01"]
    data = {s: prepare(s) for s in splits}

    def make(lam, k2, hier):
        def f(dat, r):
            task, m, items, n, d_cnt, d_n, g_share, d_c2_share, within, _ = dat
            d = r["d"]
            if d < 0:
                return g_share
            target = (d_c2_share[d][m.cls2_of_codes] * within) if hier else g_share
            cnt = d_cnt[d] + lam * d_n[d] * target
            s = cnt / (cnt.sum() + EPS)
            if r["calls"]:
                s = s * (1.0 + k2 * r["hit2"])
            return s
        return f

    # 为 hierarchical 目标准备索引
    for s in splits:
        task, m, items, n, d_cnt, d_n, g_share, d_c2_share, within, rows = data[s]
        codes = {}
        arr = []
        for c in items:
            k = m.cls2_of.get(c, "?")
            if k not in codes:
                codes[k] = len(codes)
            arr.append(codes[k])
        m.cls2_of_codes = np.asarray(arr)

    print(f"{'变体':<30}{'09总分':>9}{'03总分':>9}{'均值':>8}")
    print("-" * 58)
    for hier in (False, True):
        for lam in (0.2, 0.35, 0.5, 0.7):
            for k2 in (0.8, 1.0, 1.3):
                fn = make(lam, k2, hier)
                tot = []
                for s in splits:
                    dat = data[s]
                    tot.append(run(dat[0], dat[2], dat[9],
                                   lambda r, dat=dat, fn=fn: fn(dat, r))["total"])
                tag = f"{'分层' if hier else '全局'} λ={lam} k2={k2}"
                print(f"{tag:<30}{tot[0]:9.2f}{tot[1]:9.2f}{np.mean(tot):8.2f}")


if __name__ == "__main__":
    main()
