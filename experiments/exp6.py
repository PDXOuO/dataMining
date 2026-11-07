"""第六轮：季节窗口、类目强度、院内共现。

用法：python experiments/exp6.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate
from src.model_final import DeptBayesRecommender
from src.splits import build_backtest

EPS = 1e-9
SPLITS = ["2023-09-01", "2024-03-01", "2024-06-01", "2024-09-01"]


def prepare(split):
    task = build_backtest(split)
    m = DeptBayesRecommender({"newbook_quantile": 0.0}).fit(task["hist"], task["newbooks"])
    items, n = m.items, len(m.items)
    idx = m.item_idx
    h = task["hist"]
    jj = h["call_no"].map(idx)
    ok = jj.notna().values
    jarr = jj[ok].astype(int).values
    depts = sorted(set(h["dept"].dropna().values))
    dcode = {d: i for i, d in enumerate(depts)}
    darr = np.asarray([dcode[d] for d in h["dept"].values[ok]])
    ts = pd.to_datetime(h["ts"].values[ok]).month

    def counts(mask):
        c = np.zeros((len(depts), n))
        np.add.at(c, (darr[mask], jarr[mask]), 1.0)
        return c

    d_all = counts(np.ones(len(jarr), bool))
    d_season = counts(np.isin(ts, [9, 10, 11, 12]))
    d_blend = d_all + d_season  # 同月份记录额外计一次

    m_ = m
    # 类目层级先验（与定稿一致）
    member = m_._member
    g_cnt = d_all.sum(axis=0)
    g_c2 = g_cnt @ member
    within = g_cnt / (np.asarray(g_c2 @ member.T).ravel() + EPS)
    d_c2 = d_all @ member
    d_c2_share = d_c2 / (d_c2.sum(axis=1, keepdims=True) + EPS)

    rows = []
    hm = {r: df["call_no"].tolist() for r, df in task["hist_by_reader"].items()}
    for rid in task["truth"]:
        prof = task["profile"].get(rid, {})
        calls = hm.get(rid) or []
        seen = np.zeros(n, bool)
        si = [idx[c] for c in calls if c in idx]
        if si:
            seen[si] = True
        cnt_c2 = defaultdict(int)
        for c in calls:
            if c in idx:
                cnt_c2[m_.cls2[idx[c]]] += 1
        tot = sum(cnt_c2.values()) or 1
        inten = np.asarray([cnt_c2.get(c, 0) / tot for c in m_.cls2]) if calls else np.zeros(n)
        hit2 = (inten > 0).astype(float)
        seen_c1 = {m_.cls1[idx[c]] for c in calls if c in idx}
        hit1 = np.asarray([1.0 if c in seen_c1 else 0.0 for c in m_.cls1]) if calls else np.zeros(n)
        rows.append(dict(rid=rid, seen=seen, calls=calls, hit2=hit2, hit1=hit1, inten=inten,
                         d=dcode.get(prof.get("dept", "其他"), -1)))
    return task, m_, items, n, d_all, d_season, d_blend, d_c2_share, within, rows


def score_matrix(dat, which="all", lam=1.5):
    task, m, items, n, d_all, d_season, d_blend, d_c2_share, within, rows = dat
    cnt = {"all": d_all, "season": d_season, "blend": d_blend}[which]
    dn = cnt.sum(axis=1)
    out = {}
    for d in range(cnt.shape[0]):
        prior = d_c2_share[d][m.cls2_code] * within
        v = cnt[d] + lam * dn[d] * prior
        out[d] = v / (v.sum() + EPS)
    return out


def run(dat, smat, tag, k2=1.6, inten_w=0.0, quiet=False):
    task, m, items, n = dat[0], dat[1], dat[2], dat[3]
    rows = dat[9]
    g = smat[-1] if -1 in smat else None
    pred = {}
    for r in rows:
        d = r["d"]
        s = (smat.get(d) if d >= 0 else None)
        if s is None:
            s = np.asarray([sum(mm[d2][j] for d2 in smat) for j in range(n)]) if g is None else g
        s = s.copy()
        if r["calls"]:
            boost = r["hit2"] * k2 if inten_w <= 0 else k2 * (r["inten"] / (r["inten"].max() + EPS))
            s = s * (1.0 + boost)
        s[r["seen"]] = -1e9
        order = np.argsort(-s)
        pred[r["rid"]] = [items[j] for j in order[:10] if s[j] > 0]
    res = evaluate(pred, task["truth"], task["utype"], verbose=False)
    if not quiet:
        print(f"{tag:<34}{res['new']['hit']:8.4f}{res['old']['hit']:8.4f}"
              f"{res['old']['ndcg']:8.4f}{res['total']:8.2f}")
    return res["total"]


def main():
    data = {s: prepare(s) for s in SPLITS}
    print(f"\n{'变体':<30}" + "".join(f"{s[2:7]:>9}" for s in SPLITS) + f"{'均值':>8}")
    print("-" * 72)

    def sweep(tag, fn):
        vals = [fn(data[s], s) for s in SPLITS]
        print(f"{tag:<30}" + "".join(f"{v:9.2f}" for v in vals) + f"{np.mean(vals):8.2f}")

    for which in ("all", "season", "blend"):
        sweep(f"计数窗口 {which}", lambda d, s, w=which: run(d, score_matrix(d, w), "_", k2=1.6, quiet=True))

    for k2 in (0.0, 1.0, 1.6, 2.2):
        sweep(f"细分 k2={k2}", lambda d, s, k=k2: run(d, score_matrix(d, "all"), "_", k2=k, quiet=True))

    for k2 in (1.0, 1.6, 2.5):
        sweep(f"强度加权 k2={k2}", lambda d, s, k=k2: run(d, score_matrix(d, "all"), "_", k2=k, inten_w=1.0, quiet=True))


if __name__ == "__main__":
    main()
