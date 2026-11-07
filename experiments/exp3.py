"""第三轮实验：题名聚合、贝叶斯平滑、院校相似度平滑。

用法：python scripts/exp3.py --split 2024-09-01
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
    utype, profile = task["utype"], task["profile"]
    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}
    hist = task["hist"]

    m = HybridRecommender({}).fit(hist, task["newbooks"])
    items = m.items
    n = len(items)
    idx_map = m.item_idx
    jj = hist["call_no"].map(idx_map)
    ok = jj.notna().values
    jarr = jj[ok].astype(int).values
    h = hist[ok]

    # 全局物品计数
    g_cnt = np.zeros(n, dtype=np.float64)
    np.add.at(g_cnt, jarr, 1.0)

    # 题名分组
    titles = h["title"].fillna("").values
    tcode = {t: i for i, t in enumerate(sorted(set(titles)))}
    tarr = np.asarray([tcode[t] for t in titles])
    title_of_item = {}
    for c, t in zip(items, [None] * n):
        pass
    info = hist.drop_duplicates("call_no").set_index("call_no")
    title_of_item = {c: tcode.get(str(info["title"].get(c, "")), -1) for c in items}
    title_members = defaultdict(list)
    for j, c in enumerate(items):
        t = title_of_item[c]
        if t >= 0:
            title_members[t].append(j)
    n_titles = len(tcode)

    # 学院计数（物品级与题名级）
    depts = sorted(set(h["dept"].values))
    dcode = {d: i for i, d in enumerate(depts)}
    darr = np.asarray([dcode[d] for d in h["dept"].values])
    D = len(depts)
    d_cnt = np.zeros((D, n), dtype=np.float64)
    np.add.at(d_cnt, (darr, jarr), 1.0)
    d_tcnt = np.zeros((D, n_titles), dtype=np.float64)
    np.add.at(d_tcnt, (darr, tarr), 1.0)
    d_n = d_cnt.sum(axis=1)

    # 院校相似度：基于一级类目分布
    cls_list = sorted({m.cls_of.get(c, "?") for c in items})
    ccode = {c: i for i, c in enumerate(cls_list)}
    item_cls = np.asarray([ccode[m.cls_of.get(c, "?")] for c in items])
    dc = np.zeros((D, len(cls_list)), dtype=np.float64)
    np.add.at(dc, (darr, item_cls[jarr]), 1.0)
    dc = dc / (dc.sum(axis=1, keepdims=True) + EPS)
    dn = np.sqrt((dc ** 2).sum(axis=1, keepdims=True)) + EPS
    dsim = (dc / dn) @ (dc / dn).T

    g_share = g_cnt / g_cnt.sum()
    g_title = np.zeros(n_titles)
    np.add.at(g_title, tarr, 1.0)
    g_title_share = g_title / g_title.sum()

    rows = []
    for rid in readers:
        prof = profile.get(rid, {})
        calls = hist_map.get(rid) or []
        seen = np.zeros(n, dtype=bool)
        si = [idx_map[c] for c in calls if c in idx_map]
        if si:
            seen[si] = True
        hit2 = np.zeros(n, dtype=bool)
        if calls:
            c2 = {m.cls2_of.get(c, "?") for c in calls if c in idx_map}
            hit2 = np.asarray([m.cls2_of.get(c, "?") in c2 for c in items])
        rows.append(dict(rid=rid, seen=seen, hit2=hit2, calls=calls,
                         d=dcode.get(prof.get("dept", "其他"), -1)))

    def run(tag, fn):
        pred = {}
        for r in rows:
            s = np.asarray(fn(r), dtype=np.float64).copy()
            s[r["seen"]] = -1e9
            order = np.argsort(-s)
            pred[r["rid"]] = [items[j] for j in order[:10] if s[j] > 0]
        res = evaluate(pred, task["truth"], utype, verbose=False)
        print(f"{tag:<40}{res['new']['hit']:8.4f}{res['old']['hit']:8.4f}"
              f"{res['old']['ndcg']:8.4f}{res['total']:8.2f}")
        return res

    def with_hist(s, r, k=1.0):
        return s * (1.0 + k * r["hit2"]) if r["calls"] else s

    def dept_vec(d, smoothing=0.0, sim_w=0.0):
        if d < 0:
            return g_share.copy()
        cnt = d_cnt[d].copy()
        if sim_w > 0:
            w = dsim[d].copy()
            w[d] = 0.0
            w = w / (w.sum() + EPS)
            cnt = (1 - sim_w) * cnt + sim_w * (w @ d_cnt) * (d_n[d] / (d_n.mean() + EPS))
        if smoothing > 0:
            cnt = cnt + smoothing * d_n[d] * g_share
        return cnt / (cnt.sum() + EPS)

    print(f"\n切分点 {args.split}")
    print(f"{'变体':<36}{'新人Hit':>9}{'老人Hit':>9}{'老人NDCG':>9}{'总分':>8}")
    print("-" * 74)

    run("R0 学院物品计数（基准）", lambda r: with_hist(dept_vec(r["d"]), r))

    # 贝叶斯平滑：向全局回归
    for lam in (0.05, 0.2, 0.5, 1.0):
        run(f"R1 贝叶斯平滑 λ={lam}",
            lambda r, lam=lam: with_hist(dept_vec(r["d"], smoothing=lam), r))

    # 院校相似度平滑
    for w in (0.15, 0.3, 0.5):
        run(f"R2 院校相似度平滑 w={w}",
            lambda r, w=w: with_hist(dept_vec(r["d"], sim_w=w), r))

    # 题名级聚合：把同题名的借阅量按副本数摊回
    def title_vec(d, mix):
        if d < 0:
            return g_share.copy()
        t_share = d_tcnt[d] / (d_tcnt[d].sum() + EPS)
        out = np.zeros(n)
        for t, mem in title_members.items():
            if t_share[t] <= 0:
                continue
            per = t_share[t] / len(mem)
            for j in mem:
                out[j] = per
        if mix <= 0:
            return out
        iv = d_cnt[d] / (d_cnt[d].sum() + EPS)
        return (1 - mix) * iv + mix * out

    for mix in (0.3, 0.5, 0.7, 1.0):
        run(f"R3 题名聚合 mix={mix}",
            lambda r, mix=mix: with_hist(title_vec(r["d"], mix), r))

    # 组合：平滑 + 题名
    run("R4 贝叶斯0.2 + 题名0.5",
        lambda r: with_hist(0.5 * dept_vec(r["d"], 0.2)
                            + 0.5 * title_vec(r["d"], 1.0), r))
    run("R5 贝叶斯0.2 + 题名0.5 + 院校0.2",
        lambda r: with_hist(0.5 * dept_vec(r["d"], 0.2, 0.2)
                            + 0.5 * title_vec(r["d"], 1.0), r))


if __name__ == "__main__":
    main()
