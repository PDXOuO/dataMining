"""第七轮：老人组专项（全局热度融合、标题去重、一级类目提权）。

用法：python experiments/exp7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.exp6 import SPLITS, prepare, score_matrix
from src.metrics import evaluate

EPS = 1e-9


def build_assets(dat):
    task, m, items, n, d_all, d_season, d_blend, d_c2_share, within, rows = dat
    g_cnt = d_all.sum(axis=0)
    g = g_cnt / (g_cnt.sum() + EPS)
    # 标题分组：同一题名视为重复，避免占两个推荐位
    import pandas as pd

    title = {}
    info = task["hist"].drop_duplicates("call_no").set_index("call_no")
    for c in items:
        tinfo = None
        if c in info.index:
            tinfo = info.at[c, "title"] if "title" in info.columns else None
        title[c] = tinfo if isinstance(tinfo, str) and tinfo else c
    title_id = {}
    for c in items:
        title_id[c] = title_id.setdefault(title[c], len(title_id))
    tcode = np.asarray([title_id[title[c]] for c in items])
    return g, tcode


def predict(dat, smat, k2=1.6, w_glob=0.0, use_cls1=0.0, dedup_title=False):
    task, m, items, n = dat[0], dat[1], dat[2], dat[3]
    rows = dat[9]
    g, tcode = predict.g, predict.tcode
    pred = {}
    for r in rows:
        d = r["d"]
        s = (smat.get(d) if d >= 0 else g).copy()
        if w_glob > 0:
            s = np.power(s + EPS, 1 - w_glob) * np.power(g + EPS, w_glob)
        if r["calls"]:
            s = s * (1.0 + k2 * r["hit2"])
            if use_cls1 > 0:
                s = s * (1.0 + use_cls1 * r["hit1"])
        s[r["seen"]] = -1e9
        order = np.argsort(-s)
        picked, seen_t = [], set()
        for j in order:
            if len(picked) >= 10:
                break
            if s[j] <= 0:
                continue
            if dedup_title and tcode[j] in seen_t:
                continue
            picked.append(items[j])
            seen_t.add(tcode[j])
        pred[r["rid"]] = picked
    return evaluate(pred, task["truth"], task["utype"], verbose=False)


def main():
    data = {s: prepare(s) for s in SPLITS}
    assets = {s: build_assets(data[s]) for s in SPLITS}

    print(f"\n{'变体':<28}" + "".join(f"{s[2:7]:>9}" for s in SPLITS) + f"{'均值':>8}")
    print("-" * 70)

    def sweep(tag, **kw):
        vals = []
        for s in SPLITS:
            dat = data[s]
            predict.g, predict.tcode = assets[s]
            sm = score_matrix(dat, "all")
            vals.append(predict(dat, sm, **kw)["total"])
        print(f"{tag:<28}" + "".join(f"{v:9.2f}" for v in vals) + f"{np.mean(vals):8.2f}")
        return np.mean(vals)

    sweep("基线 k2=1.6")
    for w in (0.1, 0.2, 0.3):
        sweep(f"+全局热度融合 w={w}", w_glob=w)
    sweep("+标题去重", dedup_title=True)
    for c1 in (0.2, 0.4):
        sweep(f"+一级类目提权 {c1}", use_cls1=c1)
    sweep("组合：w=0.1 + 去重", w_glob=0.1, dedup_title=True)

    # 老人/新人分别看指标，确认改动只影响目标群体
    for s in ("2024-09-01",):
        dat = data[s]
        predict.g, predict.tcode = assets[s]
        sm = score_matrix(dat, "all")
        for tag, kw in (("基线", {}), ("w=0.1", {"w_glob": 0.1}), ("去重", {"dedup_title": True})):
            r = predict(dat, sm, **kw)
            print(f"  [{s}] {tag:<8} 新人Hit={r['new']['hit']:.4f} 老人Hit={r['old']['hit']:.4f} "
                  f"老人Rec={r['old']['recall']:.4f} NDCG={r['old']['ndcg']:.4f}")


if __name__ == "__main__":
    main()
