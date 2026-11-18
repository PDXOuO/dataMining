"""模拟「旧底座 + 新历史」结构，用来标定 hist_weight。

决赛的群体信号有两个来源：初赛训练集停在 2025-01（旧、量大），
决赛输入历史是切分点前的最新行为（新、量小）。两者该按什么比例合并，
决赛上无法验证，只能把初赛训练集按时间切成「旧底座 + 近端窗口」来模拟：

    ts <  T - K 个月   →  群体底座（模拟训练集）
    T-K ≤ ts < T      →  群体计数里的新数据（模拟决赛输入历史）
    ts ≥ T            →  答案

个体历史仍给全量（决赛输入的历史就是该读者的完整历史），
只有「群体计数时新数据占多大权重」在变。

用法：python scripts/exp_final_recent.py --months 12 --weights 0,1,2,4,8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    cls_full_of_callno, cls_of_callno, load_train, normalize_dept, normalize_major,
)
from src.final_data import Bench  # noqa: E402
from src.final_model import FinalRecommender  # noqa: E402
from src.metrics import evaluate  # noqa: E402


def build(split: str, months: int):
    tr = load_train()
    T = pd.Timestamp(split)
    cut = T - pd.DateOffset(months=months)

    hist_all = tr[tr["ts"] < T]
    ans = tr[(tr["ts"] >= T) & (tr["ts"] < T + pd.DateOffset(months=14))]
    old = hist_all[hist_all["ts"] < cut].copy()
    recent = hist_all[hist_all["ts"] >= cut].copy()

    hist: dict[str, list[str]] = {}
    for rid, g in hist_all.groupby("reader", sort=False):
        seq, seen = [], set()
        for c in g.sort_values("ts")["call_no"]:
            if c not in seen:
                seen.add(c)
                seq.append(c)
        hist[rid] = seq

    truth, utype = {}, {}
    for rid, g in ans.groupby("reader", sort=False):
        seen = set(hist.get(rid, []))
        seq = [c for c in g.sort_values("ts")["call_no"].drop_duplicates() if c not in seen]
        if seq:
            truth[rid] = seq[:50]
            utype[rid] = "老人" if hist.get(rid) else "新人"

    prof = (
        tr.groupby("reader")
        .agg(dept=("dept", lambda s: s.mode().iat[0] if not s.mode().empty else "其他"),
             major=("major", lambda s: s.mode().iat[0] if not s.mode().empty else "未知"))
        .to_dict("index")
    )

    calls = sorted(set(hist_all["call_no"].dropna()))
    seen_all = set(hist_all["call_no"].dropna())
    # 新人没有历史，但仍要出现在待预测名单里（画像由 prof 提供）
    rids = list(hist) + [r for r in truth if r not in hist]
    readers = pd.DataFrame({
        "读者ID": rids,
        "dept_n": [normalize_dept(prof.get(r, {}).get("dept", "其他")) for r in rids],
        "major_n": [normalize_major(prof.get(r, {}).get("major", "未知")) for r in rids],
        "grade_n": ["未知"] * len(rids),
    })
    readers["is_old"] = [1 if hist.get(r) else 0 for r in rids]

    bench = Bench("sim", readers, pd.DataFrame({"索书号": calls}), calls,
                  [cls_of_callno(c) for c in calls],
                  [cls_full_of_callno(c) for c in calls],
                  [0 if c in seen_all else 1 for c in calls], hist)

    old = old.assign(dept_n=old["dept"].map(normalize_dept))
    recent = recent.assign(dept_n=recent["dept"].map(normalize_dept))
    new_group = recent[["reader", "call_no", "dept_n"]].rename(columns={"dept_n": "dept"})
    return bench, old, new_group, truth, utype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-09-01")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--weights", default="0,1,2,4,8")
    args = ap.parse_args()

    bench, old, new_group, truth, utype = build(args.split, args.months)
    print(f"切分 {args.split}  近端窗口 {args.months} 个月  "
          f"底座 {len(old)} 条 / 新数据 {len(new_group)} 条 / 读者 {len(bench.hist)} / 答案 {len(truth)} 人")

    for w in [float(x) for x in args.weights.split(",")]:
        m = FinalRecommender({"hist_weight": w}).fit(bench, old, hist_for_group=new_group)
        pred = m.recommend_all()
        r = evaluate(pred, truth, utype, verbose=False)
        print(f"  hist_weight={w:<5} 总 {r['total']:5.2f}  "
              f"新 {r['new']['score'] * 100:5.2f}  老 {r['old']['score'] * 100:5.2f}")


if __name__ == "__main__":
    main()
