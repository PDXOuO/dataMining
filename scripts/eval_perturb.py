"""扰动鲁棒性的离线实测：模拟决赛的扰动 A（画像缺失）与扰动 B（历史截断）。

决赛总分的 14% 来自鲁棒分 R = min(1, (扰动A + 扰动B) / 2 ÷ 15)，看的是扰动条件下的
**绝对成绩**。本地没有决赛答案，因此把两种情况施加到初赛四折的输入上，看分数掉多少、
掉在新人还是老人组，从而判断兜底路径是否有效。

用法：
    python scripts/eval_perturb.py                  # 四折，A 抽 30%、B 抽 10%
    python scripts/eval_perturb.py --frac-a 0.30 --frac-b 0.10 --seeds 3
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_final import SPLITS, bench_from_task  # noqa: E402
from src.final_data import Bench, hist_frame  # noqa: E402
from src.final_model import FinalRecommender  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.splits import build_backtest  # noqa: E402


def train_from_bench(bench: Bench) -> pd.DataFrame:
    """把输入历史摊平成训练帧，并补上 major_n（二级群体层要用）。"""
    df = hist_frame(bench)
    major = dict(zip(bench.readers["读者ID"], bench.readers["major_n"]))
    df["major_n"] = df["reader"].map(major)
    return df


def eval_bench(bench: Bench, truth: dict, utype: dict, params: dict) -> dict:
    """群体底座与输入历史同源（决赛即如此），避免两次计数的口径差。"""
    model = FinalRecommender(params).fit(bench, train_from_bench(bench))
    return evaluate(model.recommend_all(), truth, utype, verbose=False)


def mask_profiles(bench: Bench, frac: float, seed: int) -> Bench:
    """扰动 A：随机 frac 比例读者的画像字段抹为「未知」。"""
    rng = random.Random(seed)
    readers = bench.readers.copy()
    idx = rng.sample(range(len(readers)), int(round(frac * len(readers))))
    for col in ("学院", "专业", "年级"):
        readers.loc[idx, col] = "未知"
    for col in ("dept_n", "major_n", "grade_n"):
        readers.loc[idx, col] = "未知"
    return Bench(bench.name + "-A", readers, bench.whitelist, bench.calls,
                 bench.call_cls1, bench.call_cls2, bench.call_new, bench.hist)


def truncate_hist(bench: Bench, frac: float, seed: int) -> Bench:
    """扰动 B：随机 frac 比例读者（有历史者）的历史截断到 30%~70% 并打乱顺序。"""
    rng = random.Random(seed)
    cand = [r for r, seq in bench.hist.items() if len(seq) >= 3]
    pick = rng.sample(cand, min(int(round(frac * len(bench.readers))), len(cand)))
    hist = dict(bench.hist)
    for r in pick:
        seq = hist[r]
        keep = max(1, int(round(len(seq) * rng.uniform(0.3, 0.7))))
        sub = seq[:keep]
        rng.shuffle(sub)
        hist[r] = sub                                  # shuffle 后保持唯一性（原序列已去重）
    return Bench(bench.name + "-B", bench.readers.copy(), bench.whitelist, bench.calls,
                 bench.call_cls1, bench.call_cls2, bench.call_new, hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac-a", type=float, default=0.30)
    ap.add_argument("--frac-b", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for split in SPLITS:
        task = build_backtest(split)
        bench = bench_from_task(task)
        truth, utype = task["truth"], task["utype"]

        base = eval_bench(bench, truth, utype, {})
        a_runs = [eval_bench(mask_profiles(bench, args.frac_a, s), truth, utype, {})
                  for s in range(args.seeds)]
        b_runs = [eval_bench(truncate_hist(bench, args.frac_b, s), truth, utype, {})
                  for s in range(args.seeds)]
        def m(runs, key, sub=None):
            vals = [r[key] if sub is None else r[key][sub] for r in runs]
            return sum(vals) / len(vals)
        a = {"total": m(a_runs, "total"),
             "new": m(a_runs, "new", "score") * 100,
             "old": m(a_runs, "old", "score") * 100}
        b = {"total": m(b_runs, "total"),
             "new": m(b_runs, "new", "score") * 100,
             "old": m(b_runs, "old", "score") * 100}
        rows.append((split, base, a, b))
        print(f"{split}  原始 {base['total']:5.2f}(新 {base['new']['score']*100:5.2f} 老 {base['old']['score']*100:5.2f})  "
              f"A {a['total']:5.2f}(新 {a['new']:5.2f} 老 {a['old']:5.2f})  "
              f"B {b['total']:5.2f}(新 {b['new']:5.2f} 老 {b['old']:5.2f})")

    N = len(rows)
    m_base = sum(r[1]["total"] for r in rows) / N
    m_a = sum(r[2]["total"] for r in rows) / N
    m_b = sum(r[3]["total"] for r in rows) / N
    a_new = sum(r[2]["new"] for r in rows) / N
    b_new = sum(r[1]["new"]["score"] for r in rows) / N * 100
    a_old = sum(r[2]["old"] for r in rows) / N
    b_old = sum(r[1]["old"]["score"] for r in rows) / N * 100
    mean_ab = (m_a + m_b) / 2
    r_score = min(1.0, mean_ab / 15)
    print(f"\n均值  原始 {m_base:5.2f}  A {m_a:5.2f}  B {m_b:5.2f}  扰动均值 {mean_ab:5.2f}")
    print(f"A 分组  新人 {b_new:.2f} → {a_new:.2f} ({a_new - b_new:+.2f})   "
          f"老人 {b_old:.2f} → {a_old:.2f} ({a_old - b_old:+.2f})")
    print(f"鲁棒分 R = min(1, {mean_ab:.2f}/15) = {r_score:.2f}  → 贡献 {14 * r_score:.1f}/14 分")
    print(f"若隐藏榜与四折同水平：总分 ≈ {0.86 * m_base + 14 * r_score:.1f}"
          f"（0.30×公开 + 0.56×隐藏 均按 {m_base:.1f} 估）")


if __name__ == "__main__":
    main()
