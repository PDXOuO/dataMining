"""生成技术报告用的对照表：四折 × 全部变体，输出 Markdown。

用法：python scripts/report_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate
from src.model_final import DeptBayesRecommender
from src.recommender import HybridRecommender
from src.splits import build_backtest

SPLITS = ["2023-09-01", "2024-03-01", "2024-06-01", "2024-09-01"]
OUT = ROOT / "docs" / "results.md"


def simple_baseline(task, model, by_dept: bool):
    """全局/同学院热门基线：直接按热度排序取前 10，剔除已借。"""
    pred = {}
    for rid in task["truth"]:
        prof = task["profile"].get(rid, {})
        calls = [c for c in (task["hist_by_reader"].get(rid).call_no.tolist()
                             if rid in task["hist_by_reader"] else [])]
        if by_dept:
            d = model.dept_code.get(prof.get("dept", "其他"))
            score = model.dept_score[d] if d is not None else model.global_score
        else:
            score = model.global_score
        s = score.copy()
        for c in calls:
            j = model.item_idx.get(c)
            if j is not None:
                s[j] = -1.0
        order = np.argsort(-s)[:10]
        pred[rid] = [model.items[j] for j in order if s[j] > 0]
    return pred


def main():
    rows = {k: [] for k in ("total",)}
    table = []
    for split in SPLITS:
        task = build_backtest(split)
        readers = list(task["truth"])
        hm = {r: df["call_no"].tolist() for r, df in task["hist_by_reader"].items()}
        prof = task["profile"]
        uty = task["utype"]

        base = DeptBayesRecommender({"lam": 0.0, "k_cls2": 0.0, "newbook_quantile": 0.0})
        base.fit(task["hist"], task["newbooks"])

        variants = []
        variants.append(("全局热门 Top10", simple_baseline(task, base, False)))
        variants.append(("同学院热门 Top10", simple_baseline(task, base, True)))

        m = DeptBayesRecommender({"lam": 0.0, "k_cls2": 0.0, "newbook_quantile": 0.70})
        m.fit(task["hist"], task["newbooks"])
        variants.append(("物品级学院热度（λ=0，无提权）", m.recommend_all(readers, prof, hm)))

        m = DeptBayesRecommender({"lam": 1.5, "k_cls2": 0.0, "newbook_quantile": 0.70})
        m.fit(task["hist"], task["newbooks"])
        variants.append(("＋分层贝叶斯平滑（λ=1.5）", m.recommend_all(readers, prof, hm)))

        m = DeptBayesRecommender({"lam": 1.5, "k_cls2": 1.6, "newbook_quantile": 0.0})
        m.fit(task["hist"], task["newbooks"])
        variants.append(("＋同类目提权（k=1.6，关闭新书先验）", m.recommend_all(readers, prof, hm)))

        m = DeptBayesRecommender()
        m.fit(task["hist"], task["newbooks"])
        variants.append(("完整模型", m.recommend_all(readers, prof, hm)))

        # 前代方案 C（热门候选池 + 类目重排）
        mc = HybridRecommender({
            "top_n": 1000, "beta_pop": 0.40, "gamma_cls": 0.80, "w_self": 0.70,
            "max_per_cls_new": 6, "max_per_cls_old": 4,
        }).fit(task["hist"], task["newbooks"])
        variants.append(("前代方案 C（热门池＋类目重排）",
                         mc.recommend_all(readers, prof, hm, uty)))

        for tag, pred in variants:
            r = evaluate(pred, task["truth"], uty, verbose=False)
            table.append({
                "split": split, "variant": tag,
                "new_hit": r["new"]["hit"], "new_recall": r["new"]["recall"],
                "old_hit": r["old"]["hit"], "old_recall": r["old"]["recall"],
                "old_ndcg": r["old"]["ndcg"], "total": r["total"],
            })
        print(f"{split} 完成：{len(table)} 行")

    df = pd.DataFrame(table)
    df.to_csv(ROOT / "docs" / "results.csv", index=False, encoding="utf-8-sig")

    pivot = df.pivot_table(index="variant", columns="split", values="total")
    pivot["均值"] = pivot.mean(axis=1)
    order = [
        "全局热门 Top10", "同学院热门 Top10", "前代方案 C（热门池＋类目重排）",
        "物品级学院热度（λ=0，无提权）", "＋分层贝叶斯平滑（λ=1.5）",
        "＋同类目提权（k=1.6，关闭新书先验）", "完整模型",
    ]
    pivot = pivot.reindex([o for o in order if o in pivot.index])

    lines = ["| 变体 | " + " | ".join(SPLITS) + " | 四折均值 |",
             "|---|" + "---|" * (len(SPLITS) + 1)]
    for name, r in pivot.iterrows():
        cells = " | ".join(f"{r[s]:.2f}" for s in SPLITS)
        lines.append(f"| {name} | {cells} | **{r['均值']:.2f}** |")
    md = "\n".join(lines)
    OUT.write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
