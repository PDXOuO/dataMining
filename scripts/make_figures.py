"""生成技术报告配图。

用法：python scripts/make_figures.py
输出：docs/figs/fig1_dept_cls.png、fig2_ablation.png、fig3_popularity.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.data_loader import load_newbooks, load_train, load_valid
from src.splits import build_backtest

FIG = ROOT / "docs" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, RED = "#2f5d8a", "#b5432f"


def fig_dept_cls(tr):
    """学院 × 中图法一级类目的借阅占比（行归一化）。"""
    tab = pd.crosstab(tr["dept"], tr["cls1"])
    top_d = tab.sum(axis=1).sort_values(ascending=False).head(10).index
    top_c = tab.sum(axis=0).sort_values(ascending=False).head(10).index
    sub = tab.loc[top_d, top_c]
    share = sub.div(sub.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    im = ax.imshow(share.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=0.75)
    ax.set_xticks(range(len(top_c)), top_c)
    ax.set_yticks(range(len(top_d)), top_d)
    ax.set_xlabel("中图法一级类目")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.4 else "#333")
    ax.set_title("学院—类目借阅偏好矩阵（行归一化，前 10 学院 × 前 10 类目）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_dept_cls.png", dpi=180)
    plt.close(fig)


def fig_ablation():
    df = pd.read_csv(ROOT / "docs" / "results.csv")
    order = ["全局热门 Top10", "同学院热门 Top10", "前代方案 C（热门池＋类目重排）",
             "物品级学院热度（λ=0，无提权）", "＋分层贝叶斯平滑（λ=1.5）",
             "＋同类目提权（k=1.6，关闭新书先验）"]
    labels = ["全局热门", "同学院热门", "前代方案 C", "学院物品热度", "+分层平滑", "+同类目提权"]
    piv = df.pivot_table(index="variant", columns="split", values="total").reindex(order)
    vals = piv.mean(axis=1).values

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.bar(labels, vals, color=[("#9aa5b1" if i < 3 else BLUE) for i in range(len(vals))])
    bars[-1].set_color(RED)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("四折平均总分（百分制）")
    ax.set_title("消融路径：从热门基线到定稿模型", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation.png", dpi=180)
    plt.close(fig)


def fig_popularity(tr):
    """答案书在全期热度上的分位分布（老人组）。"""
    task = build_backtest("2024-09-01")
    m = __import__("src.model_final", fromlist=["DeptBayesRecommender"]).DeptBayesRecommender()
    m.fit(task["hist"], task["newbooks"])
    g = m.global_score
    order = np.argsort(-g)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(g))
    pct = rank / max(len(g) - 1, 1)

    vals = []
    for rid, seq in task["truth"].items():
        if task["utype"][rid] != "老人":
            continue
        for c in seq:
            j = m.item_idx.get(c)
            if j is not None:
                vals.append(pct[j])
    vals = np.asarray(vals)

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.hist(vals, bins=20, color=BLUE, edgecolor="white")
    ax.axvline(0.1, color=RED, ls="--", lw=1.2)
    ax.text(0.11, ax.get_ylim()[1] * 0.85, f"前 10% 热度区间\n占 {np.mean(vals < 0.1):.1%}",
            color=RED, fontsize=9)
    ax.set_xlabel("答案图书在全期热度排序中的分位")
    ax.set_ylabel("答案条目数")
    ax.set_title("老人组答案的热度分布：正确答案高度集中于热门书", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_popularity.png", dpi=180)
    plt.close(fig)
    return float(np.mean(vals < 0.1))


def main():
    tr = load_train()
    fig_dept_cls(tr)
    fig_ablation()
    share = fig_popularity(tr)
    v = load_valid()
    nb = load_newbooks()
    print("图已生成：", sorted(p.name for p in FIG.glob("*.png")))
    print(f"老人答案位于热度前 10% 的比例：{share:.3f}")
    print(f"验证集 {len(v)} 人｜新人 {(v['utype'] == '新人').sum()}｜老人 {(v['utype'] == '老人').sum()}")
    print(f"训练集 {len(tr)} 条｜读者 {tr['reader'].nunique()}｜索书号 {tr['call_no'].nunique()}")
    print(f"新书池 {len(nb)} 册")


if __name__ == "__main__":
    main()
