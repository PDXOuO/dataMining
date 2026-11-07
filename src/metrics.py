"""评测指标，严格对齐组委会《验证集说明》4.2/4.3 节。

- 新人组 = 0.5 * HitRate@10 + 0.5 * Recall@10
- 老人组 = 0.4 * HitRate@10 + 0.3 * Recall@10 + 0.3 * NDCG@10
- 总分   = (0.4 * 新人组 + 0.6 * 老人组) * 100
所有组内指标均为宏平均（对组内全部读者取算术平均）。
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .config import (
    TOP_K,
    W_NEW_GROUP,
    W_NEW_HIT,
    W_NEW_RECALL,
    W_OLD_GROUP,
    W_OLD_HIT,
    W_OLD_NDCG,
    W_OLD_RECALL,
)


def _dedup(seq: Iterable[str], k: int = TOP_K) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= k:
            break
    return out


def per_user(pred: Sequence[str], truth: Sequence[str], k: int = TOP_K) -> tuple[float, float, float]:
    """返回 (hit, recall, ndcg)。"""
    p = _dedup(pred, k)
    g = set(truth)
    if not g:
        return 0.0, 0.0, 0.0
    hits = [i for i, x in enumerate(p, start=1) if x in g]
    h = len(hits)
    hit = 1.0 if h else 0.0
    recall = h / len(g)
    if h:
        dcg = sum(1.0 / math.log2(i + 1) for i in hits)
        ideal_n = min(len(g), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
        ndcg = dcg / idcg if idcg else 0.0
    else:
        ndcg = 0.0
    return hit, recall, ndcg


def evaluate(
    pred: Mapping[str, Sequence[str]],
    truth: Mapping[str, Sequence[str]],
    utype: Mapping[str, str],
    verbose: bool = True,
) -> dict:
    """pred/truth/utype 均以读者 ID 为键。缺失的预测按 0 命中计。"""
    buckets = {"新人": [], "老人": []}
    for rid, ut in utype.items():
        g = truth.get(rid)
        if not g:
            continue
        buckets.setdefault(ut, []).append(per_user(pred.get(rid, []), g))

    def agg(rows):
        if not rows:
            return {"n": 0, "hit": 0.0, "recall": 0.0, "ndcg": 0.0}
        n = len(rows)
        return {
            "n": n,
            "hit": sum(r[0] for r in rows) / n,
            "recall": sum(r[1] for r in rows) / n,
            "ndcg": sum(r[2] for r in rows) / n,
        }

    new, old = agg(buckets.get("新人", [])), agg(buckets.get("老人", []))
    s_new = W_NEW_HIT * new["hit"] + W_NEW_RECALL * new["recall"]
    s_old = (
        W_OLD_HIT * old["hit"] + W_OLD_RECALL * old["recall"] + W_OLD_NDCG * old["ndcg"]
    )
    total = (W_NEW_GROUP * s_new + W_OLD_GROUP * s_old) * 100

    res = {
        "total": total,
        "new": {**new, "score": s_new},
        "old": {**old, "score": s_old},
    }
    if verbose:
        print(
            f"[新人 n={new['n']:4d}] Hit={new['hit']:.4f} Recall={new['recall']:.4f} "
            f"-> {s_new:.4f}\n"
            f"[老人 n={old['n']:4d}] Hit={old['hit']:.4f} Recall={old['recall']:.4f} "
            f"NDCG={old['ndcg']:.4f} -> {s_old:.4f}\n"
            f"[总分] {total:.2f}"
        )
    return res
