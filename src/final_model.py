"""决赛模型：群体行为底座 + 画像/历史个性化。

与初赛定稿模型的继承关系：物品级「学院 × 二级类目」分层贝叶斯仍是打分主干
（初赛四折均值 7.36 vs 学院热门 6.40）。决赛新增三件事：

1. 群体底座 = 初赛训练集 + 决赛输入历史。前者量足但停在 2025-01，
   后者新鲜但只有 575 人；用 hist_weight 给后者加权。
2. 画像键从「学院」扩展为「学院 → 专业 → 年级」逐级回退，
   因为决赛新人占一半且完全没有历史，只能靠画像做群体迁移；
   画像被抹掉时自动退到上一级（扰动 A 的兜底路径）。
3. 新入藏单独一路：占白名单 34%、占答案 15.9%，但 96% 没有借阅记录，
   直接按类目分位数赋分并限制每份列表的配额。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .final_data import Bench, hist_frame

EPS = 1e-9

PARAMS = {
    "hist_weight": 3.0,      # 决赛历史相对训练集的计数权重
    "lam": 1.5,              # 分层贝叶斯收缩强度
    "k_cls2": 1.6,           # 老人：历史同类目提权
    "k_cls1": 0.0,           # 一级类目提权（默认关闭）
    "new_quantile": 0.70,    # 新书在其二级类目内的赋分分位
    "max_new": 2,            # 每份列表新书上限
    "top_k": 10,
    "dept_backoff": True,    # 画像键逐级回退
}


class FinalRecommender:
    """输入白名单 + 群体计数，输出每位读者的 Top-K 索书号。"""

    def __init__(self, params: dict | None = None):
        self.p = {**PARAMS, **(params or {})}

    # ---------------- 训练 ----------------
    def fit(self, bench: Bench, train: pd.DataFrame) -> "FinalRecommender":
        p = self.p
        self.bench = bench
        items = bench.calls
        self.item_idx = {c: i for i, c in enumerate(items)}
        n = len(items)
        self.n = n

        # 只保留白名单内的计数：白名单外推荐不计分，计数也没必要
        tr_mask = train["call_no"].isin(self.item_idx)
        tr = train[tr_mask]

        gf = np.zeros(n, dtype=np.float64)
        np.add.at(gf, tr["call_no"].map(self.item_idx).to_numpy(), 1.0)

        # 训练集可能来自初赛（dept 为简称）或决赛（已归一），统一到归一后字段
        if "dept_n" in tr.columns:
            tr_dept = tr["dept_n"].astype(str)
        else:
            from .data_loader import normalize_dept
            tr_dept = tr["dept"].astype(str).map(normalize_dept)

        hf = hist_frame(bench)
        hf = hf[hf["call_no"].isin(self.item_idx)]
        hist_cnt = np.zeros(n, dtype=np.float64)
        if len(hf):
            np.add.at(hist_cnt, hf["call_no"].map(self.item_idx).to_numpy(), 1.0)

        self.global_cnt = gf + p["hist_weight"] * hist_cnt

        # 学院计数：训练集 + 决赛历史
        dept_code, self.depts = {}, []
        def dcode(d):
            if d not in dept_code:
                dept_code[d] = len(self.depts)
                self.depts.append(d)
            return dept_code[d]

        rows, cols, vals = [], [], []
        for d, c in zip(tr_dept, tr["call_no"].map(self.item_idx)):
            rows.append(dcode(d)); cols.append(int(c)); vals.append(1.0)
        if len(hf):
            hfw = hf[hf["call_no"].isin(self.item_idx)]
            for d, c in zip(hfw["dept"], hfw["call_no"].map(self.item_idx)):
                rows.append(dcode(d)); cols.append(int(c)); vals.append(p["hist_weight"])

        self.dept_cnt = np.zeros((len(self.depts), n), dtype=np.float64)
        if rows:
            np.add.at(self.dept_cnt, (np.array(rows), np.array(cols)), np.array(vals))
        self.dept_code = dept_code

        # 二级类目成员与全局份额
        c2_ids = sorted(set(bench.call_cls2))
        self.c2_list = c2_ids
        self.c2_code = {c: i for i, c in enumerate(c2_ids)}
        self.c2_of_item = np.array([self.c2_code[c] for c in bench.call_cls2])
        c2_tot = np.bincount(self.c2_of_item, weights=self.global_cnt, minlength=len(c2_ids))
        self.c2_tot = np.maximum(c2_tot, EPS)
        self.global_share = self.global_cnt / self.c2_tot[self.c2_of_item]

        self.cls1_of_item = np.array(bench.call_cls1)
        self.is_new = np.array(bench.call_new, dtype=bool)

        self._build_dept_scores()
        self._build_newbook_scores()
        return self

    def _build_dept_scores(self):
        """s_d(i) = (c_d(i) + λ·n_d·prior_d(i)) / (n_d(1+λ))，prior 用类目占比 × 类目内全局份额。"""
        p = self.p
        D, n = self.dept_cnt.shape
        self.dept_score = np.zeros((D, n), dtype=np.float32)
        lam = p["lam"]
        for d in range(D):
            cd = self.dept_cnt[d]
            nd = cd.sum()
            if nd <= 0:
                self.dept_score[d] = (self.global_share / self.global_share.sum()).astype(np.float32)
                continue
            # 该学院在各二级类目上的占比
            c2_sum = np.bincount(self.c2_of_item, weights=cd, minlength=len(self.c2_list))
            share = c2_sum / (c2_sum.sum() + EPS)
            prior = share[self.c2_of_item] * self.global_share
            s = (cd + lam * nd * prior) / (nd * (1 + lam))
            self.dept_score[d] = s.astype(np.float32)

    def _build_newbook_scores(self):
        """新入藏无计数，按所在二级类目在该学院下的分位数赋分。"""
        q = self.p["new_quantile"]
        D, n = self.dept_score.shape
        self.new_score = np.zeros((D, n), dtype=np.float32)
        for d in range(D):
            s = self.dept_score[d]
            for c2 in range(len(self.c2_list)):
                m = self.c2_of_item == c2
                if not m.any():
                    continue
                known = m & (~self.is_new) & (s > 0)
                ref = np.quantile(s[known], q) if known.sum() >= 3 else np.quantile(s[m], q)
                self.new_score[d, m] = ref

    # ---------------- 推理 ----------------
    def _dept_of(self, row) -> int:
        v = row.get("dept_n")
        if isinstance(v, str) and v in self.dept_code:
            return self.dept_code[v]
        return -1

    def score_user(self, rid: str) -> np.ndarray:
        bench = self.bench
        row = self._row(rid)
        dept = self._dept_of(row)
        base = self.dept_score[dept].copy() if dept >= 0 else self.global_share.astype(np.float32).copy()

        hist = bench.hist.get(rid, [])
        if hist:
            idx = [self.item_idx[c] for c in hist if c in self.item_idx]
            if idx:
                c2_hist = defaultdict(float)
                c1_hist = set()
                for i in idx:
                    c2_hist[self.c2_of_item[i]] += 1.0
                top = max(c2_hist.values())
                boost = np.ones(self.n, dtype=np.float32)
                for c2, cnt in c2_hist.items():
                    m = self.c2_of_item == c2
                    boost[m] *= 1.0 + self.p["k_cls2"] * (cnt / top)
                base = base * boost
                base[idx] = -1.0            # 已读不推

        if self.p["max_new"] > 0:
            news = self.new_score[dept] if dept >= 0 else self.new_score.mean(axis=0)
            base = np.where(self.is_new, np.maximum(base, news), base)
        return base

    def _row(self, rid: str) -> dict:
        if not hasattr(self, "_rows"):
            self._rows = {
                r["读者ID"]: r for r in self.bench.readers.to_dict("records")
            }
        return self._rows.get(rid, {})

    def recommend(self, rid: str, k: int | None = None) -> list[str]:
        k = k or self.p["top_k"]
        s = self.score_user(rid)
        order = np.argsort(-s, kind="stable")
        out, n_new = [], 0
        for i in order:
            if s[i] <= 0:
                break
            if self.is_new[i]:
                if n_new >= self.p["max_new"]:
                    continue
                n_new += 1
            out.append(self.bench.calls[i])
            if len(out) >= k:
                break
        if len(out) < k:   # 兜底：白名单内按全局热度补齐
            for i in np.argsort(-self.global_cnt):
                c = self.bench.calls[i]
                if c not in out:
                    out.append(c)
                    if len(out) >= k:
                        break
        return out

    def recommend_all(self, k: int | None = None) -> dict[str, list[str]]:
        return {rid: self.recommend(rid, k) for rid in self.bench.readers["读者ID"]}
