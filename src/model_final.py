"""定稿模型：院校-类目分层贝叶斯物品热度 + 历史同类目提权。

回测否定了前几版的核心假设，最终收敛到这一版：

1. 物品级学院热度 > 类目级学院分布。先前用「学院类目分布 × 全局热度」重排，
   4 折均值 4.5 左右；直接统计学院内各索书号的借阅次数即达 6.6。
   原因：学院内部的类目偏好已隐含在物品计数里，先降维到类目再乘全局热度，
   等于把学院信号压缩掉，只留下全局热度。
2. 分层平滑优于向全局平滑。物品计数 → 二级类目（中图法完整分类号）在学院内的
   占比 × 该类目内全局份额，作为贝叶斯先验回填，4 折均值 7.36（λ=1.5）。
3. 学院全期计数优于近一年窗口（6.77 vs 5.05）。本数据量下，一年窗口的计数过于
   稀疏，噪声盖过了时效收益。
4. 老人侧只用「同二级类目」提权，不用精确分类号、作者、共现。
   实测答案同二级类目仅 19.6%、同作者 1.2%，而同一级类目 70.3%。

新书处理无法回测（新书池最早入藏 2025-01-06，晚于所有回测切分点），
采用保守先验：新书按其所在二级类目的分位数赋分，与老书公平竞争，
不预留固定名额。依据是 956 人历史借阅中仅 7.3% 命中新书池，
说明新书并非借阅主力。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from .data_loader import cls_full_of_callno, cls_of_callno
except ImportError:  # 允许脚本直接导入
    from src.data_loader import cls_full_of_callno, cls_of_callno

EPS = 1e-9

PARAMS = {
    "lam": 1.5,              # 向「院系类目占比 × 类目内全局份额」的平滑强度
    "k_cls2": 1.6,           # 老人：同二级类目提权系数
    "newbook_quantile": 0.70,  # 新书赋分位（在其二级类目内的分位）
    "max_new_per_list": 2,   # 每份推荐列表中新书的数量上限
    "top_k": 10,
}


class DeptBayesRecommender:
    def __init__(self, params: dict | None = None):
        self.p = {**PARAMS, **(params or {})}
        self.items: list[str] = []
        self.item_idx: dict[str, int] = {}
        self.cls1: list[str] = []
        self.cls2: list[str] = []
        self.cls2_code: np.ndarray | None = None
        self.has_count: np.ndarray | None = None
        self.dept_score: dict[int, np.ndarray] = {}
        self.global_score: np.ndarray | None = None
        self.dept_code: dict[str, int] = {}

    # ---------------- 训练 ----------------
    def fit(self, hist: pd.DataFrame, newbooks: pd.DataFrame) -> "DeptBayesRecommender":
        p = self.p
        nb_calls = set(newbooks["call_no"].dropna()) if "call_no" in newbooks else set()
        self.items = sorted(set(hist["call_no"].dropna()) | nb_calls)
        self.item_idx = {c: i for i, c in enumerate(self.items)}
        n = len(self.items)

        # 分类号：优先取借阅记录里的，其次从索书号解析
        c1_map, c2_map = {}, {}
        if "cls1" in hist:
            c1_map = hist.drop_duplicates("call_no").set_index("call_no")["cls1"].to_dict()
        if "cls2" in hist:
            c2_map = hist.drop_duplicates("call_no").set_index("call_no")["cls2"].to_dict()
        if "cls1" in newbooks:
            for c, v in zip(newbooks["call_no"], newbooks["cls1"]):
                c1_map.setdefault(c, v)
        if "cls2" in newbooks:
            for c, v in zip(newbooks["call_no"], newbooks["cls2"]):
                c2_map.setdefault(c, v)
        self.cls1 = [c1_map.get(c) or cls_of_callno(c) for c in self.items]
        self.cls2 = [c2_map.get(c) or cls_full_of_callno(c) for c in self.items]

        # 二级类目编码与成员指示
        codes: dict[str, int] = {}
        arr = []
        for c in self.cls2:
            if c not in codes:
                codes[c] = len(codes)
            arr.append(codes[c])
        self.cls2_code = np.asarray(arr)
        n_c2 = len(codes)
        member = np.zeros((n, n_c2), dtype=np.float64)
        member[np.arange(n), self.cls2_code] = 1.0

        # 院系 × 物品计数
        h = hist[hist["call_no"].isin(set(self.items))]
        jj = h["call_no"].map(self.item_idx)
        ok = jj.notna().values
        jarr = jj[ok].astype(int).values
        depts = sorted(set(h["dept"].dropna().values))
        self.dept_code = {d: i for i, d in enumerate(depts)}
        darr = np.asarray([self.dept_code[d] for d in h["dept"].values[ok]])

        d_cnt = np.zeros((len(depts), n), dtype=np.float64)
        np.add.at(d_cnt, (darr, jarr), 1.0)
        d_n = d_cnt.sum(axis=1)
        g_cnt = d_cnt.sum(axis=0)
        self.has_count = g_cnt > 0
        self.is_new = ~self.has_count

        # 分层先验：院系内二级类目占比 × 类目内全局份额
        g_c2 = g_cnt @ member
        g_c2_share = g_c2 / (g_c2.sum() + EPS)
        within = g_cnt / (np.asarray(g_c2 @ member.T).ravel() + EPS)
        d_c2 = d_cnt @ member
        d_c2_share = d_c2 / (d_c2.sum(axis=1, keepdims=True) + EPS)

        self.global_score = self._finalize(g_cnt / (g_cnt.sum() + EPS), member)
        for name, d in self.dept_code.items():
            prior = d_c2_share[d][self.cls2_code] * within
            cnt = d_cnt[d] + p["lam"] * d_n[d] * prior
            self.dept_score[d] = self._finalize(cnt / (cnt.sum() + EPS), member)
        self._member = member
        return self

    def _finalize(self, score: np.ndarray, member: np.ndarray) -> np.ndarray:
        """给无借阅记录的新书赋分：取所在二级类目内有记录图书的分位数。"""
        q = float(self.p["newbook_quantile"])
        score = score.copy()
        if q <= 0:
            return score
        for c in range(member.shape[1]):
            js = np.flatnonzero(member[:, c] > 0)
            if not len(js):
                continue
            known = js[self.has_count[js]] if self.has_count is not None else js
            if not len(known):
                known = js
            base = score[known]
            fill = np.quantile(base, q) if len(base) else 0.0
            if fill <= 0:
                continue
            empty = js[~self.has_count[js]] if self.has_count is not None else js
            if len(empty):
                score[empty] = np.maximum(score[empty], fill)
        return score

    # ---------------- 推荐 ----------------
    def score_for(self, dept: str, calls: list[str]) -> np.ndarray:
        d = self.dept_code.get(dept)
        s = self.dept_score[d] if d is not None else self.global_score
        s = s.copy()
        if calls:
            seen_c2 = {self.cls2[self.item_idx[c]] for c in calls if c in self.item_idx}
            if seen_c2:
                boost = np.asarray([1.0 if c in seen_c2 else 0.0 for c in self.cls2])
                s = s * (1.0 + float(self.p["k_cls2"]) * boost)
            idx = [self.item_idx[c] for c in calls if c in self.item_idx]
            if idx:
                s[idx] = -1.0
        return s

    def recommend_all(
        self,
        readers: list[str],
        profiles: dict[str, dict],
        hist_map: dict[str, list[str]],
        k: int | None = None,
    ) -> dict[str, list[str]]:
        k = k or int(self.p["top_k"])
        out: dict[str, list[str]] = {}
        for rid in readers:
            prof = profiles.get(rid, {})
            calls = hist_map.get(rid) or []
            s = self.score_for(prof.get("dept", "其他"), calls)
            order = np.argsort(-s)[: k * 6]
            cap_new = int(self.p.get("max_new_per_list", 10))
            picked, n_new = [], 0
            for j in order:
                if len(picked) >= k:
                    break
                if s[j] <= 0:
                    continue
                if self.is_new[j]:
                    if n_new >= cap_new:
                        continue
                    n_new += 1
                picked.append(self.items[j])
            if len(picked) < k:
                for j in order:
                    c = self.items[j]
                    if c not in picked and s[j] > 0:
                        picked.append(c)
                    if len(picked) >= k:
                        break
            out[rid] = picked[:k]
        return out
