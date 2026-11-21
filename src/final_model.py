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
    # 以下三项是 v2 定稿值：不传参数直接运行即可复现提交文件，
    # 复现审查要求「输出与提交文件逐行一致」，任何调参都必须先改这里。
    "use_major": 1,          # 是否用「学院×专业」二级群体（0 关闭）
    "lam_major": 1.5,        # 专业层向学院层回退的收缩强度
    "min_major_n": 100,      # 专业层最少样本，低于此直接回退学院
    # v3 待验证的两项（默认值 = 不改行为，便于单独消融）
    # v3 定稿：类目偏好比开四次方根。历史中位仅 3 本，频次比 (cnt/top)
    # 在短历史上噪声很大，开方后提权更平缓；四折 8.40 → 8.49，四折全升。
    "boost_pow": 0.4,
    "len_shrink": 0.0,       # 历史过短时按 len/(len+s) 削弱提权强度（实测无增益，关闭）
    # 群体计数的时间衰减：四折 +0.15、自监督 +0.21，但「旧底座+新历史」模拟
    # 三折全跌 0.35~0.38（与 hist_weight 叠加后重复惩罚旧数据），证据冲突，不启用。
    "recency_tau": 0.0,
}


class FinalRecommender:
    """输入白名单 + 群体计数，输出每位读者的 Top-K 索书号。"""

    def __init__(self, params: dict | None = None):
        self.p = {**PARAMS, **(params or {})}

    # ---------------- 训练 ----------------
    def fit(self, bench: Bench, train: pd.DataFrame,
            hist_for_group: pd.DataFrame | None = None) -> "FinalRecommender":
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
        # 群体计数的时间衰减：决赛答案窗口距训练集末端约 20 个月，
        # 越旧的借阅越可能反映已过时的阅读倾向（τ=0 时退化为不加权）
        if p["recency_tau"] > 0 and "ts" in tr.columns:
            days = (tr["ts"].max() - tr["ts"]).dt.days.to_numpy()
            w_tr = np.exp(-days / p["recency_tau"])
        else:
            w_tr = np.ones(len(tr), dtype=np.float64)
        np.add.at(gf, tr["call_no"].map(self.item_idx).to_numpy(), w_tr)

        # 训练集可能来自初赛（dept 为简称）或决赛（已归一），统一到归一后字段
        if "dept_n" in tr.columns:
            tr_dept = tr["dept_n"].astype(str)
        else:
            from .data_loader import normalize_dept
            tr_dept = tr["dept"].astype(str).map(normalize_dept)

        hf = hist_for_group if hist_for_group is not None else hist_frame(bench)
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
        for d, c, wv in zip(tr_dept, tr["call_no"].map(self.item_idx), w_tr):
            rows.append(dcode(d)); cols.append(int(c)); vals.append(float(wv))
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
        self._build_group_counts(tr, tr_dept, hf, w_tr)
        return self

    def _build_group_counts(self, tr: pd.DataFrame, tr_dept: pd.Series,
                            hf: pd.DataFrame, w_tr: np.ndarray | None = None) -> None:
        """「学院×专业」二级群体计数。决赛新人占一半，专业往往比学院更有分辨力；
        样本不足时在评分阶段回退到学院层，不做硬分组。"""
        self.group_cnt: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        if not self.p["use_major"]:
            return
        from .data_loader import normalize_major

        if "major_n" in tr.columns:
            tr_major = tr["major_n"].astype(str)
        else:
            tr_major = tr["major"].astype(str).map(normalize_major)

        w_tr = np.ones(len(tr), dtype=np.float64) if w_tr is None else w_tr
        idx = tr["call_no"].map(self.item_idx)
        df = pd.DataFrame({"d": tr_dept.values, "m": tr_major.values,
                           "i": idx.values, "w": w_tr}).dropna(subset=["i"])
        frames = [df]
        if len(hf):
            hm = self.bench.readers.set_index("读者ID")["major_n"].to_dict()
            hd = pd.DataFrame({
                "d": hf["dept"].values,
                "m": [hm.get(r, "未知") for r in hf["reader"]],
                "i": hf["call_no"].map(self.item_idx).values,
                "w": 1.0,   # 注意：专业层历史按 1 计，不乘 hist_weight（实测加权会更差）
            }).dropna(subset=["i"])
            frames.append(hd)
        df = pd.concat(frames, ignore_index=True)
        df = df[~df["m"].isin(["未知", "nan", ""])]

        for (d, m), g in df.groupby(["d", "m"], sort=False):
            agg = g.groupby("i")["w"].sum()
            self.group_cnt[(d, m)] = (agg.index.to_numpy(), agg.to_numpy(dtype=np.float64))

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
        """新入藏无计数，按所在二级类目在该学院下的分位数赋分。

        实现上按「类目分组、学院成列」批量分位（一次排序出 23 个学院的分位值），
        避免逐 (学院, 类目) 调 np.quantile —— 那样是 15 万次调用，实测要 20 秒。
        """
        q = self.p["new_quantile"]
        D, n = self.dept_score.shape
        S = self.dept_score.T.astype(np.float64)          # n × D
        out = np.zeros((n, D), dtype=np.float32)

        order = np.argsort(self.c2_of_item, kind="stable")
        sizes = np.bincount(self.c2_of_item, minlength=len(self.c2_list))
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        for g in range(len(self.c2_list)):
            idx = order[bounds[g]:bounds[g + 1]]
            if idx.size == 0:
                continue
            known = ~self.is_new[idx]
            sub = S[idx][known] if known.sum() >= 3 else S[idx]
            sub = np.sort(sub, axis=0)
            k = min(int(q * (sub.shape[0] - 1)), sub.shape[0] - 1)
            out[idx, :] = sub[k][None, :]
        self.new_score = out.T.astype(np.float32)

    # ---------------- 推理 ----------------
    def _dept_of(self, row) -> int:
        v = row.get("dept_n")
        if isinstance(v, str) and v in self.dept_code:
            return self.dept_code[v]
        return -1

    def _group_score(self, dept: int, row) -> np.ndarray | None:
        """专业层分数，向学院层收缩；样本不足或专业缺失返回 None 走学院层。"""
        if dept < 0 or not self.group_cnt:
            return None
        major = row.get("major_n")
        if not isinstance(major, str) or major in ("未知", ""):
            return None
        got = self.group_cnt.get((self.depts[dept], major))
        if got is None:
            return None
        idx, cnt = got
        ng = cnt.sum()
        if ng < self.p["min_major_n"]:
            return None
        arr = np.zeros(self.n, dtype=np.float32)
        arr[idx] = cnt
        lam = self.p["lam_major"]
        sd = self.dept_score[dept]
        return ((arr + lam * ng * sd) / (ng * (1 + lam))).astype(np.float32)

    def score_user(self, rid: str) -> np.ndarray:
        bench = self.bench
        row = self._row(rid)
        dept = self._dept_of(row)
        if dept >= 0:
            base = self.dept_score[dept].copy()
            gs = self._group_score(dept, row)
            if gs is not None:
                base = gs
        else:
            base = self.global_share.astype(np.float32).copy()

        hist = bench.hist.get(rid, [])
        if hist:
            idx = [self.item_idx[c] for c in hist if c in self.item_idx]
            if idx:
                c2_hist = defaultdict(float)
                c1_hist = defaultdict(float)
                for i in idx:
                    c2_hist[self.c2_of_item[i]] += 1.0
                    c1_hist[self.cls1_of_item[i]] += 1.0
                boost = np.ones(self.n, dtype=np.float32)
                top = max(c2_hist.values())
                k = self.p["k_cls2"]
                if self.p["len_shrink"] > 0:
                    # 历史只有一两本时，类目偏好本身就是噪声，按样本量打折
                    L = len(idx)
                    k = k * L / (L + self.p["len_shrink"])
                for c2, cnt in c2_hist.items():
                    m = self.c2_of_item == c2
                    boost[m] *= 1.0 + k * (cnt / top) ** self.p["boost_pow"]
                if self.p["k_cls1"] > 0:
                    top1 = max(c1_hist.values())
                    for c1, cnt in c1_hist.items():
                        m = self.cls1_of_item == c1
                        boost[m] *= 1.0 + self.p["k_cls1"] * (cnt / top1)
                base = base * boost
                base[idx] = -1.0            # 已读不推

        if self.p["max_new"] > 0:
            news = self.new_score[dept] if dept >= 0 else self.new_score.mean(axis=0)
            base = np.where(self.is_new, np.maximum(base, news), base)
            # 新书赋分会把已读标记的 -1 抬回正值（新入藏且已被借过的书就是这种情况），
            # 手册明确答案已剔除历史书目，必须再压一次，否则老人组等于 0 命中。
            if hist:
                base[[self.item_idx[c] for c in hist if c in self.item_idx]] = -1.0
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
        if len(out) < k:   # 兜底：白名单内按全局热度补齐（仍不得复推已读）
            seen = set(self.bench.hist.get(rid, []))
            for i in np.argsort(-self.global_cnt):
                c = self.bench.calls[i]
                if c not in out and c not in seen:
                    out.append(c)
                    if len(out) >= k:
                        break
        return out

    def recommend_all(self, k: int | None = None) -> dict[str, list[str]]:
        return {rid: self.recommend(rid, k) for rid in self.bench.readers["读者ID"]}
