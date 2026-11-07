"""方案 C：物品共现 + 内容相似 + 分群热门 三路召回，加权 RRF 融合。

设计要点：
- 不使用任何用户级协同过滤——验证集读者与训练集读者 ID 无法对齐；
- 共现图由训练集与验证集历史列表合并构建，物品侧信息可跨集合共享；
- 融合用 RRF，因为没有任何标注答案可用于训练融合权重。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from .config import DEFAULTS, TOP_K
from .data_loader import cls_of_callno


def _l2_normalize_rows(m: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m


class HybridRecommender:
    def __init__(self, params: dict | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.items: list[str] = []
        self.item_idx: dict[str, int] = {}
        self.pop = None
        self.sim = None
        self.cls_sim = None
        self.cls2_sim = None
        self.auth_sim = None
        self.dept_vec: dict[str, np.ndarray] = {}
        self.major_vec: dict[str, np.ndarray] = {}
        self.major_n: dict[str, int] = {}
        self.global_vec = None
        self.cls_of: dict[str, str] = {}
        self.new_weight: dict[str, float] = {}

    # ---------- 训练 ----------
    def fit(self, hist: pd.DataFrame, newbooks: pd.DataFrame, extra_hist: dict[str, list[str]] | None = None):
        p = self.p
        rows, cols, vals = [], [], []
        reader_ids = {r: i for i, r in enumerate(hist["reader"].unique())}
        self.items = sorted(set(hist["call_no"].dropna()) | set(newbooks["call_no"]))
        self.item_idx = {c: i for i, c in enumerate(self.items)}
        n_items = len(self.items)

        # 交互矩阵（训练集）
        for r, c in zip(hist["reader"].map(reader_ids), hist["call_no"]):
            j = self.item_idx.get(c)
            if j is not None:
                rows.append(int(r))
                cols.append(j)
                vals.append(1.0)
        n_reader = len(reader_ids)

        # 验证集历史列表并入共现图（额外读者编号从 n_reader 起）
        if extra_hist:
            base = n_reader
            for k, (rid, calls) in enumerate(extra_hist.items()):
                for c in calls:
                    j = self.item_idx.get(c)
                    if j is not None:
                        rows.append(base + k)
                        cols.append(j)
                        vals.append(1.0)
            n_reader = base + len(extra_hist)

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_reader, n_items))
        X.data[:] = 1.0  # 二值化，避免复借放大
        X.sum_duplicates()
        X.data[:] = 1.0

        self.pop = np.asarray(X.sum(axis=0)).ravel()
        self._build_cf(X)
        self._build_content(hist, newbooks)
        self._build_group(hist)
        self._build_new_weight(newbooks)
        self._prepare_quota(hist)
        self.X = X
        return self

    def _build_cf(self, X: sparse.csr_matrix):
        p = self.p
        co = (X.T @ X).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        pop = np.maximum(self.pop, 1.0)
        # 余弦 × 热门惩罚：(i,j) 共现 / (pop_i^0.5 pop_j^0.5) / (pop_i^a pop_j^a)
        expo = 0.5 + float(p["alpha_pop"])
        d = np.power(pop, -expo)
        D = sparse.diags(d)
        sim = (D @ co @ D).tocsr()
        # 每行截断，控制内存与噪声
        k = int(p.get("k_neighbors", 200))
        sim = self._topk_rows(sim, k)
        self.sim = sim

    @staticmethod
    def _topk_rows(m: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        m = m.tocsr()
        indptr, indices, data = [0], [], []
        for i in range(m.shape[0]):
            a, b = m.indptr[i], m.indptr[i + 1]
            if a == b:
                indptr.append(len(indices))
                continue
            row = m.data[a:b]
            col = m.indices[a:b]
            if len(row) > k:
                sel = np.argpartition(-row, k)[:k]
                row, col = row[sel], col[sel]
            order = np.argsort(-row)
            indices.extend(col[order].tolist())
            data.extend(row[order].tolist())
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=m.shape,
        )

    def _build_content(self, hist: pd.DataFrame, newbooks: pd.DataFrame):
        cols = ["cls1", "cls2", "author"]
        info = hist[["call_no"] + cols].drop_duplicates("call_no").set_index("call_no")
        nb = newbooks.set_index("call_no")[cols] if not newbooks.empty else pd.DataFrame(columns=cols)
        info = info.combine_first(nb).reindex(self.items)

        cls1 = info["cls1"].fillna("?").values
        cls2 = info["cls2"].fillna("?").values
        auth = info["author"].fillna("").values
        self.cls_of = dict(zip(self.items, cls1))
        self.cls2_of = dict(zip(self.items, cls2))

        n = len(self.items)
        self.cls_sim = self._onehot_sim(cls1, "?")
        self.cls2_sim = self._onehot_sim(cls2, "?")
        self.auth_sim = self._onehot_sim(np.where(auth == "", "__na__", auth), "__na__")

    @staticmethod
    def _onehot_sim(values: np.ndarray, na_token: str) -> sparse.csr_matrix:
        codes, uniq = pd.factorize(values)
        na_idx = {i for i, u in enumerate(uniq) if u == na_token}
        m = sparse.csr_matrix(
            (np.ones(len(values)), (np.arange(len(values)), codes)),
            shape=(len(values), len(uniq)),
        )
        sim = (m @ m.T).tocsr()
        # 缺失值不构成相似（否则所有作者缺失的书会互相相似）
        if na_idx:
            na_code = next(iter(na_idx))
            hit = np.where(values == na_token)[0]
            if len(hit):
                sim = sim.tolil()
                for i in hit:
                    sim.rows[i] = []
                    sim.data[i] = []
                sim = sim.tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()
        return sim.astype(np.float32)

    def _build_group(self, hist: pd.DataFrame):
        """构建群体偏好分布，按 (dept, major) 两级。"""
        n_items = len(self.items)
        dept_acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_items, dtype=np.float32))
        major_acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_items, dtype=np.float32))
        major_n: dict[str, int] = defaultdict(int)

        idx = hist["call_no"].map(self.item_idx)
        mask = idx.notna().values
        h = hist[mask]
        idx = idx[mask].astype(int).values
        for d, j in zip(h["dept"].values, idx):
            dept_acc[d][j] += 1.0
        for m, j in zip(h["major"].values, idx):
            major_acc[m][j] += 1.0
            major_n[m] += 1

        self.global_vec = self._norm(np.asarray(self.pop, dtype=np.float32))
        self.dept_vec = {k: self._norm(v) for k, v in dept_acc.items()}
        self.major_vec = {k: self._norm(v) for k, v in major_acc.items()}
        self.major_n = dict(major_n)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        return v / s if s > 0 else v

    def _build_new_weight(self, newbooks: pd.DataFrame, ref_date=None):
        """新书按入藏时间给权重；回测时 ref_date 为切分点。"""
        if "in_date" not in newbooks or newbooks.empty:
            return
        ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp.now()
        age = (ref - newbooks["in_date"]).dt.days
        w = np.where(age < 0, 0.3, np.where(age <= 210, 1.0, np.where(age <= 450, 0.7, 0.4)))
        self.new_weight = dict(zip(newbooks["call_no"], w))

    # ---------- 类目配额打分 ----------
    def _prepare_quota(self, hist: pd.DataFrame):
        """按中图法一级类目分组，并计算类目内的近 12 个月热度。

        依据：老人答案中 70.3% 与历史同属一个一级类目，但仅 19.6% 同细分分类号、
        1.2% 同作者。因此「类目级画像 + 类目内热度」比「精确分类号/作者匹配」
        更贴合数据，也比共现更稳。
        """
        n_items = len(self.items)
        idx = hist["call_no"].map(self.item_idx)
        mask = idx.notna().values
        h = hist[mask].copy()
        h["_j"] = idx[mask].astype(int).values

        end = h["ts"].max()
        recent_from = end - pd.Timedelta(days=365)
        rec = h[h["ts"] >= recent_from]

        pop_recent = np.zeros(n_items, dtype=np.float32)
        np.add.at(pop_recent, rec["_j"].values, 1.0)
        self.pop_recent = pop_recent
        self.pop_all = np.asarray(self.pop, dtype=np.float32)

        self.cls_items: dict[str, np.ndarray] = {}
        grp = h.groupby(h["_j"].map(lambda j: self.cls_of.get(self.items[j], "?")))
        for c, g in grp:
            js = np.unique(g["_j"].values)
            self.cls_items[c] = js

        tot = {c: float(self.pop_all[js].sum()) for c, js in self.cls_items.items()}
        s = sum(tot.values()) or 1.0
        self.global_cls_share = {c: v / s for c, v in tot.items()}

    def _cls_dist_from_hist(self, calls: list[str]) -> dict[str, float]:
        cnt: dict[str, float] = defaultdict(float)
        for c in calls:
            j = self.item_idx.get(c)
            if j is not None:
                cnt[self.cls_of.get(c, "?")] += 1.0
        s = sum(cnt.values())
        return {k: v / s for k, v in cnt.items()} if s else {}

    def _group_dist_and_base(self, profile: dict) -> tuple[dict[str, float], np.ndarray]:
        """返回群体的类目分布（按 lift 提权）与物品级先验向量。"""
        dept = profile.get("dept", "其他")
        major = profile.get("major", "未知")
        mv = self.major_vec.get(major)
        dv = self.dept_vec.get(dept)
        if dv is None:
            dv = self.global_vec
        n = self.major_n.get(major, 0)
        lam = n / (n + float(self.p["k_smooth"])) if mv is not None else 0.0
        base = lam * mv + (1 - lam) * dv if mv is not None else dv
        if base is None:
            base = self.global_vec
        cnt: dict[str, float] = defaultdict(float)
        top = np.argsort(-base)[:300]
        for j in top:
            cnt[self.cls_of.get(self.items[j], "?")] += float(base[j])
        s = sum(cnt.values())
        if not s:
            return {}, base
        share = {k: v / s for k, v in cnt.items()}
        g = self.global_cls_share
        out = {k: share[k] * (share[k] / (g.get(k, 1e-6))) for k in share}
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}, base

    def _cls_dist_from_group(self, profile: dict) -> dict[str, float]:
        return self._group_dist_and_base(profile)[0]

    def _quota_pick(
        self,
        cls_dist: dict[str, float],
        user_cls2: set[str],
        seen: set[int],
        k: int,
        prefer_new: bool = True,
        prior: np.ndarray | None = None,
    ) -> list[str]:
        if not cls_dist:
            cls_dist = {"I": 1.0}
        top = sorted(cls_dist.items(), key=lambda x: -x[1])[:5]
        tot = sum(p for _, p in top) or 1.0
        alloc = []
        left = k
        for c, p in top:
            if left <= 0:
                break
            n = max(1, int(round(k * p / tot)))
            n = min(n, left)
            alloc.append((c, n))
            left -= n
        if left > 0:
            alloc.append((top[0][0] if top else "I", left))

        picked: list[str] = []
        for c, n in alloc:
            if len(picked) >= k or n <= 0:
                break
            js = self.cls_items.get(c)
            if js is None:
                continue
            js = np.array([j for j in js if j not in seen])
            if js.size == 0:
                continue
            base = 0.7 * self.pop_recent[js] / (self.pop_recent.max() + 1e-9) + 0.3 * self.pop_all[js] / (
                self.pop_all.max() + 1e-9
            )
            if prior is not None:
                pv = prior[js]
                mx = pv.max()
                pv = pv / mx if mx > 0 else pv
                rec_n = self.pop_recent[js] / (self.pop_recent.max() + 1e-9)
                base = 0.9 * pv + 0.1 * rec_n
            if user_cls2:
                same = np.array(
                    [1.0 if self.cls2_of.get(self.items[j], "?") in user_cls2 else 0.0 for j in js]
                )
                base = base * (1.0 + 0.8 * same)
            if prefer_new and self.new_weight:
                nw = np.array([self.new_weight.get(self.items[j], 1.0) for j in js])
                base = base * nw
            order = np.argsort(-base)
            for t in order:
                if len(picked) >= k or n <= 0:
                    break
                call = self.items[js[t]]
                if call in picked or js[t] in seen:
                    continue
                picked.append(call)
                n -= 1
        if len(picked) < k:
            for j in np.argsort(-self.pop_recent):
                if len(picked) >= k:
                    break
                if j in seen:
                    continue
                call = self.items[j]
                if call not in picked:
                    picked.append(call)
        return picked[:k]

    # ---------- 推荐 ----------
    def _group_score(self, profile: dict) -> np.ndarray:
        p = self.p
        lam_gate = float(p["k_smooth"])
        dept = profile.get("dept", "其他")
        major = profile.get("major", "未知")
        dv = self.dept_vec.get(dept, self.global_vec)
        mv = self.major_vec.get(major)
        eps = 0.05
        if mv is not None and major != "未知":
            n = self.major_n.get(major, 0)
            lam = n / (n + lam_gate)
            base = lam * mv + (1.0 - lam) * dv
        else:
            base = dv
        return (1.0 - eps) * base + eps * self.global_vec

    def _hot_personalized(
        self,
        cls_dist: dict[str, float],
        user_cls2: set[str],
        seen: set[int],
        k: int,
        top_n: int = 300,
        beta: float = 0.45,
        gamma: float = 1.2,
        max_per_cls: int = 3,
    ) -> list[str]:
        """热门候选池 + 个性化重排。

        实测依据：老人答案中 61.2% 位于全局热度前 10%，单纯全局热门（Hit 0.0588）
        已优于类目内热门（0.0484）。因此正确做法不是缩小到专业类目，而是在
        热门候选池内按类目偏好重排。
        """
        order = np.argsort(-self.pop_recent)
        cand = [j for j in order if j not in seen][:top_n]
        if not cand:
            return []
        cand = np.asarray(cand)
        pop = self.pop_recent[cand]
        pop_n = pop / (pop.max() + 1e-9)
        prob = np.array([cls_dist.get(self.cls_of.get(self.items[j], "?"), 0.0) for j in cand])
        mx = prob.max()
        prob_n = prob / mx if mx > 0 else prob
        if user_cls2:
            same = np.array(
                [1.0 if self.cls2_of.get(self.items[j], "?") in user_cls2 else 0.0 for j in cand]
            )
        else:
            same = np.zeros(len(cand))
        score = (pop_n ** beta) * ((prob_n + 0.05) ** gamma) * (1.0 + 0.35 * same)
        if self.new_weight:
            nw = np.array([self.new_weight.get(self.items[j], 1.0) for j in cand])
            score = score * nw
        order = np.argsort(-score)
        # 类目上限：新人组无 NDCG 项，10 个位置应尽量覆盖不同方向；
        # 若不加约束，674 名新人会塌缩到同一批热门书（实测仅 163 本唯一）。
        picked: list[str] = []
        cls_cnt: dict[str, int] = defaultdict(int)
        for t in order:
            if len(picked) >= k:
                break
            j = cand[t]
            cl = self.cls_of.get(self.items[j], "?")
            if cls_cnt[cl] >= max_per_cls:
                continue
            picked.append(self.items[j])
            cls_cnt[cl] += 1
        if len(picked) < k:
            for t in order:
                if len(picked) >= k:
                    break
                it = self.items[cand[t]]
                if it not in picked:
                    picked.append(it)
        return picked[:k]

    def prepare_dists(self, readers, profiles, hist_map) -> dict[str, tuple[dict, set]]:
        """预计算每位读者的类目分布，供参数搜索复用（这部分与超参无关）。"""
        out = {}
        w_self = float(self.p.get("w_self", 0.8))
        for rid in readers:
            prof = profiles.get(rid, {})
            calls = hist_map.get(rid) or []
            group = self._cls_dist_from_group(prof)
            if calls:
                user_cls2 = {self.cls2_of.get(c, "?") for c in calls if c in self.item_idx}
                self_d = self._cls_dist_from_hist(calls)
                dist: dict[str, float] = defaultdict(float)
                for c, v in self_d.items():
                    dist[c] += w_self * v
                for c, v in group.items():
                    dist[c] += (1.0 - w_self) * v
            else:
                user_cls2 = set()
                dist = group
            out[rid] = (dist, user_cls2)
        return out

    def recommend_all(
        self,
        readers: list[str],
        profiles: dict[str, dict],
        hist_map: dict[str, list[str]],
        utype_map: dict[str, str],
        k: int = TOP_K,
        dists: dict | None = None,
    ) -> dict[str, list[str]]:
        p = self.p
        top_n = int(p.get("top_n", 300))
        beta = float(p.get("beta_pop", 0.45))
        gamma = float(p.get("gamma_cls", 1.2))
        dists = dists if dists is not None else self.prepare_dists(readers, profiles, hist_map)
        out: dict[str, list[str]] = {}
        for rid in readers:
            calls = hist_map.get(rid) or []
            seen = {self.item_idx[c] for c in calls if c in self.item_idx}
            dist, user_cls2 = dists[rid]
            mp = int(p.get("max_per_cls_new", 3)) if not calls else int(p.get("max_per_cls_old", 3))
            out[rid] = self._hot_personalized(
                dist, user_cls2, seen, k, top_n, beta, gamma, mp
            )
        return out

    @staticmethod
    def _rrf(s: np.ndarray, k: float) -> np.ndarray:
        """把任意分值转为 RRF 权重：只依赖排名，对量纲免疫。"""
        order = np.argsort(-s)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(s) + 1)
        return (1.0 / (k + rank)).astype(np.float32)

    def _shape(self, score: np.ndarray, i: int, ut: str, seen: set[int], k: int) -> list[str]:
        """老人：前 4 位按分精排，后 6 位类目分散以提升 Recall。"""
        order = np.argsort(-score)
        picked: list[str] = []
        cls_cnt: dict[str, int] = defaultdict(int)
        for j in order:
            if len(picked) >= 4:
                break
            if score[j] <= 0:
                continue
            c = self.items[j]
            if c in picked:
                continue
            picked.append(c)
            cls_cnt[self.cls_of.get(c, "?")] += 1
        for j in order:
            if len(picked) >= k:
                break
            c = self.items[j]
            if c in picked or score[j] <= 0:
                continue
            cl = self.cls_of.get(c, "?")
            if cls_cnt[cl] >= 2:
                continue
            picked.append(c)
            cls_cnt[cl] += 1
        for j in order:
            if len(picked) >= k:
                break
            c = self.items[j]
            if c not in picked and score[j] > 0:
                picked.append(c)
        return picked

    def _shape_cold(self, score: np.ndarray, k: int, profile: dict) -> list[str]:
        """新人：专业类目 6 + 文学保底 3 + 其余 1，最大化 10 次押注的覆盖。"""
        order = np.argsort(-score)
        dept = profile.get("dept", "其他")
        mv = self.major_vec.get(profile.get("major", ""))
        dv = self.dept_vec.get(dept)
        base = mv if mv is not None else dv
        prefer: set[str] = set()
        if base is not None and self.global_vec is not None:
            ratio = np.divide(base, self.global_vec + 1e-9)
            cls_score: dict[str, float] = defaultdict(float)
            for j in np.argsort(-ratio)[:400]:
                cls_score[self.cls_of.get(self.items[j], "?")] += ratio[j]
            top_cls = [c for c, _ in sorted(cls_score.items(), key=lambda x: -x[1])[:4]]
            prefer = set(top_cls) - {"I"}

        picked: list[str] = []
        if prefer:
            for j in order:
                if len(picked) >= 6:
                    break
                c = self.items[j]
                if score[j] > 0 and c not in picked and self.cls_of.get(c, "?") in prefer:
                    picked.append(c)
        for j in order:
            if len(picked) >= 9:
                break
            c = self.items[j]
            if c not in picked and score[j] > 0 and self.cls_of.get(c, "?") == "I":
                picked.append(c)
        for j in order:
            if len(picked) >= k:
                break
            c = self.items[j]
            if c not in picked and score[j] > 0:
                picked.append(c)
        if len(picked) < k:
            for j in order:
                if len(picked) >= k:
                    break
                c = self.items[j]
                if c not in picked:
                    picked.append(c)
        return picked[:k]
