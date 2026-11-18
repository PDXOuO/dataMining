"""决赛数据层：统一读取 4 份输入、白名单与学院映射，并把初赛训练集作为群体行为底座。

与初赛的差异：
- 决赛只给输入（画像 + 历史），不给行为明细，群体信号只有两个来源：
  初赛训练集（29841 条，2022-01 ~ 2025-01）与决赛输入自身的历史（更新但样本小）；
- 候选集合被白名单限定（公开榜 21536 / 隐藏榜 22150），越界不计分；
- 白名单带「是否新入藏」标记，新入藏 7386 本占 34%，但其中仅 318 本有借阅记录。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_loader import cls_full_of_callno, cls_of_callno, normalize_dept, normalize_major

ROOT = Path(__file__).resolve().parent.parent
FIN = ROOT / "02_数据"

BENCHES = {
    "public": ("决赛_公开榜_输入.csv", "图书白名单_公开榜.csv"),
    "hidden": ("决赛_隐藏榜_输入.csv", "图书白名单_隐藏榜.csv"),
    "pert_a": ("决赛_隐藏榜_输入_扰动A_画像缺失.csv", "图书白名单_隐藏榜.csv"),
    "pert_b": ("决赛_隐藏榜_输入_扰动B_历史扰动.csv", "图书白名单_隐藏榜.csv"),
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIN / name, encoding="utf-8-sig", dtype=str)


@dataclass
class Bench:
    name: str
    readers: pd.DataFrame
    whitelist: pd.DataFrame
    calls: list[str]
    call_cls1: list[str]
    call_cls2: list[str]
    call_new: list[int]
    hist: dict[str, list[str]]          # 读者 -> 历史索书号（保序、去重）

    def __len__(self):
        return len(self.readers)


def load_bench(name: str) -> Bench:
    inp, wht = BENCHES[name]
    readers = read_csv(inp)
    whitelist = read_csv(wht)

    calls = whitelist["索书号"].astype(str).tolist()
    cls1 = whitelist["分类号"].fillna("").tolist()
    cls2 = [cls_full_of_callno(c) for c in calls]
    # 白名单分类号只到一级，二级类目统一从索书号解析，避免两套口径
    cls2 = [c if c else cls_full_of_callno(calls[i]) for i, c in enumerate(cls2)]
    new = (whitelist["是否新入藏"].astype(str) == "1").astype(int).tolist()

    hist: dict[str, list[str]] = {}
    for rid, s in zip(readers["读者ID"], readers["历史借阅索书号列表"].fillna("")):
        seq = [x.strip() for x in s.split("|") if x.strip()]
        seen, uniq = set(), []
        for c in seq:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        hist[rid] = uniq

    readers = readers.copy()
    readers["dept_n"] = readers["学院"].fillna("未知").map(normalize_dept)
    readers["major_n"] = readers["专业"].fillna("未知").map(normalize_major)
    readers["grade_n"] = readers["年级"].fillna("未知")
    readers["is_old"] = (readers["用户类型"] == "老人").astype(int)

    return Bench(name, readers, whitelist, calls, cls1, cls2, new, hist)


def load_train_base() -> pd.DataFrame:
    """初赛训练集，作为群体行为底座。"""
    from .data_loader import load_train

    tr = load_train().copy()
    tr["dept_n"] = tr["dept"].map(normalize_dept)
    tr["major_n"] = tr["major"].map(normalize_major)
    return tr


def hist_frame(bench: Bench) -> pd.DataFrame:
    """把决赛输入里的历史展开成 (reader, call_no, dept) 三元组。"""
    dept = dict(zip(bench.readers["读者ID"], bench.readers["dept_n"]))
    rows = [(rid, c, dept.get(rid, "其他")) for rid, seq in bench.hist.items() for c in seq]
    return pd.DataFrame(rows, columns=["reader", "call_no", "dept"])
