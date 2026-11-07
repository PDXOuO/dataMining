"""时间切分与回测任务构造。

严格遵循全局时间轴：训练侧只用切分点之前的记录，候选池只保留切分点之前已入藏的图书。
任何在切分点之后的信息（热度、共现、新书权重）都不得进入特征，否则指标会虚高
（Ji et al., TOIS 2023 报告泄漏可使 nDCG 虚高 21.7%~73.4%，且模型排名反转）。
"""
from __future__ import annotations

import pandas as pd

from .config import ANSWER_WINDOW_MONTHS, TOP_K
from .data_loader import load_newbooks, load_train, load_valid


def build_backtest(split_date: str, months: int = ANSWER_WINDOW_MONTHS) -> dict:
    """构造一折回测任务。

    返回：
        hist      reader -> 切分点前的借阅记录 DataFrame
        truth     reader -> 答案索书号列表（老人已剔除历史已借）
        utype     reader -> "新人" / "老人"
        profile   reader -> 画像 dict（dept, level）
        cand      切分点前可用的索书号全集（候选池）
    """
    tr = load_train()
    T = pd.Timestamp(split_date)
    end = T + pd.DateOffset(months=months)

    hist = tr[tr["ts"] < T]
    ans = tr[(tr["ts"] >= T) & (tr["ts"] < end)]

    hist_by_reader = {
        rid: g.sort_values("ts") for rid, g in hist.groupby("reader", sort=False)
    }

    # 画像：从该读者的全部记录中取众数（画像本身不是行为信号，轻微前视可接受）
    prof = (
        tr.groupby("reader")
        .agg(
            dept=("dept", lambda s: s.mode().iat[0] if not s.mode().empty else "其他"),
            level=("level", lambda s: s.mode().iat[0] if not s.mode().empty else "未知"),
            major=("major", lambda s: s.mode().iat[0] if not s.mode().empty else "未知"),
        )
        .to_dict("index")
    )

    truth: dict[str, list[str]] = {}
    utype: dict[str, str] = {}
    for rid, g in ans.groupby("reader", sort=False):
        seq = g.sort_values("ts")["call_no"].drop_duplicates().tolist()
        if rid in hist_by_reader:
            seen = set(hist_by_reader[rid]["call_no"])
            seq = [c for c in seq if c not in seen]
            utype[rid] = "老人"
        else:
            utype[rid] = "新人"
        if seq:
            truth[rid] = seq[: max(TOP_K, 50)]

    # 候选池：切分点前出现过的索书号 + 切分点前已入藏的新书
    nb = load_newbooks()
    nb_before = nb[(nb["in_date"].notna()) & (nb["in_date"] < T)]
    cand = set(hist["call_no"].dropna().unique()) | set(nb_before["call_no"])

    return {
        "split": split_date,
        "hist": hist,
        "hist_by_reader": hist_by_reader,
        "truth": truth,
        "utype": utype,
        "profile": prof,
        "cand": cand,
        "newbooks": nb_before,
    }


def build_valid_task() -> dict:
    """最终预测任务：验证集 956 人。"""
    v = load_valid()
    nb = load_newbooks()
    hist_by_reader = {}
    for _, row in v.iterrows():
        calls = row["hist_calls"]
        if calls:
            # 验证集列表按时间升序，位置越靠后越近
            hist_by_reader[row["rid"]] = pd.DataFrame(
                {
                    "call_no": calls,
                    "rank": range(len(calls)),
                }
            )
    return {
        "valid": v,
        "hist_by_reader": hist_by_reader,
        "utype": dict(zip(v["rid"], v["utype"])),
        "profile": {
            r["rid"]: {"dept": r["dept"], "level": r["level"], "major": r["major"],
                       "grade": r["grade"], "gender": r["gender"]}
            for _, r in v.iterrows()
        },
        "newbooks": nb,
        "cand": set(nb["call_no"]) | {c for cs in hist_by_reader.values() for c in cs["call_no"]},
    }
