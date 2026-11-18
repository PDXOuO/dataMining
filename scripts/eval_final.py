"""决赛模型的离线回测：把初赛的四折时间切分套进决赛模型的输入结构。

决赛没有公开答案，本地只能用初赛训练集造「切分点前的群体底座 + 切分点后的真实借阅」，
这与决赛任务（用已有行为预测下一次借阅）在结构上一致，因此用于变体之间的相对比较。

用法：
    python scripts/eval_final.py                 # 四折
    python scripts/eval_final.py --split 2024-09-01
    python scripts/eval_final.py --set lam=2.0 --set k_cls2=2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import cls_full_of_callno, cls_of_callno  # noqa: E402
from src.final_data import Bench  # noqa: E402
from src.final_model import FinalRecommender  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.splits import build_backtest  # noqa: E402

SPLITS = ["2023-09-01", "2024-03-01", "2024-06-01", "2024-09-01"]


def bench_from_task(task: dict) -> Bench:
    """把回测任务包装成决赛输入的形状。"""
    hist_df = task["hist"]
    calls = sorted(task["cand"])
    seen = set(hist_df["call_no"].dropna())
    new = [0 if c in seen else 1 for c in calls]

    hist: dict[str, list[str]] = {}
    for rid, g in task["hist_by_reader"].items():
        seq, s = [], set()
        for c in g.sort_values("ts")["call_no"]:
            if c not in s:
                s.add(c)
                seq.append(c)
        hist[rid] = seq
    for rid in task["truth"]:
        hist.setdefault(rid, [])
        hist.setdefault(rid, [])

    prof = task["profile"]
    readers = pd.DataFrame(
        [
            {
                "读者ID": rid,
                "用户类型": task["utype"].get(rid, "新人"),
                "学院": prof.get(rid, {}).get("dept", "其他"),
                "专业": prof.get(rid, {}).get("major", "未知"),
                "年级": str(prof.get(rid, {}).get("grade", "未知")),
            }
            for rid in hist
        ]
    )
    readers["dept_n"] = readers["学院"]
    readers["major_n"] = readers["专业"]
    readers["grade_n"] = readers["年级"]
    readers["is_old"] = (readers["用户类型"] == "老人").astype(int)

    return Bench(
        name="backtest",
        readers=readers,
        whitelist=pd.DataFrame({"索书号": calls}),
        calls=calls,
        call_cls1=[cls_of_callno(c) for c in calls],
        call_cls2=[cls_full_of_callno(c) for c in calls],
        call_new=new,
        hist=hist,
    )


def run_one(split: str, params: dict, verbose: bool = True) -> dict:
    task = build_backtest(split)
    bench = bench_from_task(task)
    # 群体底座 = 切分点前的全部借阅；与决赛「训练集 + 输入历史」同构
    model = FinalRecommender(params).fit(bench, task["hist"])
    pred = model.recommend_all()
    res = evaluate(pred, task["truth"], task["utype"], verbose=False)
    if verbose:
        print(
            f"{split}  总 {res['total']:5.2f}  "
            f"新 {res['new']['score']*100:5.2f}(n={res['new']['n']})  "
            f"老 {res['old']['score']*100:5.2f}(n={res['old']['n']})"
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    params = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v or "e" in v.lower() else int(v)

    splits = [args.split] if args.split else SPLITS
    tot = []
    for s in splits:
        r = run_one(s, params)
        tot.append(r["total"])
    print(f"均值 {sum(tot)/len(tot):.2f}  ({len(tot)} 折)")


if __name__ == "__main__":
    main()
