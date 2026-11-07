"""生成验证集提交文件（定稿模型）。

与回测的唯一差异：把验证集 956 人的历史借阅列表并入统计。
这份列表覆盖 2025-01 至 2025-08，紧邻答案窗口（2025-09 起），
是训练集中不存在的最新行为信号；回测无法复现这一点，
但它只会增强「学院近期热度」的时效性。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR
from src.data_loader import cls_full_of_callno, cls_of_callno, load_train
from src.model_final import DeptBayesRecommender
from src.splits import build_valid_task
from src.submission import validate, write_submission


def build_history_frame(task: dict) -> pd.DataFrame:
    """把 956 人的历史索书号列表展开成借阅流水（时间铺到 2025-01 ~ 2025-08）。"""
    tr = load_train()
    info = tr.drop_duplicates("call_no").set_index("call_no")[["cls1", "cls2"]]
    rows = []
    base = pd.Timestamp("2025-01-20")
    for rid, df in task["hist_by_reader"].items():
        prof = task["profile"][rid]
        calls = df["call_no"].tolist()
        n = len(calls)
        for i, c in enumerate(calls):
            ts = base + pd.Timedelta(days=int(i * 210 / max(n, 1)))
            rows.append(
                {
                    "reader": rid,
                    "call_no": c,
                    "ts": ts,
                    "dept": prof.get("dept", "其他"),
                    "major": prof.get("major", "未知"),
                    "level": prof.get("level", "未知"),
                    "cls1": info["cls1"].get(c, cls_of_callno(c)),
                    "cls2": info["cls2"].get(c, cls_full_of_callno(c)),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_DIR / "result.csv"))
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--k_cls2", type=float, default=1.6)
    ap.add_argument("--newbook_quantile", type=float, default=0.70)
    args = ap.parse_args()

    task = build_valid_task()
    readers = task["valid"]["rid"].tolist()

    tr = load_train()
    extra = build_history_frame(task)
    hist = pd.concat([tr, extra], ignore_index=True)

    model = DeptBayesRecommender(
        {
            "lam": args.lam,
            "k_cls2": args.k_cls2,
            "newbook_quantile": args.newbook_quantile,
        }
    ).fit(hist, task["newbooks"])

    hist_map = {rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()}
    pred = model.recommend_all(readers, task["profile"], hist_map, k=10)

    out = Path(args.out)
    write_submission(pred, readers, out)
    problems = validate(out, readers)
    df = pd.read_csv(out, encoding="utf-8-sig")
    print(f"写出：{out}")
    print(f"行数：{len(df)}")
    if problems:
        print("校验未通过：")
        for p in problems[:10]:
            print("  -", p)
    else:
        print("校验通过：ID 全覆盖、无重复、≤10 条、无空行")

    nb = set(task["newbooks"]["call_no"])
    utype = task["utype"]
    for g in ("新人", "老人"):
        sub = [r for r in readers if utype[r] == g]
        calls = [c for r in sub for c in pred.get(r, [])]
        uniq = len(set(calls))
        nbs = sum(1 for c in calls if c in nb)
        print(f"{g}：{len(sub)} 人，推荐 {len(calls)} 条，唯一书 {uniq} 本，新书占比 {nbs / max(len(calls), 1):.1%}")


if __name__ == "__main__":
    main()
