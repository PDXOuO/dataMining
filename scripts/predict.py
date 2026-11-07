"""生成验证集提交文件。

与回测的关键差异：
1. 验证集的「历史借阅索书号列表」代表 2025-01~2025-08 的真实借阅，时间上紧邻答案窗口
   （2025-09 起），因此把它们并入热度统计，作为「近期趋势」信号；
2. 新书池在验证任务中可用（8297 册），按入藏时间分桶加权。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR
from src.data_loader import cls_full_of_callno, cls_of_callno, load_train
from src.recommender import HybridRecommender
from src.splits import build_valid_task
from src.submission import validate, write_submission


def build_history_frame(task: dict) -> pd.DataFrame:
    """把 956 人的历史列表展开成借阅流水，时间落在 2025-01~2025-08 区间。"""
    tr = load_train()
    info = tr.drop_duplicates("call_no").set_index("call_no")[["cls1", "cls2", "author"]]
    rows = []
    base = pd.Timestamp("2025-01-20")
    for rid, df in task["hist_by_reader"].items():
        prof = task["profile"][rid]
        n = len(df)
        for i, c in enumerate(df["call_no"]):
            # 越靠后越接近切分点：均匀铺开到 2025-08
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
                    "author": info["author"].get(c, ""),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_DIR / "result.csv"))
    ap.add_argument("--top_n", type=int, default=1000)
    ap.add_argument("--beta_pop", type=float, default=0.40)
    ap.add_argument("--gamma_cls", type=float, default=0.80)
    ap.add_argument("--w_self", type=float, default=0.70)
    ap.add_argument("--max_new", type=int, default=6)
    ap.add_argument("--max_old", type=int, default=4)
    args = ap.parse_args()

    task = build_valid_task()
    valid = task["valid"]
    readers = valid["rid"].tolist()

    tr = load_train()
    extra = build_history_frame(task)
    hist = pd.concat([tr, extra], ignore_index=True)

    model = HybridRecommender(
        {
            "top_n": args.top_n,
            "beta_pop": args.beta_pop,
            "gamma_cls": args.gamma_cls,
            "w_self": args.w_self,
            "max_per_cls_new": args.max_new,
            "max_per_cls_old": args.max_old,
        }
    ).fit(hist, task["newbooks"])

    hist_map = {
        rid: df["call_no"].tolist() for rid, df in task["hist_by_reader"].items()
    }
    pred = model.recommend_all(readers, task["profile"], hist_map, task["utype"], k=10)

    out = Path(args.out)
    write_submission(pred, readers, out)
    problems = validate(out, readers)
    print(f"写出：{out}")
    print(f"行数：{len(pd.read_csv(out, encoding='utf-8-sig'))}")
    if problems:
        print("校验未通过：")
        for p in problems[:10]:
            print("  -", p)
    else:
        print("校验通过：ID 全覆盖、无重复、≤10 条、无空行")


if __name__ == "__main__":
    main()
