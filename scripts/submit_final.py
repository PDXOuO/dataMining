"""生成决赛四份提交：公开榜 / 隐藏榜 / 扰动A / 扰动B。

同一套模型，只是输入不同——手册 Q3 明确要求三份提交是同一模型在不同输入上的推理，
不要为扰动单独调参。

用法：
    python scripts/submit_final.py                  # 全部四份
    python scripts/submit_final.py --bench public   # 只生成公开榜
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.final_data import FIN, load_bench, load_train_base  # noqa: E402
from src.final_model import FinalRecommender  # noqa: E402

OUT = ROOT / "submission" / "final"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {
    "public": "提交_公开榜.csv",
    "hidden": "提交_隐藏榜.csv",
    "pert_a": "提交_隐藏榜_扰动A.csv",
    "pert_b": "提交_隐藏榜_扰动B.csv",
}
TEMPLATES = {
    "public": "决赛_公开榜_提交模板.csv",
    "hidden": "决赛_隐藏榜_提交模板.csv",
    "pert_a": "决赛_隐藏榜_提交模板.csv",
    "pert_b": "决赛_隐藏榜_提交模板.csv",
}
INPUTS = {
    "public": "决赛_公开榜_输入.csv",
    "hidden": "决赛_隐藏榜_输入.csv",
    "pert_a": "决赛_隐藏榜_输入_扰动A_画像缺失.csv",
    "pert_b": "决赛_隐藏榜_输入_扰动B_历史扰动.csv",
}


def validate(path: Path, bench_name: str, bench) -> list[str]:
    """对齐《决赛参赛手册》三节规则做自查。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    tmpl = pd.read_csv(FIN / TEMPLATES[bench_name], encoding="utf-8-sig", dtype=str)
    problems = []
    if list(df.columns) != ["读者ID", "推荐索书号列表"]:
        problems.append(f"列名不符：{list(df.columns)}")
    if len(df) != len(tmpl):
        problems.append(f"行数 {len(df)} != 模板 {len(tmpl)}")
    if set(df["读者ID"]) != set(tmpl["读者ID"]):
        miss = set(tmpl["读者ID"]) - set(df["读者ID"])
        problems.append(f"读者 ID 不一致，缺 {len(miss)} 个")
    if df["读者ID"].duplicated().any():
        problems.append("读者 ID 重复")

    white = set(bench.calls)
    for rid, s in zip(df["读者ID"], df["推荐索书号列表"].fillna("")):
        items = [x for x in s.split("|") if x]
        if not items:
            problems.append(f"{rid} 空推荐")
            continue
        if len(items) > 10:
            problems.append(f"{rid} 超过 10 条（{len(items)}）")
        if len(set(items)) != len(items):
            problems.append(f"{rid} 行内重复")
        out = [x for x in items if x not in white]
        if out:
            problems.append(f"{rid} 白名单外 {len(out)} 条：{out[:2]}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None, choices=list(NAMES))
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    params = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v or "e" in v.lower() else int(v)

    train = load_train_base()
    targets = [args.bench] if args.bench else list(NAMES)

    for name in targets:
        bench = load_bench(name)
        model = FinalRecommender(params).fit(bench, train)
        rec = model.recommend_all()

        rows = [(rid, "|".join(rec[rid])) for rid in bench.readers["读者ID"]]
        df = pd.DataFrame(rows, columns=["读者ID", "推荐索书号列表"])
        path = OUT / NAMES[name]
        df.to_csv(path, index=False, encoding="utf-8-sig")

        probs = validate(path, name, bench)
        new_flag = dict(zip(bench.calls, bench.call_new))
        uniq = len({c for r in rec.values() for c in r})
        new_cnt = sum(1 for r in rec.values() for c in r if new_flag.get(c))
        print(
            f"{NAMES[name]:<24} {len(df)} 行  唯一书 {uniq:5d}  "
            f"新书占比 {new_cnt / max(len(df) * 10, 1):.1%}  "
            f"{'校验通过' if not probs else '校验失败 ' + str(probs[:3])}"
        )


if __name__ == "__main__":
    main()
