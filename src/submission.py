"""写提交文件与格式校验（对照组委会验证集说明第三节的校验规则）。"""
from __future__ import annotations

import pandas as pd

from .config import TOP_K


def write_submission(pred: dict[str, list[str]], valid_ids: list[str], path) -> None:
    rows = []
    for rid in valid_ids:
        items = []
        seen = set()
        for c in pred.get(rid, []):
            if c and c not in seen:
                seen.add(c)
                items.append(c)
            if len(items) >= TOP_K:
                break
        rows.append({"读者ID": rid, "推荐索书号列表": "|".join(items)})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def validate(path, valid_ids: list[str]) -> list[str]:
    """返回问题列表，空列表表示通过。"""
    problems: list[str] = []
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    if "读者ID" not in df.columns or "推荐索书号列表" not in df.columns:
        return [f"列名不符：{list(df.columns)}"]

    ids = df["读者ID"].tolist()
    if len(ids) != len(set(ids)):
        problems.append("存在重复的读者ID")
    missing = set(valid_ids) - set(ids)
    unknown = set(ids) - set(valid_ids)
    if missing:
        problems.append(f"缺失 {len(missing)} 个读者ID，例如 {sorted(missing)[:3]}")
    if unknown:
        problems.append(f"出现 {len(unknown)} 个未知ID，例如 {sorted(unknown)[:3]}")

    for _, row in df.iterrows():
        items = [x for x in str(row["推荐索书号列表"]).split("|") if x]
        if len(items) > TOP_K:
            problems.append(f"{row['读者ID']} 超过 {TOP_K} 条")
        if len(items) != len(set(items)):
            problems.append(f"{row['读者ID']} 列表内索书号重复")
        if not items:
            problems.append(f"{row['读者ID']} 推荐为空")
    return problems
