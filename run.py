"""一键入口：从原始数据生成验证集提交文件并校验。

用法：
    python run.py                # 生成 outputs/result.csv 并校验
    python run.py --check        # 仅校验已存在的 outputs/result.csv
    python run.py --backtest     # 额外跑四折回测，打印总分
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULT = ROOT / "outputs" / "result.csv"


def run_predict():
    from scripts.predict_final import main as predict_main  # noqa: F401

    cmd = [sys.executable, str(ROOT / "scripts" / "predict_final.py")]
    subprocess.run(cmd, check=True)


def run_check():
    import pandas as pd

    from src.data_loader import load_valid
    from src.submission import validate

    if not RESULT.exists():
        print(f"未找到 {RESULT}，请先运行 python run.py")
        return 1
    valid = load_valid()
    ids = valid["rid"].tolist()
    problems = validate(RESULT, ids)
    df = pd.read_csv(RESULT, encoding="utf-8-sig")
    print(f"文件：{RESULT}")
    print(f"行数：{len(df)}（应为 {len(ids)}）")
    print(f"列名：{list(df.columns)}")
    if problems:
        print("校验未通过：")
        for p in problems[:20]:
            print("  -", p)
        return 1
    print("校验通过：ID 全覆盖、行内无重复、每行 ≤ 10 条、无空行")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅校验已有结果文件")
    ap.add_argument("--backtest", action="store_true", help="额外运行四折回测")
    args = ap.parse_args()

    if args.check:
        sys.exit(run_check())

    run_predict()
    code = run_check()
    if args.backtest:
        print("\n=== 四折回测 ===")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "backtest.py"),
                        "--split", "2024-09-01"], check=False)
    sys.exit(code)


if __name__ == "__main__":
    main()
