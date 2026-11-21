"""打包提交物：生成一个可整包上传的 zip。

平台只允许上传单个附件，且要求“文件直接在压缩包根目录、不可嵌套文件夹”，
因此采用「外层包 = 提交件平铺 + 一个内层源码包」的结构：
提交件能被平台直接读取，源码包保持完整目录结构、解压即跑。

用法：
    python scripts/make_package.py            # 决赛包（默认，四份 CSV）
    python scripts/make_package.py --stage classic   # 初赛包（结果 + 承诺书 + 报告）
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
TEAM = "洛杉矶湖人"

OUTER = SUB / f"{TEAM}_提交包.zip"
INNER = SUB / f"{TEAM}_源代码与可运行包.zip"
OUTER_FINAL = SUB / f"{TEAM}_决赛提交包.zip"
INNER_FINAL = SUB / f"{TEAM}_决赛源代码与可运行包.zip"
OUTER_S1 = SUB / f"{TEAM}_决赛阶段1_公开榜.zip"

INNER_FILES = [
    "run.py",
    "main.py",
    "requirements.txt",
    "README.md",
    "Makefile",
    "Dockerfile",
    "pyproject.toml",
    "LICENSE",
]
INNER_DIRS = ["src", "scripts", "experiments", "docs", "assets"]

# 初赛数据文件：保留原始数据即可复现，缓存由脚本自动重建
DATA_FILES = [
    "2022-2025处理后借阅信息.xlsx",
    "中文入藏（新书）.csv",
    "验证集发放.csv",
    "验证集说明与评分标准.docx",
]

# 决赛四份提交（顺序即平台提交顺序）
FINAL_SUBS = [
    "提交_公开榜.csv",
    "提交_隐藏榜.csv",
    "提交_隐藏榜_扰动A.csv",
    "提交_隐藏榜_扰动B.csv",
]

NOTE = f"""图书馆推荐算法竞赛（C 题）提交说明
队伍：{TEAM}

【压缩包内文件】
1. 提交结果_{TEAM}.csv     验证集推荐结果，956 行，每行 10 个索书号
2. 承诺书.pdf              已填写队伍信息并签署
3. 技术报告.pdf            13 页，含方法、四折消融与复现细节
4. 源代码与可运行包.zip    完整工程，解压后见 README.md

【运行方式】
解压“源代码与可运行包.zip”，执行：
    pip install -r requirements.txt
    python run.py
将在 outputs/result.csv 重新生成提交结果，并自动完成四项格式校验。

【环境】
Python 3.10+；无 GPU、无随机过程，单核 CPU 约 60 秒；依赖见 requirements.txt。

【数据】
包内 data/ 为组委会发放的原始数据，未作任何外部补充。
"""

NOTE_FINAL = f"""图书馆推荐算法竞赛（C 题）决赛提交说明
队伍：{TEAM}

【压缩包内文件（四份提交，与手册一一对应）】
1. 提交_公开榜.csv            对应 决赛_公开榜_输入.csv            545 行
2. 提交_隐藏榜.csv            对应 决赛_隐藏榜_输入.csv            575 行
3. 提交_隐藏榜_扰动A.csv      对应 决赛_隐藏榜_输入_扰动A_画像缺失.csv  575 行
4. 提交_隐藏榜_扰动B.csv      对应 决赛_隐藏榜_输入_扰动B_历史扰动.csv  575 行
5. 技术报告_决赛.pdf          决赛方案说明（方法、三套离线验证、消融、鲁棒性分析）
6. {INNER_FINAL.name}
   完整工程（源码 + 依赖 + 决赛原始数据 + 运行说明），解压后见 README.md 第十节。

四份提交为同一套模型在不同输入上的推理结果，未为扰动条件单独调参。

【复现方式】
解压内层包，执行：
    pip install -r requirements.txt
    python run.py --final
约 20 秒在 submission/final/ 生成上述四份 CSV，并逐份完成格式自查
（列名、行数、读者 ID、每行 ≤10 条、行内无重复、白名单、UTF-8 BOM）。
不传任何参数即为提交所用参数，输出与提交文件逐字节一致。

【环境】
Python 3.10+，无 GPU、无随机过程，8 核 16G 环境下四份合计约 20 秒，
远低于单份 40 分钟 / 三份 2 小时的上限；依赖见 requirements.txt。

【数据合规】
仅使用组委会发放的数据（初赛训练集 + 决赛输入 + 白名单），未接入任何外部数据。
"""

# 阶段 1 只交一份 CSV；平台附件不支持 .csv，故打成 zip，且只放根目录一层
NOTE_S1 = f"""图书馆推荐算法竞赛（C 题）决赛 · 阶段 1 提交说明
队伍：{TEAM}

【压缩包内文件】
提交_公开榜.csv    对应 决赛_公开榜_输入.csv，545 行
                   两列：读者ID,推荐索书号列表
                   每人 10 个索书号，按置信度降序，UTF-8 BOM

【自查结果】
读者 ID 与 决赛_公开榜_输入.csv 完全一致（含顺序）
每行恰好 10 条、行内无重复、全部落在 图书白名单_公开榜.csv 内
未推荐任何读者历史中已借过的索书号

【说明】
模型与阶段 2 所用完全一致，未针对公开榜单独调参。
完整源码、依赖清单与技术报告随阶段 2 提交。
"""


def add_tree(zf: zipfile.ZipFile, base: Path, rel: str | Path) -> int:
    """把 base/rel 递归写入 zip，保持相对路径。返回写入文件数。"""
    src = base / rel if str(rel) else base
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(base).as_posix())
            n += 1
    return n


def sync_submission() -> list[str]:
    """把 outputs/ 下的最新产物同步到 submission/（带队伍名）。返回同步成功的文件。"""
    SUB.mkdir(parents=True, exist_ok=True)
    plan = [
        (ROOT / "outputs" / "result.csv", SUB / f"提交结果_{TEAM}.csv"),
        (ROOT / "outputs" / "技术报告.pdf", SUB / "技术报告.pdf"),
        (ROOT / "outputs" / "承诺书.pdf", SUB / "承诺书.pdf"),
        (ROOT / "outputs" / "承诺书.docx", SUB / "承诺书.docx"),
    ]
    done = []
    for src, dst in plan:
        if not src.exists():
            print(f"  ! 缺少 {src.relative_to(ROOT)}，跳过")
            continue
        try:
            shutil.copyfile(src, dst)
            done.append(dst.name)
        except PermissionError:
            # 常见于 PDF 正被预览器 / Word 打开
            print(f"  ! {src.name} 被占用，沿用 submission 内旧版本")
    return done


def build_inner(path: Path, with_final: bool) -> tuple[int, int]:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        total = 0
        for f in INNER_FILES:
            if (ROOT / f).exists():
                zf.write(ROOT / f, f)
                total += 1
        for d in INNER_DIRS:
            if (ROOT / d).exists():
                total += add_tree(zf, ROOT, d)
        for f in DATA_FILES:
            if (ROOT / "data" / f).exists():
                zf.write(ROOT / "data" / f, f"data/{f}")
                total += 1
        if with_final and (ROOT / "02_数据").exists():
            total += add_tree(zf, ROOT, "02_数据")
    return total, path.stat().st_size


def build_classic() -> None:
    print("同步 outputs → submission")
    for name in sync_submission():
        print("  ✓", name)
    INNER.unlink(missing_ok=True)
    n, inner_sz = build_inner(INNER, with_final=False)
    with zipfile.ZipFile(OUTER, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("提交说明.txt", NOTE)
        zf.write(SUB / f"提交结果_{TEAM}.csv", f"提交结果_{TEAM}.csv")
        zf.write(SUB / "承诺书.pdf", "承诺书.pdf")
        zf.write(SUB / "技术报告.pdf", "技术报告.pdf")
        zf.write(INNER, INNER.name)
    print(f"\n内层包：{INNER.name}  {n} 个文件  {inner_sz / 1024 / 1024:.1f} MB")
    print(f"外层包：{OUTER.name}  {OUTER.stat().st_size / 1024 / 1024:.1f} MB")
    _list(OUTER)


def build_final() -> None:
    src = SUB / "final"
    missing = [f for f in FINAL_SUBS if not (src / f).exists()]
    if missing:
        raise SystemExit(f"缺少决赛提交件：{missing}\n请先运行 python run.py --final")
    INNER_FINAL.unlink(missing_ok=True)
    n, inner_sz = build_inner(INNER_FINAL, with_final=True)
    with zipfile.ZipFile(OUTER_FINAL, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("提交说明.txt", NOTE_FINAL)
        for f in FINAL_SUBS:
            zf.write(src / f, f)
        report = SUB / "技术报告_决赛.pdf"
        if report.exists():
            zf.write(report, report.name)
        else:
            print("  ! 缺少 技术报告_决赛.pdf，按无报告打包")
        zf.write(INNER_FINAL, INNER_FINAL.name)
    print(f"内层包：{INNER_FINAL.name}  {n} 个文件  {inner_sz / 1024 / 1024:.1f} MB")
    print(f"外层包：{OUTER_FINAL.name}  {OUTER_FINAL.stat().st_size / 1024 / 1024:.1f} MB")
    _list(OUTER_FINAL)


def build_stage1() -> None:
    """阶段 1：只交公开榜一份 CSV，压成单层 zip。"""
    sub = SUB / "final" / FINAL_SUBS[0]
    if not sub.exists():
        raise SystemExit(f"缺少 {sub}\n请先运行 python run.py --final")
    OUTER_S1.unlink(missing_ok=True)
    with zipfile.ZipFile(OUTER_S1, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("提交说明.txt", NOTE_S1)
        zf.write(sub, FINAL_SUBS[0])
    print(f"阶段 1 包：{OUTER_S1.name}  {OUTER_S1.stat().st_size / 1024:.0f} KB")
    _list(OUTER_S1)


def _list(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        print("\n根目录内容：")
        for i in zf.infolist():
            print(f"  {i.filename:<36} {i.file_size / 1024:>8.0f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s1", "classic", "final", "both"], default="final")
    args = ap.parse_args()
    if args.stage == "s1":
        build_stage1()
        return
    if args.stage in ("classic", "both"):
        build_classic()
    if args.stage in ("final", "both"):
        if args.stage == "both":
            print()
        build_final()


if __name__ == "__main__":
    main()
