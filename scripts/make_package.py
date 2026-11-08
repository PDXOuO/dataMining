"""打包提交物：生成一个可整包上传的 zip。

平台只允许上传单个附件，且要求"文件直接在压缩包根目录、不可嵌套文件夹"，
因此采用「外层包 = 三份预览件平铺 + 一个内层源码包」的结构：
预览件能被平台直接读取，源码包保持完整目录结构、解压即跑。

用法：python scripts/make_package.py
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
TEAM = "洛杉矶湖人"

OUTER = SUB / f"{TEAM}_提交包.zip"
INNER = SUB / f"{TEAM}_源代码与可运行包.zip"

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

# 数据文件：保留原始数据即可复现，缓存由脚本自动重建
DATA_FILES = [
    "2022-2025处理后借阅信息.xlsx",
    "中文入藏（新书）.csv",
    "验证集发放.csv",
    "验证集说明与评分标准.docx",
]

NOTE = f"""图书馆推荐算法竞赛（C 题）提交说明
队伍：{TEAM}

【压缩包内文件】
1. 提交结果_{TEAM}.csv     验证集推荐结果，956 行，每行 10 个索书号
2. 承诺书.pdf              已填写队伍信息并签署
3. 技术报告.pdf            13 页，含方法、四折消融与复现细节
4. 源代码与可运行包.zip    完整工程，解压后见 README.md

【运行方式】
解压"源代码与可运行包.zip"，执行：
    pip install -r requirements.txt
    python run.py
将在 outputs/result.csv 重新生成提交结果，并自动完成四项格式校验。

【环境】
Python 3.10+；无 GPU、无随机过程，单核 CPU 约 60 秒；依赖见 requirements.txt。

【数据】
包内 data/ 为组委会发放的原始数据，未作任何外部补充。
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


def build_inner() -> tuple[int, int]:
    with zipfile.ZipFile(INNER, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in INNER_FILES:
            if (ROOT / f).exists():
                zf.write(ROOT / f, f)
        total = len(INNER_FILES)
        for d in INNER_DIRS:
            if (ROOT / d).exists():
                total += add_tree(zf, ROOT, d)
        for f in DATA_FILES:
            if (ROOT / "data" / f).exists():
                zf.write(ROOT / "data" / f, f"data/{f}")
                total += 1
    return total, INNER.stat().st_size


def build_outer() -> int:
    with zipfile.ZipFile(OUTER, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("提交说明.txt", NOTE)
        zf.write(SUB / f"提交结果_{TEAM}.csv", f"提交结果_{TEAM}.csv")
        zf.write(SUB / "承诺书.pdf", "承诺书.pdf")
        zf.write(SUB / "技术报告.pdf", "技术报告.pdf")
        zf.write(INNER, INNER.name)
    return OUTER.stat().st_size


def main():
    print("同步 outputs → submission")
    for name in sync_submission():
        print("  ✓", name)
    INNER.unlink(missing_ok=True)
    n, inner_sz = build_inner()
    outer_sz = build_outer()
    print(f"内层包：{INNER.name}  {n} 个文件  {inner_sz / 1024 / 1024:.1f} MB")
    print(f"外层包：{OUTER.name}  {outer_sz / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(OUTER) as zf:
        print("\n根目录内容：")
        for i in zf.infolist():
            print(f"  {i.filename:<34} {i.file_size / 1024:>8.0f} KB")


if __name__ == "__main__":
    main()
