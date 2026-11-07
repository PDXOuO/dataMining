"""把 docs/report.md 渲染为 PDF（reportlab，中文宋体/黑体）。

用法：python scripts/make_report.py [--out outputs/技术报告.pdf]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Preformatted,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

FONTS = [
    ("Songti", "C:/Windows/Fonts/simsun.ttc", 0),
    ("Heiti", "C:/Windows/Fonts/simhei.ttf", None),
    ("Yahei", "C:/Windows/Fonts/msyh.ttc", 0),
]
for name, path, idx in FONTS:
    if not Path(path).exists():
        continue
    try:
        pdfmetrics.registerFont(
            TTFont(name, path) if idx is None else TTFont(name, path, subfontIndex=idx)
        )
    except Exception:
        pass

BODY_FONT = "Songti" if "Songti" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
HEAD_FONT = "Heiti" if "Heiti" in pdfmetrics.getRegisteredFontNames() else BODY_FONT
MONO_FONT = BODY_FONT

ACCENT = colors.HexColor("#1f3a5f")
GREY = colors.HexColor("#666666")

ST = {
    "title": ParagraphStyle("title", fontName=HEAD_FONT, fontSize=20, leading=30,
                            alignment=TA_CENTER, spaceAfter=6, textColor=ACCENT),
    "h2": ParagraphStyle("h2", fontName=HEAD_FONT, fontSize=14, leading=22,
                         spaceBefore=14, spaceAfter=6, textColor=ACCENT),
    "h3": ParagraphStyle("h3", fontName=HEAD_FONT, fontSize=11.5, leading=19,
                         spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#333333")),
    "body": ParagraphStyle("body", fontName=BODY_FONT, fontSize=10.5, leading=17.5,
                           alignment=TA_JUSTIFY, firstLineIndent=21, wordWrap='CJK', spaceAfter=5),
    "plain": ParagraphStyle("plain", fontName=BODY_FONT, fontSize=10.5, leading=17.5,
                            alignment=TA_JUSTIFY, spaceAfter=5, wordWrap='CJK'),
    "bullet": ParagraphStyle("bullet", fontName=BODY_FONT, fontSize=10.5, leading=17,
                             leftIndent=18, bulletIndent=6, spaceAfter=3, wordWrap='CJK'),
    "formula": ParagraphStyle("formula", fontName=BODY_FONT, fontSize=10.5, leading=18,
                              alignment=TA_CENTER, spaceBefore=5, spaceAfter=8, wordWrap='CJK',
                              backColor=colors.HexColor("#f4f6f9"),
                              borderPadding=6, textColor=colors.HexColor("#1a1a1a")),
    "caption": ParagraphStyle("caption", fontName=BODY_FONT, fontSize=9, leading=14,
                              alignment=TA_CENTER, textColor=GREY, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=BODY_FONT, fontSize=8.8, leading=13, wordWrap='CJK'),
    "code": ParagraphStyle("code", fontName=BODY_FONT, fontSize=9, leading=14.5,
                           leftIndent=12, backColor=colors.HexColor("#f5f5f5"),
                           borderPadding=6, textColor=colors.HexColor("#1a1a1a")),
    "cellh": ParagraphStyle("cellh", fontName=HEAD_FONT, fontSize=8.8, leading=13,
                            textColor=colors.white, wordWrap='CJK'),
}


def inline(text: str) -> str:
    """处理行内标记。"""
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="%s">\1</font>' % MONO_FONT, text)
    return text


def parse_table(lines: list[str], i: int):
    """从 i 开始解析表格，返回 (rows, next_index)。"""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not re.fullmatch(r"[-: ]+", "".join(cells)):
            rows.append(cells)
        i += 1
    return rows, i


def build_table(rows: list[list[str]], width: float):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = []
    for k, r in enumerate(rows):
        st = ST["cellh"] if k == 0 else ST["cell"]
        data.append([Paragraph(inline(c), st) for c in r])
    t = Table(data, colWidths=[width / ncol] * ncol, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def figure(path: Path, caption: str, width: float):
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, 12.5 * cm / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    parts = [img]
    if caption:
        parts.append(Paragraph(inline(caption), ST["caption"]))
    return KeepTogether(parts)


def render(md_path: Path, out_path: Path):
    lines = md_path.read_text(encoding="utf-8").split("\n")
    flow = []
    i = 0
    first_h1 = True
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        s = line.strip()

        if s == "<!--pagebreak-->":
            flow.append(PageBreak())
            i += 1
            continue
        if not s:
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # 跳过结束围栏
            if buf:
                pre = Preformatted("\n".join(buf), ST["code"])
                box = Table([[pre]], colWidths=[16.0 * cm], hAlign="LEFT")
                box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcdfe4")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                flow.append(box)
                flow.append(Spacer(1, 8))
            continue

        # 图片：![fig](path|caption)
        m = re.match(r"!\[\w*\]\(([^|)]+)\|?([^)]*)\)", s)
        if m:
            p = DOCS / m.group(1)
            if p.exists():
                flow.append(figure(p, m.group(2), 16.0 * cm))
            i += 1
            continue

        # 表格
        if s.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                flow.append(Spacer(1, 4))
                flow.append(build_table(rows, 16.0 * cm))
                flow.append(Spacer(1, 8))
            continue

        # 标题
        if s.startswith("# "):
            txt = s[2:].strip()
            if first_h1:
                flow.append(Paragraph(inline(txt), ST["title"]))
                flow.append(Spacer(1, 4))
                flow.append(Paragraph(
                    '<font color="#666666" size="9">图书馆推荐算法竞赛 · 初赛技术报告</font>',
                    ST["caption"]))
                flow.append(Spacer(1, 10))
                first_h1 = False
            else:
                flow.append(Paragraph(inline(txt), ST["h2"]))
            i += 1
            continue
        if s.startswith("## "):
            flow.append(Paragraph(inline(s[3:]), ST["h2"]))
            i += 1
            continue
        if s.startswith("### "):
            flow.append(Paragraph(inline(s[4:]), ST["h3"]))
            i += 1
            continue

        # 公式/引用行
        if s.startswith("> "):
            flow.append(Paragraph(inline(s[2:]), ST["formula"]))
            i += 1
            continue

        # 无序列表
        if s.startswith("- "):
            flow.append(Paragraph(inline(s[2:]), ST["bullet"], bulletText="•"))
            i += 1
            continue
        # 有序列表
        m = re.match(r"(\d+)\.\s+(.*)", s)
        if m:
            flow.append(Paragraph(inline(m.group(2)), ST["bullet"],
                                  bulletText=f"{m.group(1)}."))
            i += 1
            continue

        flow.append(Paragraph(inline(s), ST["body"]))
        i += 1

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 8)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"— {doc.page} —")
        canvas.setStrokeColor(colors.HexColor("#d9dde3"))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=1.9 * cm,
        title="图书馆个性化图书推荐技术报告", author="参赛队伍",
    )
    doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DOCS / "report.md"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "技术报告.pdf"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(Path(a.src), out)
    print("已生成：", out, f"({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
