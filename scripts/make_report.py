"""把 docs/report.md 渲染为 PDF（reportlab，中文宋体/黑体）。

支持的扩展指令（HTML 注释形式，不会出现在输出中）：
  <!--pagebreak-->            分页
  <!--cap: 表 1　标题-->       给紧随其后的表格加表题（置于表格上方）
  <!--entitle: ...-->         英文标题（Times-Bold）
  <!--enabstract: ...-->      英文摘要（Times-Roman）
  <!--enkw: ...-->            英文关键词

公式行以 `> ` 开头，行尾写作 ` (1)` 时自动右对齐编号。

用法：python scripts/make_report.py [--out outputs/技术报告.pdf]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
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
LATIN = "Times-Roman"
LATIN_B = "Times-Bold"

ACCENT = colors.HexColor("#1f3a5f")
GREY = colors.HexColor("#666666")
WIDTH = 16.0 * cm
REPORT_NAME = "图书馆推荐算法竞赛（2026）· 初赛技术报告"
SUBTITLE = "赛题方向：高校图书馆个性化图书推荐　·　2026 年 11 月"

ST = {
    "title": ParagraphStyle("title", fontName=HEAD_FONT, fontSize=18.5, leading=28,
                            alignment=TA_CENTER, spaceAfter=4, textColor=ACCENT),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY_FONT, fontSize=9, leading=14,
                               alignment=TA_CENTER, textColor=GREY, spaceAfter=14),
    "h2": ParagraphStyle("h2", fontName=HEAD_FONT, fontSize=13.5, leading=21,
                         spaceBefore=14, spaceAfter=6, textColor=ACCENT, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=HEAD_FONT, fontSize=11, leading=18,
                         spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#333333"),
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName=BODY_FONT, fontSize=10.5, leading=17.5,
                           alignment=TA_JUSTIFY, firstLineIndent=21, wordWrap="CJK", spaceAfter=5),
    "abs": ParagraphStyle("abs", fontName=BODY_FONT, fontSize=9.8, leading=16.5,
                          alignment=TA_JUSTIFY, firstLineIndent=21, wordWrap="CJK", spaceAfter=4),
    "kw": ParagraphStyle("kw", fontName=BODY_FONT, fontSize=9.8, leading=16.5,
                         alignment=TA_JUSTIFY, wordWrap="CJK", spaceBefore=2, spaceAfter=8),
    "entitle": ParagraphStyle("entitle", fontName=LATIN_B, fontSize=13, leading=19,
                              alignment=TA_CENTER, spaceBefore=4, spaceAfter=7,
                              textColor=ACCENT),
    "enabs": ParagraphStyle("enabs", fontName=LATIN, fontSize=9.2, leading=13.6,
                            alignment=TA_JUSTIFY, spaceAfter=4),
    "enkw": ParagraphStyle("enkw", fontName=LATIN, fontSize=9.2, leading=13.6,
                           alignment=TA_JUSTIFY, spaceAfter=12),
    "bullet": ParagraphStyle("bullet", fontName=BODY_FONT, fontSize=10.5, leading=17,
                             leftIndent=18, bulletIndent=6, spaceAfter=3, wordWrap="CJK"),
    "ftxt": ParagraphStyle("ftxt", fontName=BODY_FONT, fontSize=9.8, leading=17,
                           alignment=TA_CENTER, wordWrap="CJK"),
    "fnum": ParagraphStyle("fnum", fontName=BODY_FONT, fontSize=9.8, leading=17,
                           alignment=TA_RIGHT),
    "tabcap": ParagraphStyle("tabcap", fontName=HEAD_FONT, fontSize=9.3, leading=14,
                             alignment=TA_CENTER, spaceBefore=8, spaceAfter=3,
                             textColor=colors.HexColor("#333333")),
    "caption": ParagraphStyle("caption", fontName=BODY_FONT, fontSize=9, leading=14,
                              alignment=TA_CENTER, textColor=GREY, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=BODY_FONT, fontSize=8.7, leading=12.8, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName=BODY_FONT, fontSize=9, leading=14.5,
                           leftIndent=12, backColor=colors.HexColor("#f5f5f5"),
                           borderPadding=6, textColor=colors.HexColor("#1a1a1a")),
    "cellh": ParagraphStyle("cellh", fontName=HEAD_FONT, fontSize=8.7, leading=12.8,
                            textColor=colors.white, wordWrap="CJK"),
    "ref": ParagraphStyle("ref", fontName=BODY_FONT, fontSize=8.8, leading=13.6,
                          alignment=TA_JUSTIFY, leftIndent=22, firstLineIndent=-22,
                          spaceAfter=3, wordWrap="CJK"),
}

DIRECTIVE = re.compile(r"^<!--([a-z]+):\s*(.*?)\s*-->$")
EQNUM = re.compile(r"^(.*?)\s*\((\d+)\)$")


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="%s">\1</font>' % MONO_FONT, text)
    return text


def parse_table(lines: list[str], i: int):
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


def build_formula(text: str, number: str | None):
    """公式带背景框；有编号时右对齐置于框内。"""
    if number is None:
        p = Paragraph(inline(text), ST["ftxt"])
        t = Table([[p]], colWidths=[WIDTH])
        inner = text
    else:
        t = Table(
            [[Paragraph(inline(text), ST["ftxt"]), Paragraph(f"({number})", ST["fnum"])]],
            colWidths=[WIDTH - 1.2 * cm, 1.2 * cm],
        )
        inner = text
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f9")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde2ea")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def figure(path: Path, caption: str, width: float):
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, 12.0 * cm / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    parts = [img]
    if caption:
        parts.append(Paragraph(inline(caption), ST["caption"]))
    return KeepTogether(parts)


def render(md_path: Path, out_path: Path):
    lines = md_path.read_text(encoding="utf-8").split("\n")
    flow: list = []
    i = 0
    first_h1 = True
    pending_cap: str | None = None

    while i < len(lines):
        s = lines[i].strip()

        if s == "<!--pagebreak-->":
            flow.append(PageBreak())
            i += 1
            continue
        if not s:
            i += 1
            continue

        if s.startswith("<!--"):
            m = DIRECTIVE.match(s)
            if m:
                tag, val = m.group(1), m.group(2)
                if tag == "cap":
                    pending_cap = val
                elif tag == "entitle":
                    flow.append(Paragraph(val, ST["entitle"]))
                elif tag == "enabstract":
                    flow.append(Paragraph("<b>Abstract.</b> " + val, ST["enabs"]))
                elif tag == "enkw":
                    flow.append(Paragraph("<b>Keywords:</b> " + val, ST["enkw"]))
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            if buf:
                pre = Preformatted("\n".join(buf), ST["code"])
                box = Table([[pre]], colWidths=[WIDTH], hAlign="LEFT")
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
        m = re.match(r"!\[\w*\]\(([^|)]+)\|([^)]*)\)", s)
        if m:
            p = DOCS / m.group(1)
            if p.exists():
                flow.append(figure(p, m.group(2), WIDTH))
            i += 1
            continue

        # 表格
        if s.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                parts = []
                if pending_cap:
                    parts.append(Paragraph(inline(pending_cap), ST["tabcap"]))
                    pending_cap = None
                parts.append(build_table(rows, WIDTH))
                flow.append(KeepTogether(parts))
                flow.append(Spacer(1, 9))
            continue

        # 标题
        if s.startswith("# "):
            txt = s[2:].strip()
            if first_h1:
                flow.append(Paragraph(inline(txt), ST["title"]))
                flow.append(Paragraph(SUBTITLE, ST["subtitle"]))
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

        # 公式行（可带编号）
        if s.startswith("> "):
            body = s[2:].strip()
            m = EQNUM.match(body)
            if m:
                flow.append(Spacer(1, 3))
                flow.append(build_formula(m.group(1), m.group(2)))
                flow.append(Spacer(1, 7))
            else:
                flow.append(Spacer(1, 3))
                flow.append(build_formula(body, None))
                flow.append(Spacer(1, 7))
            i += 1
            continue

        # 参考文献条目
        if re.match(r"^\[\d+\]", s):
            flow.append(Paragraph(inline(s), ST["ref"]))
            i += 1
            continue

        # 关键词行
        if s.startswith("**关键词**"):
            flow.append(Paragraph(inline(s), ST["kw"]))
            i += 1
            continue

        # 中文摘要（首行，紧跟标题）
        if s.startswith("**摘要**"):
            flow.append(Paragraph(inline(s), ST["abs"]))
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
            flow.append(Paragraph(inline(m.group(2)), ST["bullet"], bulletText=f"{m.group(1)}."))
            i += 1
            continue

        flow.append(Paragraph(inline(s), ST["body"]))
        i += 1

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 7.6)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, A4[1] - 1.42 * cm, REPORT_NAME)
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.42 * cm, "匿名参赛作品")
        canvas.setStrokeColor(colors.HexColor("#d9dde3"))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
        canvas.setFont(BODY_FONT, 8)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"— {doc.page} —")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.1 * cm, bottomMargin=1.9 * cm,
        title="图书馆个性化图书推荐技术报告", author="匿名参赛作品",
    )
    doc.build(flow, onFirstPage=decorate, onLaterPages=decorate)
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
