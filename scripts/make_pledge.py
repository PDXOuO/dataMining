"""生成参赛承诺书（DOCX 可编辑版 + PDF 可直接提交版）。

同一份内容源渲染两种格式，避免两者不一致。
用法：python scripts/make_pledge.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TITLE = "图书馆推荐算法竞赛参赛承诺书"

INTRO = [
    "本队全体成员已认真阅读并完全理解《图书馆推荐算法竞赛参赛指南》与《验证集说明"
    "与评分标准》的全部条款，现就参加本届竞赛郑重承诺如下：",
]

ITEMS = [
    ("一、作品原创性",
     "本队提交的推荐算法、源代码、技术报告及全部附件均系本队成员独立完成，不存在抄袭、"
     "代做或买卖参赛作品的情形。凡引用他人方法、代码或文献之处，均已在技术报告中明确标注"
     "来源。"),
    ("二、数据使用合规",
     "本队严格在组委会授权的数据范围内开展工作。未以任何形式探查、推断、反向利用初赛"
     "验证集的隐藏答案；未访问、猜测或使用决赛测试集；未尝试还原任何读者的真实身份，未对"
     "脱敏字段做关联推断。"),
    ("三、外部数据声明",
     "本队未使用任何外部数据。全部特征均来自组委会提供的数据集，未引入豆瓣图书信息、"
     "ISBN 元数据或其他第三方图书数据。若后续阶段引入外部数据，将在技术报告中逐项声明"
     "来源与使用方式，并确保该数据公开可获取。"),
    ("四、模型可复现",
     "本队提交的模型包完整可运行，包含完整源代码、依赖清单、数据读取脚本与一键运行说明。"
     "在技术报告所述软硬件环境下，组委会可完整复现本队提交的结果文件。模型不含随机初始化，"
     "重复运行输出一致。"),
    ("五、匿名性",
     "本队提交的技术报告、代码、文件名、注释及提交文件路径中，均不含队伍名称、学校名称、"
     "指导教师姓名及其他任何可识别身份的信息。"),
    ("六、评审纪律",
     "本队接受组委会规定的评审流程、评分口径与晋级规则，尊重专家评审意见，不干扰评审工作，"
     "不对评审过程提出无理异议。"),
]

CLOSING = (
    "本队充分知悉：若违反上述任何一项承诺，组委会有权取消本队参赛资格、撤销已获成绩，"
    "并由本队承担相应责任。"
)

TEAM = {
    "name": "洛杉矶湖人",
    "members": "江晟源　　喻思怡　　林世豪",
    "school": "成都信息工程大学",
    "teacher": "陈丁",
    "contact": "3156573374@qq.com",
    "date": "2026 年 11 月 8 日",
}

FIELDS = [
    ("队伍名称", TEAM["name"]),
    ("队员姓名", TEAM["members"]),
    ("指导教师", TEAM["teacher"]),
    ("所在学校", TEAM["school"]),
    ("联系方式", TEAM["contact"]),
]

SIGN = f"队伍负责人签字：__________________　　日期：{TEAM['date']}"


# ---------------- DOCX ----------------
def build_docx(path: Path):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.6)
    sec.bottom_margin = Cm(2.4)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(3.0)

    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    def para(text, size=12, bold=False, font="宋体", align=None, first_indent=None,
             space_before=0, space_after=6, line=1.55):
        p = doc.add_paragraph()
        if align is not None:
            p.alignment = align
        pf = p.paragraph_format
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        pf.line_spacing = line
        if first_indent is not None:
            pf.first_line_indent = Pt(first_indent)
        r = p.add_run(text)
        r.font.name = font
        r.font.size = Pt(size)
        r.bold = bold
        r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        return p

    para(TITLE, size=22, bold=True, font="黑体", align=WD_ALIGN_PARAGRAPH.CENTER,
         space_before=6, space_after=22, line=1.2)

    for t in INTRO:
        para(t, first_indent=24)

    for head, body in ITEMS:
        para(head, bold=True, font="黑体", space_before=10, space_after=4)
        para(body, first_indent=24)

    para(CLOSING, first_indent=24, space_before=10)

    para("", space_after=10)
    for label, line in FIELDS:
        para(f"{label}：{line}", space_before=2, space_after=2)

    para("", space_after=14)
    para(SIGN, align=WD_ALIGN_PARAGRAPH.RIGHT, space_before=8, space_after=2)

    doc.save(path)


# ---------------- PDF ----------------
def build_pdf(path: Path):
    for name, fp, idx in [("Song", "C:/Windows/Fonts/simsun.ttc", 0),
                          ("Hei", "C:/Windows/Fonts/simhei.ttf", None)]:
        if Path(fp).exists():
            try:
                pdfmetrics.registerFont(
                    TTFont(name, fp) if idx is None else TTFont(name, fp, subfontIndex=idx)
                )
            except Exception:
                pass
    body = "Song" if "Song" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    head = "Hei" if "Hei" in pdfmetrics.getRegisteredFontNames() else body

    st_title = ParagraphStyle("t", fontName=head, fontSize=19, leading=28,
                              alignment=TA_CENTER, spaceAfter=14, wordWrap="CJK")
    st_body = ParagraphStyle("b", fontName=body, fontSize=10.5, leading=18.5,
                             alignment=TA_JUSTIFY, firstLineIndent=21,
                             spaceAfter=2, wordWrap="CJK")
    st_head = ParagraphStyle("h", fontName=head, fontSize=11, leading=20,
                             alignment=TA_JUSTIFY, spaceBefore=5, spaceAfter=1,
                             wordWrap="CJK")
    st_right = ParagraphStyle("r", fontName=body, fontSize=11, leading=22,
                              alignment=TA_RIGHT, wordWrap="CJK")

    flow = [Paragraph(TITLE, st_title)]
    for t in INTRO:
        flow.append(Paragraph(t, st_body))
    for h, b in ITEMS:
        flow.append(Paragraph(h, st_head))
        flow.append(Paragraph(b, st_body))
    flow.append(Paragraph(CLOSING, st_body))
    flow.append(Spacer(1, 10))
    for label, line in FIELDS:
        flow.append(Paragraph(f"{label}：{line}", st_body))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(SIGN, st_right))

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=3.0 * cm, rightMargin=3.0 * cm,
        topMargin=2.1 * cm, bottomMargin=2.0 * cm,
        title=TITLE, author="参赛队伍",
    )
    doc.build(flow)


def main():
    d = OUT / "承诺书.docx"
    p = OUT / "承诺书.pdf"
    build_docx(d)
    build_pdf(p)
    print("已生成：", d, f"({d.stat().st_size / 1024:.0f} KB)")
    print("已生成：", p, f"({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
