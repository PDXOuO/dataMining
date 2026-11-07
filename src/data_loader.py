"""数据加载与标准化。

三个数据源格式各不相同，且都有坑：
- 训练集 Excel 是「横表」：表头在第 3 行，数据从第 4 行开始，第 0 列是序号；
- 新书池 CSV 是 GBK（不是 UTF-8）；
- 验证集 CSV 是 UTF-8 with BOM，列表字段用 `|` 分隔。
"""
from __future__ import annotations

import re

import pandas as pd

from .config import CACHE_DIR, NEWBOOK_CSV, TRAIN_XLSX, VALID_CSV

# 院系名归并表：(关键词, 规范名)。规范名以验证集取值为准。
DEPT_RULES = [
    # 噪声归并：行政处室、校外人员等不构成有效推荐群体
    ("其他部门", "其他"),
    ("校外", "其他"),
    ("离退休", "其他"),
    ("后勤", "其他"),
    ("工会", "其他"),
    ("办公室", "其他"),
    ("处", "其他"),
    ("部", "其他"),
    ("中心", "其他"),
    ("空", "其他"),
    ("单位", "其他"),
    ("图书馆", "其他"),
    ("攀枝花", "其他"),
    # 学院归并：验证集与训练集对同一学院的写法不一致，需统一
    ("艺术", "艺术学院"),
    ("钒钛", "钒钛学院"),
    ("法学", "法学院"),
    ("外国语", "外国语学院"),
    ("外语", "外国语学院"),
    ("文学", "文学院"),
    ("人文社科", "人文社科学院"),
    ("经济", "经管学院"),
    ("经管", "经管学院"),
    ("管理", "经管学院"),
    ("数计", "数计学院"),
    ("数学与计算机", "数计学院"),
    ("计算机", "数计学院"),
    ("智造", "智造学院"),
    ("智能制造", "智造学院"),
    ("电气", "电气学院"),
    ("电信", "电气学院"),
    ("机械", "机械学院"),
    ("材料", "材料学院"),
    ("土建", "土建学院"),
    ("土木", "土建学院"),
    ("交通", "交通学院"),
    ("汽车", "交通学院"),
    ("资环", "资环学院"),
    ("资源环境", "资环学院"),
    ("生化", "生化学院"),
    ("生物与化学", "生化学院"),
    ("化学", "生化学院"),
    ("医", "医学院"),
    ("康养", "康养学院"),
    ("体育", "体育学院"),
    ("马克思", "马克思主义学院"),
    ("思政", "马克思主义学院"),
    ("教育", "教育学院"),
    ("继教", "继教学院"),
    ("创业", "创业学院"),
]

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ（）－．／",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()-./",
)


def normalize_callno(s) -> str | None:
    """索书号标准化：全角转半角、去空白、大写字母。

    推荐粒度是索书号（种），同一索书号可能有多个复本，必须统一写法才能正确聚合。
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).translate(_FULLWIDTH)
    t = re.sub(r"\s+", "", t)
    return t or None


def normalize_dept(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "未知"
    t = str(s).strip()
    if not t:
        return "未知"
    for kw, canon in DEPT_RULES:
        if kw in t:
            return canon
    return t


def normalize_major(s) -> str:
    """专业名归一化：去掉「本科/专科/研究生」等后缀与空白，便于跨表匹配。"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "未知"
    t = re.sub(r"\s+", "", str(s))
    t = re.sub(r"(本科|专科|研究生|专升本|二学位)$", "", t)
    return t or "未知"


def parse_pipe(s) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [x for x in (normalize_callno(p) for p in str(s).split("|")) if x]


def cls_of_callno(callno: str) -> str:
    """中图法一级类目：取开头的连续大写字母，再截到第一个字母。"""
    m = re.match(r"^([A-Z]+)", str(callno))
    if not m:
        return "?"
    return m.group(1)[0]


def cls_full_of_callno(callno: str) -> str:
    """索书号的分类号部分（`/` 之前的完整中图法分类号）。

    `TQ174.6/1288` -> `TQ174.6`。图书馆场景下同分类号意味着同一细分主题，
    是比一级类目强得多的内容信号。
    """
    head = str(callno).split("/")[0].strip()
    return head or "?"


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.pkl"


def load_train(use_cache: bool = True) -> pd.DataFrame:
    """训练集借阅流水，一行一条借阅记录。"""
    cp = _cache_path("train")
    if use_cache and cp.exists():
        return pd.read_pickle(cp)

    raw = pd.read_excel(TRAIN_XLSX, header=None)
    cols = raw.iloc[2, :].tolist()
    df = raw.iloc[3:, :].copy()
    df.columns = cols
    df = df.drop(columns=["序号"], errors="ignore").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "reader": df["读者条码"].astype(str).str.strip(),
            "call_no": df["索书号"].map(normalize_callno),
            "ts": pd.to_datetime(df["处理时间"], errors="coerce"),
            "dept": df["系别"].where(df["系别"].notna(), df["单位"]).map(normalize_dept),
            "major": df["专业"].fillna("未知").astype(str).map(normalize_major),
            "level": df["读者级别"].fillna("未知").astype(str).str.strip(),
            "title": df["题名"].astype(str).str.strip(),
            "author": df["责任者"].astype(str).str.strip(),
            "publisher": df["出版者"].astype(str).str.strip(),
            "pubyear": pd.to_numeric(df["出版日期"], errors="coerce"),
            "holding": df["馆藏地址"].astype(str).str.strip(),
        }
    )
    out = out[out["call_no"].notna() & out["ts"].notna()].reset_index(drop=True)
    out["cls1"] = out["call_no"].map(cls_of_callno)
    out["cls2"] = out["call_no"].map(cls_full_of_callno)
    out.to_pickle(cp)
    return out


def load_valid(use_cache: bool = True) -> pd.DataFrame:
    """验证集：956 行，每行一名读者，含历史列表。"""
    cp = _cache_path("valid")
    if use_cache and cp.exists():
        return pd.read_pickle(cp)

    df = pd.read_csv(VALID_CSV, encoding="utf-8-sig")
    df = df.rename(
        columns={            "读者ID": "rid",
            "用户类型": "utype",
            "性别": "gender",
            "读者类别": "level",
            "学院": "dept",
            "专业": "major",
            "年级": "grade",
            "办证日期": "card_date",
            "首次借阅日期": "first_ts",
            "最近借阅日期": "last_ts",
            "历史借阅册数": "hist_n",
            "历史借阅索书号数": "hist_uniq",
            "历史偏好分类Top3": "top_cls",
            "历史借阅索书号列表": "hist_list",
            "历史借阅题名列表": "hist_titles",
        }
    )
    df["dept"] = df["dept"].map(normalize_dept)
    df["major"] = df["major"].map(normalize_major)
    df["grade"] = df["grade"].astype(str).str.strip()
    df["hist_calls"] = df["hist_list"].map(parse_pipe)
    df["is_cold"] = df["hist_calls"].map(len) == 0
    df["top_cls_list"] = df["top_cls"].fillna("").map(
        lambda s: [x.strip() for x in str(s).split("|") if x.strip()]
    )
    df.to_pickle(cp)
    return df


def load_newbooks(use_cache: bool = True) -> pd.DataFrame:
    """新入藏图书池（GBK 编码）。"""
    cp = _cache_path("newbooks")
    if use_cache and cp.exists():
        return pd.read_pickle(cp)

    df = pd.read_csv(NEWBOOK_CSV, encoding="gbk")
    out = pd.DataFrame(
        {
            "call_no": df["索书号"].map(normalize_callno),
            "title": df["题名"].astype(str).str.strip(),
            "author": df["责任者"].fillna("").astype(str).str.strip(),
            "publisher": df["出版者"].fillna("").astype(str).str.strip(),
            "pubyear": pd.to_numeric(df["出版日期"], errors="coerce"),
            "copies": pd.to_numeric(df["复本量"], errors="coerce").fillna(1),
            "in_date": pd.to_datetime(df["处理日期"], errors="coerce"),
        }
    )
    out = out[out["call_no"].notna()].reset_index(drop=True)
    out["cls1"] = out["call_no"].map(cls_of_callno)
    out["cls2"] = out["call_no"].map(cls_full_of_callno)
    out = out.drop_duplicates("call_no", keep="last").reset_index(drop=True)
    out.to_pickle(cp)
    return out
