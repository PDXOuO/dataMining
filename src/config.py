"""集中配置：路径、字段名、常量与默认超参。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_XLSX = DATA_DIR / "2022-2025处理后借阅信息.xlsx"
VALID_CSV = DATA_DIR / "验证集发放.csv"
NEWBOOK_CSV = DATA_DIR / "中文入藏（新书）.csv"

for _d in (CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TOP_K = 10

# 回测切分点：主用 2024-09-01，交叉验证用 2024-03-01
SPLIT_MAIN = "2024-09-01"
SPLIT_ALT = "2024-03-01"
ANSWER_WINDOW_MONTHS = 14

# 打分权重（与组委会验证集说明一致）
W_NEW_HIT, W_NEW_RECALL = 0.5, 0.5
W_OLD_HIT, W_OLD_RECALL, W_OLD_NDCG = 0.4, 0.3, 0.3
W_NEW_GROUP, W_OLD_GROUP = 0.4, 0.6

# 方案 C 默认超参
DEFAULTS = {
    "tau_days": 365.0,        # 共现时间衰减常数
    "alpha_pop": 0.5,         # 热门惩罚指数
    "recency_halflife": 180.0,  # 历史项新鲜度半衰期
    "k_cf": 50,               # 共现候选数
    "k_content": 40,          # 内容候选数
    "k_group": 40,            # 分群候选数
    "k_rrf": 60,              # RRF 平滑常数
    "k_smooth": 20.0,         # 分群贝叶斯平滑强度
    "w_cf": 0.60,
    "w_content": 0.25,
    "w_group": 0.15,
}
