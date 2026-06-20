"""
H&M 推荐模型 — 全局配置

所有 step 共享的路径、参数、随机种子、特征列定义。
"""

import os

# ============ 随机种子 ============
SEED = 42

# ============ 路径 ============
# 原始数据目录 (从 Kaggle 下载的 CSV)
DATA_DIR = "E:/H&M_data"

# 中间处理结果 (Parquet / pickle)
PROCESSED_DIR = "E:/H&M_data/processed"

# 输出目录 (模型 / 提交文件)
OUTPUT_DIR = "F:/HM-recommendation/output"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 验证集划分 ============
VAL_DAYS = 7  # 最后7天作为验证集

# ============ LightGBM 参数 ============
LGB_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_at': [12],
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': 1,
    'seed': SEED,
    'deterministic': True,
}

# ============ 分批推理 ============
INFER_BATCH_SIZE = 50000

# ============ 特征列名常量 ============
# 客户特征 (8维)
CUS_COLS = [
    'age', 'club_member_status_le', 'postal_le',
    'R_days', 'F_count', 'M_spend', 'avg_price_user', 'n_unique_articles'
]

# 商品特征 (33维)
ART_COLS = [
    'avg_price', 'sales_count', 'n_buyers', 'popularity_score',
    'price_log', 'sales_log',
    'product_group_name_le', 'product_type_name_le',
    'graphical_appearance_name_le', 'colour_group_name_le',
    'index_name_le', 'section_name_le', 'garment_group_name_le'
]
ART_COLS += [f'text_emb_{i}' for i in range(20)]

# 交互特征 (3维)
INTER_COLS = ['buy_count', 'last_buy_days', 'first_buy_days']

# 候选特征 (2维)
CAND_COLS = ['cf_score', 'price_match']

# 全部特征 (46维)
FEAT_COLS = CUS_COLS + ART_COLS + INTER_COLS + CAND_COLS

# 文本嵌入维度
TFIDF_MAX_FEATURES = 300
SVD_N_COMPONENTS = 20

# Item-CF 参数
ITEM_CF_TOP_K = 30
ITEM_CF_MIN_CNT = 3

# 用户历史
USER_HIST_MAX = 30
