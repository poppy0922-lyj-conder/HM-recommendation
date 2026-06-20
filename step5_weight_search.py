"""
Step 5 Weight Search: 三模型集成权重网格搜索
============================================
在 val holdout 上搜索 LightGBM / CatBoost / XGBoost 的最优集成权重

score = (w1 * s_lowlr + w2 * s_cat + w3 * s_xgb) / (w1 + w2 + w3)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle, gc, time, warnings, itertools
from collections import defaultdict
from datetime import timedelta
from catboost import CatBoost
import lightgbm as lgb

from config import DATA_DIR, PROCESSED_DIR, CUS_COLS, ART_COLS, INTER_COLS
from utils import timer, mapk

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
print("=" * 60)
print("Step 5 Weight Search: 三模型权重网格搜索")
print("=" * 60)

# ============================================================
# 特征列 (23维)
# ============================================================
CUS_COLS_CLEAN = ['age', 'postal_le', 'R_days', 'n_unique_articles']
ART_COLS_CLEAN = [
    'popularity_score', 'price_log', 'sales_log',
    'product_group_name_le', 'product_type_name_le',
    'colour_group_name_le', 'index_name_le',
]
ART_COLS_CLEAN += [f'text_emb_{i}' for i in [0, 1, 6, 7, 15, 16, 18]]
INTER_COLS_CLEAN = ['buy_count', 'last_buy_days', 'first_buy_days']
CAND_COLS_CLEAN = ['cf_score', 'price_match']
FEAT_COLS_CLEAN = CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN + CAND_COLS_CLEAN

# ============================================================
# 加载模型
# ============================================================
with timer("加载模型"):
    model_lowlr    = lgb.Booster(model_file=f"{OUTPUT_DIR}/model_lowlr.txt")
    model_catboost = CatBoost()
    model_catboost.load_model(f"{OUTPUT_DIR}/model_catboost.cbm")
    model_xgboost  = xgb.Booster()
    model_xgboost.load_model(f"{OUTPUT_DIR}/model_xgboost.json")

# ============================================================
# 加载 val 数据
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist):
    cf_map = {}
    for cid in candidates:
        scores = defaultdict(float)
        for aid in user_hist.get(cid, [])[:5]:
            if aid in item_sim:
                for rel, sc in item_sim[aid][:10]:
                    scores[rel] += sc
        cf_map[cid] = dict(scores)

    rows = []
    for cid in candidates:
        actual = labels.get(cid, set())
        for aid in candidates[cid]:
            rows.append((cid, aid, 1 if aid in actual else 0))
    df = pd.DataFrame(rows, columns=["customer_id", "article_id", "label"])
    del rows; gc.collect()

    df = df.merge(cus_feat_df[CUS_COLS + ["customer_id"]], on="customer_id", how="left")
    df = df.merge(art_feat_df[ART_COLS + ["article_id"]], on="article_id", how="left")
    df = df.merge(inter_feat_df, on=["customer_id", "article_id"], how="left")

    df["buy_count"] = df["buy_count"].fillna(0)
    df["last_buy_days"] = df["last_buy_days"].fillna(999)
    df["first_buy_days"] = df["first_buy_days"].fillna(999)

    c_arr = df["customer_id"].values
    a_arr = df["article_id"].values
    df["cf_score"] = np.float32([
        cf_map.get(c, {}).get(a, 0.0) for c, a in zip(c_arr, a_arr)
    ])
    df["price_match"] = (
        -np.abs(df["avg_price"].values - df["avg_price_user"].values)
    ).astype(np.float32)
    del c_arr, a_arr, cf_map; gc.collect()

    return df


with timer("读取数据"):
    cus_feat = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat.parquet")
    art_feat = pd.read_parquet(f"{PROCESSED_DIR}/art_feat.parquet")
    inter_feat = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat.parquet")
    val_txn = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
    val_txn["t_dat"] = pd.to_datetime(val_txn["t_dat"])

    with open(f"{PROCESSED_DIR}/item_sim.pkl", "rb") as f:
        item_sim = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist.pkl", "rb") as f:
        user_hist = pickle.load(f)
    with open(f"{PROCESSED_DIR}/val_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)

# 时间切分: holdout = 最后2天
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

print(f"  holdout 时间范围: {val_holdout_txn['t_dat'].min().date()} ~ {val_holdout_txn['t_dat'].max().date()}")
print(f"  holdout 用户: {len(val_holdout_gt):,}")

# ============================================================
# 构建 holdout LTR 数据
# ============================================================
with timer("构建 holdout LTR 数据"):
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)

for c in FEAT_COLS_CLEAN:
    if ltr_ho[c].dtype == "float64":
        ltr_ho[c] = ltr_ho[c].astype(np.float32)

X_ho = ltr_ho[FEAT_COLS_CLEAN].values
print(f"  holdout 样本: {X_ho.shape[0]:,}")

# ============================================================
# 三模型预测
# ============================================================
with timer("三模型预测"):
    s_lowlr = model_lowlr.predict(X_ho)
    s_cat   = model_catboost.predict(X_ho)
    d_ho = xgb.DMatrix(X_ho)
    s_xgb   = model_xgboost.predict(d_ho)

print(f"  lowlr  range: [{s_lowlr.min():.4f}, {s_lowlr.max():.4f}]")
print(f"  cat    range: [{s_cat.min():.4f}, {s_cat.max():.4f}]")
print(f"  xgb    range: [{s_xgb.min():.4f}, {s_xgb.max():.4f}]")

# ============================================================
# 评估函数: 给定权重 → MAP@12
# ============================================================
ho_cids = sorted(set(ltr_ho["customer_id"].unique()) & set(val_holdout_gt.keys()))
ho_mask = ltr_ho["customer_id"].isin(ho_cids).values
ho_cid_arr = ltr_ho["customer_id"].values[ho_mask]
ho_s_lowlr = s_lowlr[ho_mask]
ho_s_cat = s_cat[ho_mask]
ho_s_xgb = s_xgb[ho_mask]

del X_ho, d_ho, s_lowlr, s_cat, s_xgb; gc.collect()

def eval_weights(w1, w2, w3):
    """给定权重, 计算 holdout MAP@12"""
    scores = (w1 * ho_s_lowlr + w2 * ho_s_cat + w3 * ho_s_xgb) / (w1 + w2 + w3)
    df = pd.DataFrame({
        "customer_id": ho_cid_arr,
        "article_id": ltr_ho["article_id"].values[ho_mask],
        "score": scores,
    })
    preds = (
        df.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals = [val_holdout_gt[c] for c in ho_cids]
    preds_l = [preds.get(c, []) for c in ho_cids]
    return mapk(actuals, preds_l, k=12)

# ============================================================
# 网格搜索
# ============================================================
# w1 (lowlr) 主导, 搜索更细; w2/w3 (cat/xgb) 副模型
w1_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
w2_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
w3_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# 基线: 等权
baseline_map = eval_weights(1.0, 1.0, 1.0)
print(f"\n  等权 (1:1:1) Holdout MAP@12: {baseline_map:.5f}")

results = []
total = len(w1_range) * len(w2_range) * len(w3_range)
print(f"\n网格搜索 {total} 组权重...")

for i, (w1, w2, w3) in enumerate(itertools.product(w1_range, w2_range, w3_range)):
    m = eval_weights(w1, w2, w3)
    results.append((w1, w2, w3, m))

# 排序
results.sort(key=lambda x: -x[3])

print(f"\n{'='*55}")
print(f"Top 20 权重组合:")
print(f"{'='*55}")
print(f"  {'w_lowlr':>8}  {'w_cat':>7}  {'w_xgb':>7}  {'MAP@12':>8}")
print(f"  {'-'*35}")
for w1, w2, w3, m in results[:20]:
    marker = " <--" if abs(w1-1.0)<0.01 and abs(w2-1.0)<0.01 and abs(w3-1.0)<0.01 else ""
    print(f"  {w1:8.1f}  {w2:7.1f}  {w3:7.1f}  {m:8.5f}{marker}")

best_w1, best_w2, best_w3, best_map = results[0]
improvement = best_map - baseline_map
print(f"\n  最优权重: w_lowlr={best_w1:.1f}  w_cat={best_w2:.1f}  w_xgb={best_w3:.1f}")
print(f"  最优 MAP@12: {best_map:.5f}  (vs 等权 {baseline_map:.5f}, 提升 {improvement:+.5f})")
print(f"\n  推理公式: score = ({best_w1:.1f}*s_lowlr + {best_w2:.1f}*s_cat + {best_w3:.1f}*s_xgb) / {best_w1+best_w2+best_w3:.1f}")

# 保存结果
pd.DataFrame(results, columns=["w_lowlr", "w_cat", "w_xgb", "MAP@12"]).to_csv(
    f"{OUTPUT_DIR}/weight_search_results.csv", index=False,
)
print(f"\n结果已保存: {OUTPUT_DIR}/weight_search_results.csv")
print(f"\n完成!")
