"""
Step 5 Ensemble (4-Model): lowlr + catboost + xgboost + xendcg
===============================================================
四个异构模型等权平均推理:
  - LightGBM LambdaRank (lowlr)    — 梯度 λ-rank
  - CatBoost YetiRank (catboost)   — 概率排序
  - XGBoost rank:ndcg (xgboost)    — NDCG 直接优化
  - LightGBM rank_xendcg (xendcg)  — 期望 NDCG 优化

异构性来源: 4 种不同损失函数 → 误差模式互不相关

输入: model_lowlr.txt / model_catboost.cbm / model_xgboost.json / model_xendcg.txt
输出: output/submission_ensemble_4model.csv
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle, gc, warnings
from collections import defaultdict
from catboost import CatBoost
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR,
    INFER_BATCH_SIZE,
    CUS_COLS, ART_COLS, INTER_COLS,
    SEED,
)
from utils import timer, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 5 Ensemble (4-Model): lowlr + catboost + xgboost + xendcg")
print("=" * 60)

# ============================================================
# 特征列 (23维, 与所有模型一致)
# ============================================================
CUS_COLS_CLEAN = ["age", "postal_le", "R_days", "n_unique_articles"]
ART_COLS_CLEAN = [
    "popularity_score", "price_log", "sales_log",
    "product_group_name_le", "product_type_name_le",
    "colour_group_name_le", "index_name_le",
]
ART_COLS_CLEAN += [f"text_emb_{i}" for i in [0, 1, 6, 7, 15, 16, 18]]
INTER_COLS_CLEAN = ["buy_count", "last_buy_days", "first_buy_days"]
CAND_COLS_CLEAN = ["cf_score", "price_match"]
FEAT_COLS_CLEAN = CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN + CAND_COLS_CLEAN

print(f"  特征: {len(FEAT_COLS_CLEAN)} 维")

# ============================================================
# 加载四个模型
# ============================================================
with timer("加载模型"):
    model_lowlr    = lgb.Booster(model_file=f"{OUTPUT_DIR}/model_lowlr.txt")
    model_catboost = CatBoost()
    model_catboost.load_model(f"{OUTPUT_DIR}/model_catboost.cbm")
    model_xgboost  = xgb.Booster()
    model_xgboost.load_model(f"{OUTPUT_DIR}/model_xgboost.json")
    model_xendcg   = lgb.Booster(model_file=f"{OUTPUT_DIR}/model_xendcg.txt")

print("  lowlr (LambdaRank) + catboost (YetiRank) + xgboost (rank:ndcg) + xendcg (rank_xendcg)")

# ============================================================
# 候选生成 & LTR 构建 (与训练脚本一致)
# ============================================================
def generate_candidates(user_hist, item_sim, art_pop, customers,
                        n_hist=12, n_pop=12):
    pop_list = sorted(art_pop, key=lambda x: -art_pop[x])[:n_pop]
    out = {}
    for cid in customers:
        cands = set()
        for aid in user_hist.get(cid, [])[:n_hist]:
            cands.add(aid)
        for aid in user_hist.get(cid, [])[:5]:
            if aid in item_sim:
                for rel, _ in item_sim[aid][:10]:
                    cands.add(rel)
        for aid in pop_list:
            cands.add(aid)
        out[cid] = list(cands)
    return out


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


# ============================================================
# 加载全量特征
# ============================================================
with timer("加载全量特征"):
    cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
    art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
    inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
    with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
        item_sim_full = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
        user_hist_full = pickle.load(f)

    art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
    pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()
known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} "
      f"({len(sub_cold)/len(sub_cids)*100:.1f}%)")

# ============================================================
# 分批集成推理 (四模型等权平均)
# ============================================================
all_preds = {}
with timer(f"分批集成推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        bnum = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(
            user_hist_full, item_sim_full, art_pop_full, batch_cids
        )
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        X_inf = inf_df[FEAT_COLS_CLEAN].values

        # 四个模型分别预测 → 等权平均
        s_lowlr  = model_lowlr.predict(X_inf)
        s_cat    = model_catboost.predict(X_inf)
        dinf     = xgb.DMatrix(X_inf)
        s_xgb    = model_xgboost.predict(dinf)
        s_xendcg = model_xendcg.predict(X_inf)

        inf_df["score"] = (
            np.float64(s_lowlr) + np.float64(s_cat) +
            np.float64(s_xgb) + np.float64(s_xendcg)
        ) / 4.0

        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds, X_inf, dinf
        gc.collect()
        print(f"  Batch {bnum}: {len(batch_cids):,} 用户完成")

# ============================================================
# 生成提交文件
# ============================================================
sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_ensemble_4model.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_ensemble_4model.csv  ({len(sub):,} 行)")
print(f"\n完成!")
