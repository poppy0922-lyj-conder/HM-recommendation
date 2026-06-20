"""
Step 5: LightGBM LambdaRank 排序模型训练
========================================
使用 46 维全量特征，在验证集上:

  Phase 1: 时间切分 train(前5天) / holdout(后2天) → 早停训练 → 评估
  Phase 2: 全量 val 数据重新训练 → 保存最终模型

输出: output/model.txt
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle
import gc
import warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE,
    CUS_COLS, ART_COLS, INTER_COLS, FEAT_COLS,
    SEED,
)
from utils import timer, mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 5: LightGBM LambdaRank 排序模型训练")
print("=" * 60)

# ============================================================
# 读取所有中间数据
# ============================================================
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

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分: val → train(前5天) + holdout(后2天)
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"  val_train: {len(val_train_txn):,}  |  val_holdout: {len(val_holdout_txn):,}")
assert val_train_txn["t_dat"].max() < val_holdout_txn["t_dat"].min()

train_users = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users = sorted(train_users & holdout_users)
print(f"  共同用户: {len(common_users):,}")

val_train_labels = {
    cid: set(aids)
    for cid, aids in val_train_txn.groupby("customer_id")["article_id"].apply(list).items()
}
val_holdout_labels = {
    cid: set(aids)
    for cid, aids in val_holdout_txn.groupby("customer_id")["article_id"].apply(list).items()
}


# ============================================================
# build_ltr_data — LTR 训练数据构建
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist):
    """构建 LTR pairwise 训练/推理数据"""
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
    del rows
    gc.collect()

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
    del c_arr, a_arr, cf_map
    gc.collect()

    pos = df["label"].sum()
    print(f"  LTR pairs: {len(df):,}  pos: {pos:,}  neg: {len(df)-pos:,}  "
          f"ratio: {pos/len(df):.3f}")
    return df


# ============================================================
# Phase 1: 构建 LTR 数据
# ============================================================
print("\n[Phase 1 Train LTR]")
with timer("构建 LTR 数据"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist)

print("\n[Phase 1 Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)

# ============================================================
# GPU 检测 (回退 CPU)
# ============================================================
params = LGB_PARAMS.copy()
try:
    test_X = np.zeros((10, len(FEAT_COLS)), dtype=np.float32)
    test_y = np.zeros(10)
    test_ds = lgb.Dataset(test_X, label=test_y)
    lgb.train(params, test_ds, num_boost_round=1, callbacks=[lgb.log_evaluation(0)])
    print("\nGPU 可用 → 使用 GPU 训练")
except Exception:
    params["device"] = "cpu"
    for key in ["gpu_platform_id", "gpu_device_id"]:
        params.pop(key, None)
    print("\nGPU 不可用 → 自动切换 CPU 训练")

# ============================================================
# 准备训练数据 (46维全量特征)
# ============================================================
for c in FEAT_COLS:
    if ltr_train[c].dtype == "float64":
        ltr_train[c] = ltr_train[c].astype(np.float32)
    if ltr_holdout[c].dtype == "float64":
        ltr_holdout[c] = ltr_holdout[c].astype(np.float32)

X_train = ltr_train[FEAT_COLS].values
y_train = ltr_train["label"].values
groups_train = ltr_train.groupby("customer_id").size().values

X_valid = ltr_holdout[FEAT_COLS].values
y_valid = ltr_holdout["label"].values
groups_valid = ltr_holdout.groupby("customer_id").size().values

train_ds = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=FEAT_COLS)
valid_ds = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, feature_name=FEAT_COLS,
                        reference=train_ds)

print(f"\n  训练样本: {len(X_train):,}  |  验证样本: {len(X_valid):,}")
print(f"  特征维度: {len(FEAT_COLS)}")

# ============================================================
# Phase 1 训练 + 早停
# ============================================================
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(50),
]

with timer("Phase 1 LightGBM 训练 (含早停)"):
    model = lgb.train(
        params, train_ds,
        num_boost_round=500,
        valid_sets=[train_ds, valid_ds],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
    )

best_iter = model.best_iteration if model.best_iteration > 0 else 500
print(f"\n  最佳迭代轮数: {best_iter}")

del train_ds, valid_ds
gc.collect()

# ============================================================
# 特征重要性
# ============================================================
imp = pd.DataFrame({
    "feature": FEAT_COLS,
    "importance": model.feature_importance(),
}).sort_values("importance", ascending=False)

print("\nTop 15 特征重要性:")
print(imp.head(15).to_string(index=False))

# ============================================================
# Phase 1 评估: Train + Holdout MAP@12
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]

with timer("Phase 1 验证评估"):
    # Holdout
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS:
        if ltr_ho[c].dtype == "float64":
            ltr_ho[c] = ltr_ho[c].astype(np.float32)
    ltr_ho["score"] = model.predict(ltr_ho[FEAT_COLS].values)
    preds_ho = (
        ltr_ho.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals_ho = [val_holdout_labels[c] for c in holdout_cids]
    preds_ho_l = [preds_ho.get(c, []) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

    # Train
    ltr_tr = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS:
        if ltr_tr[c].dtype == "float64":
            ltr_tr[c] = ltr_tr[c].astype(np.float32)
    ltr_tr["score"] = model.predict(ltr_tr[FEAT_COLS].values)
    preds_tr = (
        ltr_tr.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    train_cids = [c for c in common_users if c in candidates]
    actuals_tr = [val_train_labels[c] for c in train_cids]
    preds_tr_l = [preds_tr.get(c, []) for c in train_cids]
    score_train_eval = mapk(actuals_tr, preds_tr_l, k=12)

# 基线 (流行度)
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  Phase 1 Val 评估 (46维全量特征):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 46维 LightGBM       │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    过拟合程度: train-holdout = {score_train_eval-score_holdout:.5f}")
print(f"    最佳迭代轮数: {best_iter}")
print(f"{'='*55}")

# ============================================================
# Phase 2: 全量 val 数据训练最终模型
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 2: 全量训练 (val全部7天, iterations={best_iter})")
print(f"{'='*60}")

val_all_labels = {
    cid: set(aids)
    for cid, aids in val_txn.groupby("customer_id")["article_id"].apply(list).items()
}
all_val_users = sorted(set(candidates.keys()) | set(val_all_labels.keys()))
all_val_users_in_cands = [u for u in all_val_users if u in candidates]
print(f"  全量训练用户: {len(all_val_users_in_cands):,}")

with timer("构建全量 LTR 数据"):
    ltr_full = build_ltr_data(
        {u: candidates[u] for u in all_val_users_in_cands},
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist
    )

for c in FEAT_COLS:
    if ltr_full[c].dtype == "float64":
        ltr_full[c] = ltr_full[c].astype(np.float32)

X_full = ltr_full[FEAT_COLS].values
y_full = ltr_full["label"].values
groups_full = ltr_full.groupby("customer_id").size().values

full_ds = lgb.Dataset(X_full, label=y_full, group=groups_full, feature_name=FEAT_COLS)

final_params = params.copy()
final_params["num_boost_round"] = best_iter

with timer(f"Phase 2 训练 (固定 {best_iter} 轮)"):
    final_model = lgb.train(
        final_params, full_ds,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(50)],
    )

final_model.save_model(f"{OUTPUT_DIR}/model.txt")
print(f"\n模型已保存: {OUTPUT_DIR}/model.txt")

del full_ds, X_full, y_full, ltr_full, model, ltr_train, ltr_holdout
gc.collect()

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"  结果汇总: LightGBM LambdaRank (46维)")
print(f"{'='*55}")
print(f"  Phase 1 Holdout MAP@12:          {score_holdout:.5f}")
print(f"  Phase 1 Train MAP@12:            {score_train_eval:.5f}")
print(f"  过拟合:                           {score_train_eval-score_holdout:.5f}")
print(f"  流行度 Baseline (holdout):        {score_pop_ho:.5f}")
print(f"  模型提升:                          +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:                     {best_iter}")
print(f"  特征维度:                          {len(FEAT_COLS)}")
print(f"{'='*55}")
print("\nStep 5 完成!")
