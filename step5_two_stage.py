"""
Step 5-B: 两阶段排序 (基于 fixed 评估方式)
=============================================
在 step5_fixed 的基础上叠加两阶段排序:
  - Stage 1 粗排: 46维全特征, 轻量模型(63叶/depth=6), 全量候选 → Top-30
  - Stage 2 精排: 46维全特征, 标准模型(LGB_PARAMS), Top-30 → Top-12

与 step5_fixed 一致的:
  - 时间切分: val(7天) → train(前5天) + holdout(后2天)
  - 早停: 粗排/精排均在各自 holdout 上监控 ndcg@12
  - 46维 FEAT_COLS / build_ltr_data
  - 冷启动诊断 / 全量推理提交

依赖: step1 → step2 → step4 (val_candidates.pkl)
       step6 (全量 _full 特征缓存)

输出: output/model_coarse.txt / output/model_fine.txt / output/submission_2stage.csv
"""
import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, LGB_PARAMS,
    CUS_COLS, ART_COLS, INTER_COLS, CAND_COLS, FEAT_COLS,
    INFER_BATCH_SIZE,
)
from utils import timer, mapk

warnings.filterwarnings("ignore")
print("=" * 60)
print("Step 5-B: 两阶段排序 (Coarse → Fine)")
print("=" * 60)

# ============================================================
# 粗排/精排 参数
# ============================================================
COARSE_TOP_K = 30

COARSE_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_at': [12],
    'learning_rate': 0.05,
    'num_leaves': 63,         # 轻量, 仅做粗筛
    'max_depth': 6,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
}

# 精排沿用原始 LGB_PARAMS
FINE_PARAMS = LGB_PARAMS.copy()

print(f"  粗排: {len(FEAT_COLS)}维, {COARSE_TOP_K}候选")
print(f"  精排: {len(FEAT_COLS)}维, 标准 LGB_PARAMS")

# ============================================================
# 读取数据
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

print(f"  val_txn 日期: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分: 前5天训练, 后2天留出验证
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn   = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"  val_train:  {len(val_train_txn):,} 行")
print(f"  val_holdout: {len(val_holdout_txn):,} 行")
assert val_train_txn["t_dat"].max() < val_holdout_txn["t_dat"].min(), "日期重叠!"

train_users   = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users  = sorted(train_users & holdout_users)
print(f"  共同用户: {len(common_users):,}")

val_train_gt   = val_train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_train_labels   = {cid: set(aids) for cid, aids in val_train_gt.items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

# ============================================================
# build_ltr_data (与原始 step5 完全一致)
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

    df["buy_count"]      = df["buy_count"].fillna(0)
    df["last_buy_days"]  = df["last_buy_days"].fillna(999)
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

    pos = df["label"].sum()
    print(f"  LTR pairs: {len(df):,}  pos: {pos:,}  neg: {len(df)-pos:,}  "
          f"ratio: {pos/len(df):.3f}")
    return df


# ============================================================
# 工具: 训练 + 预测 Top-K
# ============================================================
def train_lgb(ltr_train, ltr_valid, params, num_rounds=500, label=""):
    """训练 + 早停, 返回模型"""
    for df in [ltr_train, ltr_valid]:
        for c in FEAT_COLS:
            if df[c].dtype == "float64":
                df[c] = df[c].astype(np.float32)

    train_ds = lgb.Dataset(
        ltr_train[FEAT_COLS].values,
        label=ltr_train["label"].values,
        group=ltr_train.groupby("customer_id").size().values,
        feature_name=FEAT_COLS,
    )
    valid_ds = lgb.Dataset(
        ltr_valid[FEAT_COLS].values,
        label=ltr_valid["label"].values,
        group=ltr_valid.groupby("customer_id").size().values,
        feature_name=FEAT_COLS,
        reference=train_ds,
    )

    try:
        model = lgb.train(
            params, train_ds,
            num_boost_round=num_rounds,
            valid_sets=[train_ds, valid_ds],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0),
            ],
        )
    except Exception:
        cpu_params = {**params, "device": "cpu"}
        for k in ["gpu_platform_id", "gpu_device_id"]:
            cpu_params.pop(k, None)
        model = lgb.train(
            cpu_params, train_ds,
            num_boost_round=num_rounds,
            valid_sets=[train_ds, valid_ds],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0),
            ],
        )
    best = model.best_iteration if model.best_iteration > 0 else num_rounds
    print(f"  [{label}] 最佳轮数: {best}")
    return model


def predict_topk(ltr_df, model, k=12, score_col="score"):
    """预测 + 每人取 Top-K"""
    for c in FEAT_COLS:
        if ltr_df[c].dtype == "float64":
            ltr_df[c] = ltr_df[c].astype(np.float32)
    ltr_df[score_col] = model.predict(ltr_df[FEAT_COLS].values)
    return (
        ltr_df.sort_values(["customer_id", score_col], ascending=[True, False])
        .groupby("customer_id").head(k)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )


# ============================================================
# 构建 Train + Holdout LTR 数据
# ============================================================
print("\n[Train LTR]")
ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                           inter_feat, item_sim, user_hist)
print("[Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)

# ============================================================
# Stage 1: 粗排训练
# ============================================================
print("\n" + "=" * 60)
print("Stage 1: 粗排模型")
print("=" * 60)
model_coarse = train_lgb(ltr_train, ltr_holdout, COARSE_PARAMS,
                         num_rounds=300, label="Coarse")

# ============================================================
# Stage 2: 精排训练 (粗排筛选后)
# ============================================================
print("\n" + "=" * 60)
print(f"Stage 2: 精排模型 (粗排Top-{COARSE_TOP_K})")
print("=" * 60)

# 粗排对 Train 打分 → Top-K 筛选 → 精排训练集
ltr_train["coarse_score"] = model_coarse.predict(ltr_train[FEAT_COLS].values)
ltr_fine_train = (
    ltr_train.sort_values(["customer_id", "coarse_score"], ascending=[True, False])
    .groupby("customer_id").head(COARSE_TOP_K)
    .reset_index(drop=True)
)
# 粗排对 Holdout 打分 → Top-K 筛选 → 精排验证集
ltr_holdout["coarse_score"] = model_coarse.predict(ltr_holdout[FEAT_COLS].values)
ltr_fine_valid = (
    ltr_holdout.sort_values(["customer_id", "coarse_score"], ascending=[True, False])
    .groupby("customer_id").head(COARSE_TOP_K)
    .reset_index(drop=True)
)

pos_ft = ltr_fine_train["label"].sum()
pos_fv = ltr_fine_valid["label"].sum()
avg_cand = ltr_fine_train.groupby("customer_id").size().mean()
print(f"  精排训练: {len(ltr_fine_train):,} 行  pos={pos_ft:,}  均候选={avg_cand:.1f}")
print(f"  精排验证: {len(ltr_fine_valid):,} 行  pos={pos_fv:,}")

model_fine = train_lgb(ltr_fine_train, ltr_fine_valid, FINE_PARAMS,
                       num_rounds=500, label="Fine")

del ltr_fine_train, ltr_fine_valid; gc.collect()

# ============================================================
# 两阶段评估: Train + Holdout MAP@12
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]
train_cids   = [c for c in common_users if c in candidates]

with timer("验证评估"):
    # ---- Holdout 两阶段 ----
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    ltr_ho["coarse_score"] = model_coarse.predict(ltr_ho[FEAT_COLS].values)
    ltr_ho_topk = (
        ltr_ho.sort_values(["customer_id", "coarse_score"], ascending=[True, False])
        .groupby("customer_id").head(COARSE_TOP_K)
        .reset_index(drop=True)
    )
    preds_ho_2s = predict_topk(ltr_ho_topk, model_fine, k=12, score_col="fine_score")
    actuals_ho = [val_holdout_gt[c] for c in holdout_cids]
    preds_ho_l = [preds_ho_2s.get(c, []) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

    # ---- Train 两阶段 ----
    ltr_tr = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    ltr_tr["coarse_score"] = model_coarse.predict(ltr_tr[FEAT_COLS].values)
    ltr_tr_topk = (
        ltr_tr.sort_values(["customer_id", "coarse_score"], ascending=[True, False])
        .groupby("customer_id").head(COARSE_TOP_K)
        .reset_index(drop=True)
    )
    preds_tr_2s = predict_topk(ltr_tr_topk, model_fine, k=12, score_col="fine_score")
    actuals_tr = [val_train_gt[c] for c in train_cids]
    preds_tr_l = [preds_tr_2s.get(c, []) for c in train_cids]
    score_train = mapk(actuals_tr, preds_tr_l, k=12)

    # ---- 粗排单阶段 Holdout (对照) ----
    ltr_ho_c = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)
    preds_coarse = predict_topk(ltr_ho_c, model_coarse, k=12, score_col="coarse_score")
    preds_coarse_l = [preds_coarse.get(c, []) for c in holdout_cids]
    score_coarse = mapk(actuals_ho, preds_coarse_l, k=12)

# 流行度基线
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_ho    = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)
score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)

print(f"\n{'='*60}")
print(f"  Val 评估对比 (Holdout = 后2天)")
print(f"    ┌──────────────────────────┬──────────┬──────────┐")
print(f"    │                          │  Train   │ Holdout  │")
print(f"    ├──────────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline           │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 粗排单阶段                 │    —     │ {score_coarse:.5f}  │")
print(f"    │ 两阶段 (Coarse→Fine)       │ {score_train:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 两阶段 vs 粗排             │    —     │ {score_holdout-score_coarse:+.5f}  │")
print(f"    └──────────────────────────┴──────────┴──────────┘")
print(f"    过拟合: train-holdout = {score_train-score_holdout:.5f}")
print(f"{'='*60}")

# ============================================================
# 保存模型
# ============================================================
model_coarse.save_model(f"{OUTPUT_DIR}/model_coarse.txt")
model_fine.save_model(f"{OUTPUT_DIR}/model_fine.txt")
print(f"\n模型已保存: {OUTPUT_DIR}/model_coarse.txt  |  {OUTPUT_DIR}/model_fine.txt")

# ============================================================
# 全量两阶段推理 + 提交
# ============================================================
print("\n" + "=" * 60)
print("全量两阶段推理")
print("=" * 60)

cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
    item_sim_full = pickle.load(f)
with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
    user_hist_full = pickle.load(f)

art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full   = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]


def generate_candidates(user_hist, item_sim, art_pop, customers, n_hist=12, n_pop=12):
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


sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()

train_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in train_users]
print(f"  提交用户: {len(sub_cids):,}")
print(f"  冷启动 (无历史购买): {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

all_preds = {}
with timer(f"分批两阶段推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num  = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(user_hist_full, item_sim_full,
                                          art_pop_full, batch_cids)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)

        # Stage 1: 粗排 → Top-K
        inf_df["coarse_score"] = model_coarse.predict(inf_df[FEAT_COLS].values)
        inf_fine = (
            inf_df.sort_values(["customer_id", "coarse_score"], ascending=[True, False])
            .groupby("customer_id").head(COARSE_TOP_K)
            .reset_index(drop=True)
        )
        del inf_df; gc.collect()

        # Stage 2: 精排 → Top-12
        for c in FEAT_COLS:
            if inf_fine[c].dtype == "float64":
                inf_fine[c] = inf_fine[c].astype(np.float32)
        inf_fine["fine_score"] = model_fine.predict(inf_fine[FEAT_COLS].values)
        batch_preds = (
            inf_fine.sort_values(["customer_id", "fine_score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_fine, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_2stage.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_2stage.csv  ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*60}")
print(f"两阶段排序 — 结果汇总")
print(f"{'='*60}")
print(f"  Holdout MAP@12:")
print(f"    粗排单阶段:              {score_coarse:.5f}")
print(f"    两阶段 (Coarse→Fine):    {score_holdout:.5f}")
print(f"    两阶段 vs 粗排 增益:     {score_holdout-score_coarse:+.5f}")
print(f"  流行度 Baseline (holdout): {score_pop_ho:.5f}")
print(f"  两阶段过拟合:               {score_train-score_holdout:.5f}")
print(f"  冷启动用户比例:             {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*60}")
print(f"\n完成!")
