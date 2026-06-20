"""
Step 5 Full Train: 全量训练数据 + CPU 训练
============================================
基于 step5_clean_round2_cpu (当前最佳), 唯一改动:
  - Phase 1: val 评估 (5+2 时间切分, 早停, 获取 best_iter) — 与原版完全一致
  - Phase 2: 用全部 val 期 (7天) 重新训练最终模型, num_boost_round = best_iter
  - Phase 3: 用 Phase 2 的模型做全量推理提交

效果: 提交模型多 40% 训练数据 (7天 vs 5天), 无特征泄漏风险

23维精简特征 (从34维剔除11个增量噪声)
输出: output/model_full_train.txt / output/submission_full_train.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE,
)
from utils import timer, mapk

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
print("=" * 60)
print("Step 5 Full Train: 23维精简 + 全量val数据训练")
print("=" * 60)

# ============================================================
# 精简后特征列定义 (23维) — 与 clean_round2_cpu 完全一致
# ============================================================
CUS_COLS_CLEAN = [
    'age', 'postal_le', 'R_days', 'n_unique_articles',
]

ART_COLS_CLEAN = [
    'popularity_score', 'price_log', 'sales_log',
    'product_group_name_le', 'product_type_name_le',
    'colour_group_name_le', 'index_name_le',
]
ART_COLS_CLEAN += [f'text_emb_{i}' for i in [0, 1, 6, 7, 15, 16, 18]]

INTER_COLS_CLEAN = ['buy_count', 'last_buy_days', 'first_buy_days']
CAND_COLS_CLEAN = ['cf_score', 'price_match']

FEAT_COLS_CLEAN = CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN + CAND_COLS_CLEAN

print(f"  客户:      {len(CUS_COLS_CLEAN)}维  {CUS_COLS_CLEAN}")
print(f"  商品统计:  3维  popularity_score/price_log/sales_log")
print(f"  类别编码:  4维")
print(f"  文本嵌入:  7维  保留 0/1/6/7/15/16/18")
print(f"  交互:      {len(INTER_COLS_CLEAN)}维")
print(f"  候选:      {len(CAND_COLS_CLEAN)}维")
print(f"  总计:      {len(FEAT_COLS_CLEAN)}维 (从34维剔除11个增量噪声)")

from config import CUS_COLS, ART_COLS, INTER_COLS

# ============================================================
# 通用函数
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist):
    """构建 LTR 训练/推理数据"""
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

    if "label" in df.columns:
        pos = df["label"].sum()
        print(f"  LTR pairs: {len(df):,}  pos: {pos:,}  neg: {len(df)-pos:,}  "
              f"ratio: {pos/len(df):.3f}")
    return df


def generate_candidates(user_hist, item_sim, art_pop, customers, n_hist=12, n_pop=12):
    """候选生成: 历史 + Item-CF + 流行度"""
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


def prepare_ltr_for_training(ltr_df):
    """转换 float64 → float32, 提取 X/y/group"""
    for c in FEAT_COLS_CLEAN:
        if ltr_df[c].dtype == "float64":
            ltr_df[c] = ltr_df[c].astype(np.float32)
    X = ltr_df[FEAT_COLS_CLEAN].values
    y = ltr_df["label"].values
    groups = ltr_df.groupby("customer_id").size().values
    return X, y, groups


# ============================================================
# Phase 1: Val 评估 (与 clean_round2_cpu 完全一致)
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: Val 评估 (5天训练 + 2天验证, 含早停)")
print("=" * 60)

with timer("读取 val 数据"):
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

print(f"  val_txn 日期范围: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# 时间切分
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"\n  时间切分:")
print(f"    val_train:  {len(val_train_txn):,} 行  "
      f"({val_train_txn['t_dat'].min().date()} ~ {val_train_txn['t_dat'].max().date()})")
print(f"    val_holdout: {len(val_holdout_txn):,} 行  "
      f"({val_holdout_txn['t_dat'].min().date()} ~ {val_holdout_txn['t_dat'].max().date()})")

assert val_train_txn["t_dat"].max() < val_holdout_txn["t_dat"].min(), \
    "错误: train/holdout 日期有重叠!"

# 用户交集
train_users = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users = sorted(train_users & holdout_users)
print(f"\n  共同用户: {len(common_users):,}  "
      f"(train独有={len(train_users - holdout_users):,}, holdout独有={len(holdout_users - train_users):,})")

# 标签
val_train_gt = val_train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_train_labels = {cid: set(aids) for cid, aids in val_train_gt.items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

# 构建 LTR 数据
print("\n[Train LTR]")
with timer("构建 LTR 数据"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist)

print("\n[Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)

# 准备数据
X_train, y_train, groups_train = prepare_ltr_for_training(ltr_train)
X_valid, y_valid, groups_valid = prepare_ltr_for_training(ltr_holdout)

train_ds = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=FEAT_COLS_CLEAN)
valid_ds = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, feature_name=FEAT_COLS_CLEAN,
                        reference=train_ds)

print(f"\n  训练样本: {len(X_train):,}  |  验证样本: {len(X_valid):,}")

# CPU 训练参数
params = LGB_PARAMS.copy()
params["device"] = "cpu"
for key in ["gpu_platform_id", "gpu_device_id"]:
    params.pop(key, None)
print("\n使用 CPU 训练")

callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(50),
]

with timer("Phase 1 LightGBM 训练 (含早停)"):
    model_val = lgb.train(
        params, train_ds,
        num_boost_round=500,
        valid_sets=[train_ds, valid_ds],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
    )

best_iter = model_val.best_iteration if model_val.best_iteration > 0 else 500
print(f"\n  最佳迭代轮数: {best_iter}")

del train_ds, valid_ds, X_train, y_train; gc.collect()

# 特征重要性
imp = pd.DataFrame({
    "feature": FEAT_COLS_CLEAN,
    "importance": model_val.feature_importance(),
}).sort_values("importance", ascending=False)
print("\nTop 15 特征重要性:")
print(imp.head(15).to_string(index=False))

# ============================================================
# Phase 1 评估
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]

with timer("Phase 1 评估"):
    ltr_holdout["score"] = model_val.predict(ltr_holdout[FEAT_COLS_CLEAN].values)
    preds_ho = (
        ltr_holdout.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals_ho = [val_holdout_gt[c] for c in holdout_cids]
    preds_ho_l = [preds_ho.get(c, []) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

    ltr_train["score"] = model_val.predict(ltr_train[FEAT_COLS_CLEAN].values)
    preds_tr = (
        ltr_train.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    train_cids = [c for c in common_users if c in candidates]
    actuals_tr = [val_train_gt[c] for c in train_cids]
    preds_tr_l = [preds_tr.get(c, []) for c in train_cids]
    score_train_eval = mapk(actuals_tr, preds_tr_l, k=12)

art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  Phase 1 Val 评估 (5天训练 → 2天验证):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 23维 LightGBM       │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    过拟合程度: train-holdout = {score_train_eval-score_holdout:.5f}")
print(f"    最佳迭代轮数: {best_iter}")
print(f"{'='*55}")

# 释放 Phase 1 数据
del model_val, ltr_train, ltr_holdout, val_train_txn, val_holdout_txn
gc.collect()

# ============================================================
# Phase 2: 全量 val 数据训练最终模型 (7天全部数据, 不早停)
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 2: 全量训练 (val全部7天, num_boost_round={best_iter})")
print("=" * 60)

# 用全部 val_txn 构建标签
val_all_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_all_labels = {cid: set(aids) for cid, aids in val_all_gt.items()}

# 所有在 candidates + val_txn 中出现的用户
all_val_users = sorted(set(candidates.keys()) | set(val_all_labels.keys()))
all_val_users_in_cands = [u for u in all_val_users if u in candidates]
print(f"  全量训练用户: {len(all_val_users_in_cands):,}")

print("\n[Full LTR]")
with timer("构建全量 LTR 数据"):
    ltr_full = build_ltr_data(
        {u: candidates[u] for u in all_val_users_in_cands},
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist
    )

X_full, y_full, groups_full = prepare_ltr_for_training(ltr_full)
full_ds = lgb.Dataset(X_full, label=y_full, group=groups_full, feature_name=FEAT_COLS_CLEAN)
print(f"\n  全量训练样本: {len(X_full):,}  (Phase 1: {len(groups_train):,} 组, +{len(X_full)/max(len(groups_train),1)-1:.0%})")

with timer(f"Phase 2 LightGBM 训练 (固定 {best_iter} 轮)"):
    model = lgb.train(
        params, full_ds,
        num_boost_round=best_iter,
        valid_sets=[full_ds],
        valid_names=['full'],
        callbacks=[lgb.log_evaluation(50)],
    )

del full_ds, X_full, y_full, ltr_full; gc.collect()

model.save_model(f"{OUTPUT_DIR}/model_full_train.txt")
print(f"\n模型已保存: {OUTPUT_DIR}/model_full_train.txt")

# ============================================================
# Phase 3: 全量推理 + 提交
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: 全量推理")
print("=" * 60)

# 释放 val 数据, 加载全量特征
del cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist, candidates
gc.collect()

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

# 冷启动诊断
known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}")
print(f"  冷启动 (无历史购买): {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

all_preds = {}
with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(user_hist_full, item_sim_full, art_pop_full, batch_cids)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        inf_df["score"] = model.predict(inf_df[FEAT_COLS_CLEAN].values)
        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)

        del inf_df, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

# 生成提交
sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_full_train.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_full_train.csv  ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"结果汇总: 23维精简 + 全量val训练")
print(f"{'='*55}")
print(f"  Phase 1 Holdout MAP@12 (5天训练):  {score_holdout:.5f}")
print(f"  Phase 1 Train MAP@12:              {score_train_eval:.5f}")
print(f"  Phase 1 过拟合:                    {score_train_eval-score_holdout:.5f}")
print(f"  流行度 Baseline (holdout):         {score_pop_ho:.5f}")
print(f"  Phase 1 模型提升:                  +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:                      {best_iter}")
print(f"  Phase 2 训练数据:                  val全部7天 ({len(all_val_users_in_cands):,}用户)")
print(f"  提交冷启动比例:                    {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"  特征维度:                          23")
print(f"{'='*55}")
print(f"\n完成!")
