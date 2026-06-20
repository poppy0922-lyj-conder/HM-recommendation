"""
Step 5 Clean: 基于特征消融结果精简至34维
==========================================
与 step5_fixed 的区别:
  - 从全46维中剔除12个噪声特征, 保留34维有效特征
  - 评估方式不变: 前5天训练/后2天留出/早停50轮

噪声特征 (12维):
  club_member_status_le, M_spend, avg_price_user (客户)
  graphical_appearance_name_le, garment_group_name_le  (类别编码)
  text_emb_3,5,8,9,10,11,13                           (文本嵌入)

来源: step5_feature_ablation_report.md (逐维度消融实验)

依赖: step1 → step2 → step4 (val_candidates.pkl)
      step6 (生成 _full 特征缓存, 用于最终提交)

输出: output/model_clean.txt / output/submission_clean.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE,
)
from utils import timer, mapk

warnings.filterwarnings("ignore")
print("=" * 60)
print("Step 5 Clean: 精简34维 (剔除12维噪声)")
print("=" * 60)

# ============================================================
# 精简后特征列定义 (34维)
# ============================================================
# 客户特征 (5维) — 剔除: club_member_status_le, M_spend, avg_price_user
CUS_COLS_CLEAN = [
    'age', 'postal_le', 'R_days', 'F_count', 'n_unique_articles',
]

# 商品特征 (24维)
ART_COLS_CLEAN = [
    # 商品统计 (6维) — 全部保留
    'avg_price', 'sales_count', 'n_buyers', 'popularity_score',
    'price_log', 'sales_log',
    # 类别编码 (5维) — 剔除: graphical_appearance_name_le, garment_group_name_le
    'product_group_name_le', 'product_type_name_le',
    'colour_group_name_le', 'index_name_le', 'section_name_le',
]
# 文本嵌入 (13维) — 保留: 0,1,2,4,6,7,12,14,15,16,17,18,19  剔除: 3,5,8,9,10,11,13
ART_COLS_CLEAN += [f'text_emb_{i}' for i in [0, 1, 2, 4, 6, 7, 12, 14, 15, 16, 17, 18, 19]]

# 交互特征 (3维) — 全部保留
INTER_COLS_CLEAN = ['buy_count', 'last_buy_days', 'first_buy_days']

# 候选特征 (2维) — 全部保留
CAND_COLS_CLEAN = ['cf_score', 'price_match']

# 全量34维
FEAT_COLS_CLEAN = CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN + CAND_COLS_CLEAN

print(f"  客户:      {len(CUS_COLS_CLEAN)}维  {CUS_COLS_CLEAN}")
print(f"  商品统计:  6维")
print(f"  类别编码:  5维")
print(f"  文本嵌入:  13维")
print(f"  交互:      {len(INTER_COLS_CLEAN)}维")
print(f"  候选:      {len(CAND_COLS_CLEAN)}维")
print(f"  总计:      {len(FEAT_COLS_CLEAN)}维")

# 为了 build_ltr_data merge 时获取完整列, 仍导入原始列名(仅用于 merge, 不影响训练)
from config import CUS_COLS, ART_COLS, INTER_COLS

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

print(f"  val_txn 日期范围: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分: val → train(前5天) + holdout(后2天)
# ============================================================
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

# ============================================================
# build_ltr_data — merge 用原始全量列, 但最终数据帧包含所有列
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

    pos = df["label"].sum()
    print(f"  LTR pairs: {len(df):,}  pos: {pos:,}  neg: {len(df)-pos:,}  "
          f"ratio: {pos/len(df):.3f}")
    return df


# ============================================================
# 构建 train/holdout 标签
# ============================================================
train_users = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users = sorted(train_users & holdout_users)
print(f"\n  共同用户: {len(common_users):,}  "
      f"(train独有={len(train_users - holdout_users):,}, holdout独有={len(holdout_users - train_users):,})")

val_train_gt = val_train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()

val_train_labels = {cid: set(aids) for cid, aids in val_train_gt.items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

# ============================================================
# 构建训练 LTR 数据
# ============================================================
print("\n[Train LTR]")
with timer("构建 LTR 数据"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist)

print("\n[Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)

# ============================================================
# GPU 检测
# ============================================================
params = LGB_PARAMS.copy()
try:
    test_X = np.zeros((10, len(FEAT_COLS_CLEAN)), dtype=np.float32)
    test_y = np.zeros(10)
    test_ds = lgb.Dataset(test_X, label=test_y)
    lgb.train(params, test_ds, num_boost_round=1, callbacks=[lgb.log_evaluation(0)])
    print("\nGPU 可用, 使用 GPU 训练")
except Exception:
    params["device"] = "cpu"
    for key in ["gpu_platform_id", "gpu_device_id"]:
        params.pop(key, None)
    print("\nGPU 不可用, 自动切换为 CPU 训练")

# ============================================================
# 准备训练数据 ★ 仅用精简34维 ★
# ============================================================
for c in FEAT_COLS_CLEAN:
    if ltr_train[c].dtype == "float64":
        ltr_train[c] = ltr_train[c].astype(np.float32)
    if ltr_holdout[c].dtype == "float64":
        ltr_holdout[c] = ltr_holdout[c].astype(np.float32)

X_train = ltr_train[FEAT_COLS_CLEAN].values
y_train = ltr_train["label"].values
groups_train = ltr_train.groupby("customer_id").size().values

X_valid = ltr_holdout[FEAT_COLS_CLEAN].values
y_valid = ltr_holdout["label"].values
groups_valid = ltr_holdout.groupby("customer_id").size().values

train_ds = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=FEAT_COLS_CLEAN)
valid_ds = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, feature_name=FEAT_COLS_CLEAN,
                        reference=train_ds)

print(f"\n  训练样本: {len(X_train):,}  |  验证样本: {len(X_valid):,}")

# ============================================================
# 训练 + 早停
# ============================================================
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(50),
]

with timer("LightGBM 训练 (含早停)"):
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
    "feature": FEAT_COLS_CLEAN,
    "importance": model.feature_importance(),
}).sort_values("importance", ascending=False)

print("\nTop 15 特征重要性:")
print(imp.head(15).to_string(index=False))

# ============================================================
# 评估: Train + Holdout MAP@12
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]

with timer("验证评估"):
    # ---- Holdout ----
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS_CLEAN:
        if ltr_ho[c].dtype == "float64":
            ltr_ho[c] = ltr_ho[c].astype(np.float32)
    ltr_ho["score"] = model.predict(ltr_ho[FEAT_COLS_CLEAN].values)
    preds_ho = (
        ltr_ho.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals_ho = [val_holdout_gt[c] for c in holdout_cids]
    preds_ho_l = [preds_ho.get(c, []) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

    # ---- Train ----
    ltr_tr = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS_CLEAN:
        if ltr_tr[c].dtype == "float64":
            ltr_tr[c] = ltr_tr[c].astype(np.float32)
    ltr_tr["score"] = model.predict(ltr_tr[FEAT_COLS_CLEAN].values)
    preds_tr = (
        ltr_tr.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    train_cids = [c for c in common_users if c in candidates]
    actuals_tr = [val_train_gt[c] for c in train_cids]
    preds_tr_l = [preds_tr.get(c, []) for c in train_cids]
    score_train_eval = mapk(actuals_tr, preds_tr_l, k=12)

# 基线 (流行度)
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  Val 评估对比 (34维精简 vs 46维全量):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 34维 LightGBM       │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    过拟合程度: train-holdout = {score_train_eval-score_holdout:.5f}")
print(f"    (参考: 46维过拟合=0.02435)")
print(f"{'='*55}")

# ============================================================
# 保存模型
# ============================================================
model.save_model(f"{OUTPUT_DIR}/model_clean.txt")
print(f"\n模型已保存: {OUTPUT_DIR}/model_clean.txt")

# ============================================================
# 全量推理 + 提交
# ============================================================
print("\n" + "=" * 60)
print("全量推理")
print("=" * 60)

# 读取全量特征
cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
    item_sim_full = pickle.load(f)
with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
    user_hist_full = pickle.load(f)

art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

# 候选生成
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

# 冷启动诊断
train_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in train_users]
print(f"  提交用户: {len(sub_cids):,}")
print(f"  冷启动 (无历史购买): {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

all_preds = {}
with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num  = start // INFER_BATCH_SIZE + 1

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
sub.to_csv(f"{OUTPUT_DIR}/submission_clean.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_clean.csv  ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"结果汇总: 34维精简模型")
print(f"{'='*55}")
print(f"  Holdout MAP@12 (诚实):    {score_holdout:.5f}")
print(f"  Train MAP@12 (训练集):     {score_train_eval:.5f}")
print(f"  过拟合:                    {score_train_eval-score_holdout:.5f}")
print(f"  流行度 Baseline (holdout): {score_pop_ho:.5f}")
print(f"  模型提升:                  +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:              {best_iter}")
print(f"  冷启动用户比例:            {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"  特征维度:                  34 (从46精简)")
print(f"  剔除噪声:                  club_member_status_le, M_spend, avg_price_user,")
print(f"                            graphical_appearance_name_le, garment_group_name_le,")
print(f"                            text_emb_3/5/8/9/10/11/13")
print(f"{'='*55}")
print(f"\n完成!")
