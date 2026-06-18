"""
负采样训练脚本
=============
在 train_after_ablation.py 基础上，唯一改动：
  - build_ltr_data 增加 neg_ratio 参数，训练时负采样控制正负比
  - holdout 评估和推理仍用全量（neg_ratio=None）

输出:
  model_neg_sampling.txt
  submission_neg_sampling.csv

使用方法:
  cd F:/HM-ablation-study
  python step_neg_sampling.py
"""

import sys, os

PROJECT_DIR = "F:/HM-recommendation"
sys.path.insert(0, PROJECT_DIR)

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR,
    CUS_COLS, ART_COLS, INTER_COLS,
    INFER_BATCH_SIZE, SEED,
)
from utils import timer, mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

print("=" * 60)
print("负采样训练: 23维 + 训练负采样")
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
CUS_COLS_MERGE = ['age', 'club_member_status_le', 'postal_le',
                  'R_days', 'F_count', 'M_spend', 'avg_price_user', 'n_unique_articles']

# ============================================================
# ⭐ 唯一新增参数: 负采样比
# ============================================================
NEG_RATIO = 5   # 正:负 = 1:NEG_RATIO (候选池平均~30时，约保留~150负样本/用户)

print(f"  特征: {len(FEAT_COLS_CLEAN)}维  |  负采样比: 1:{NEG_RATIO}")

# ============================================================
# LightGBM 参数
# ============================================================
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_at": [12],
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": 8,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "verbose": 1,
    "seed": SEED,
    "deterministic": True,
}

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

    cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
    art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
    inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
    with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
        item_sim_full = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
        user_hist_full = pickle.load(f)

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()
assert val_train_txn["t_dat"].max() < val_holdout_txn["t_dat"].min()

train_users = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users = sorted(train_users & holdout_users)

val_train_labels = {cid: set(aids) for cid, aids in val_train_txn.groupby("customer_id")["article_id"].apply(list).items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_txn.groupby("customer_id")["article_id"].apply(list).items()}

print(f"  val_train: {len(val_train_txn):,}  holdout: {len(val_holdout_txn):,}  共同用户: {len(common_users):,}")


# ============================================================
# build_ltr_data ⭐ 增加 neg_ratio 参数
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist, neg_ratio=None):
    """
    构建 LTR 数据。
    neg_ratio=None → 全量 (推理/评估)
    neg_ratio=N   → 训练时负采样，每用户保留 正样本数 * N 个负样本
    """
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
        clist = candidates[cid]

        # ⭐ 负采样逻辑
        if neg_ratio is not None and len(actual) > 0:
            pos = [a for a in clist if a in actual]
            neg = [a for a in clist if a not in actual]
            n_neg = min(len(neg), len(pos) * neg_ratio)
            if n_neg < len(neg):
                neg = list(np.random.choice(neg, size=n_neg, replace=False))
            for a in pos + neg:
                rows.append((cid, a, 1 if a in actual else 0))
        else:
            for a in clist:
                rows.append((cid, a, 1 if a in actual else 0))

    df = pd.DataFrame(rows, columns=["customer_id", "article_id", "label"])
    del rows; gc.collect()

    df = df.merge(cus_feat_df[CUS_COLS_MERGE + ["customer_id"]], on="customer_id", how="left")
    df = df.merge(art_feat_df[ART_COLS + ["article_id"]], on="article_id", how="left")
    df = df.merge(inter_feat_df, on=["customer_id", "article_id"], how="left")

    df["buy_count"] = df["buy_count"].fillna(0)
    df["last_buy_days"] = df["last_buy_days"].fillna(999)
    df["first_buy_days"] = df["first_buy_days"].fillna(999)

    c_arr = df["customer_id"].values; a_arr = df["article_id"].values
    df["cf_score"] = np.float32([cf_map.get(c, {}).get(a, 0.0) for c, a in zip(c_arr, a_arr)])
    df["price_match"] = (-np.abs(df["avg_price"].values - df["avg_price_user"].values)).astype(np.float32)
    del c_arr, a_arr, cf_map; gc.collect()

    return df


# ============================================================
# 构建 LTR (训练用负采样, holdout/推理用全量)
# ============================================================
print("\n[Train LTR] — 负采样")
with timer("构建训练 LTR (neg_ratio={})".format(NEG_RATIO)):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist, neg_ratio=NEG_RATIO)

print("\n[Holdout LTR] — 全量")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist, neg_ratio=None)

# ============================================================
# GPU 检测 + 训练
# ============================================================
try:
    test_X = np.zeros((10, len(FEAT_COLS_CLEAN)), dtype=np.float32)
    test_y = np.zeros(10)
    test_ds = lgb.Dataset(test_X, label=test_y)
    lgb.train(params, test_ds, num_boost_round=1, callbacks=[lgb.log_evaluation(0)])
    print("\nGPU 可用")
except Exception:
    params["device"] = "cpu"
    for key in ["gpu_platform_id", "gpu_device_id"]:
        params.pop(key, None)
    print("\nGPU 不可用 → CPU")

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

callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]

with timer("训练 (早停100)"):
    model = lgb.train(params, train_ds, num_boost_round=2000,
                      valid_sets=[train_ds, valid_ds],
                      valid_names=['train', 'valid'], callbacks=callbacks)

best_iter = model.best_iteration or 2000
print(f"\n最佳迭代轮数: {best_iter}")
del train_ds, valid_ds; gc.collect()

# ============================================================
# 评估
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]
train_cids = [c for c in common_users if c in candidates]
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]

with timer("评估"):
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS_CLEAN:
        if ltr_ho[c].dtype == "float64":
            ltr_ho[c] = ltr_ho[c].astype(np.float32)
    ltr_ho["score"] = model.predict(ltr_ho[FEAT_COLS_CLEAN].values)
    preds_ho = (ltr_ho.sort_values(["customer_id", "score"], ascending=[True, False])
                .groupby("customer_id").head(12)
                .groupby("customer_id")["article_id"].apply(list).to_dict())
    actuals_ho = [list(val_holdout_labels.get(c, set())) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, [preds_ho.get(c, []) for c in holdout_cids], k=12)

    ltr_tr = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist)
    for c in FEAT_COLS_CLEAN:
        if ltr_tr[c].dtype == "float64":
            ltr_tr[c] = ltr_tr[c].astype(np.float32)
    ltr_tr["score"] = model.predict(ltr_tr[FEAT_COLS_CLEAN].values)
    preds_tr = (ltr_tr.sort_values(["customer_id", "score"], ascending=[True, False])
                .groupby("customer_id").head(12)
                .groupby("customer_id")["article_id"].apply(list).to_dict())
    actuals_tr = [list(val_train_labels.get(c, set())) for c in train_cids]
    score_train_eval = mapk(actuals_tr, [preds_tr.get(c, []) for c in train_cids], k=12)

score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  评估 (负采样 1:{NEG_RATIO}):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 负采样模型          │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    最佳迭代轮数: {best_iter}")
print(f"{'='*55}")

# ============================================================
# Phase 2: 全量训练 (负采样)
# ============================================================
print(f"\nPhase 2: 全量训练 + 负采样 (1:{NEG_RATIO})")

val_all_labels = {cid: set(aids) for cid, aids in val_txn.groupby("customer_id")["article_id"].apply(list).items()}
all_val_users = sorted(set(candidates.keys()) | set(val_all_labels.keys()))
all_val_users_in_cands = [u for u in all_val_users if u in candidates]

with timer("构建全量 LTR (负采样)"):
    ltr_full = build_ltr_data(
        {u: candidates[u] for u in all_val_users_in_cands},
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist,
        neg_ratio=NEG_RATIO
    )
for c in FEAT_COLS_CLEAN:
    if ltr_full[c].dtype == "float64":
        ltr_full[c] = ltr_full[c].astype(np.float32)

X_full = ltr_full[FEAT_COLS_CLEAN].values
y_full = ltr_full["label"].values
groups_full = ltr_full.groupby("customer_id").size().values
full_ds = lgb.Dataset(X_full, label=y_full, group=groups_full, feature_name=FEAT_COLS_CLEAN)

with timer(f"Phase 2 训练 ({best_iter}轮)"):
    final_model = lgb.train(params, full_ds, num_boost_round=best_iter,
                            callbacks=[lgb.log_evaluation(50)])

model_path = os.path.join(OUT_DIR, "model_neg_sampling.txt")
final_model.save_model(model_path)
print(f"\n模型: {model_path}")
del full_ds, X_full, y_full, ltr_full, model, ltr_train, ltr_holdout; gc.collect()

# ============================================================
# 全量推理
# ============================================================
print("\n全量推理")
del cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist, candidates; gc.collect()

art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()

known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

def gen_pop_cands(uhist, isim, apop, cids, n_hist=12, n_pop=12):
    plist = sorted(apop, key=lambda x: -apop[x])[:n_pop]
    out = {}
    for cid in cids:
        cands = set()
        for a in uhist.get(cid, [])[:n_hist]:
            cands.add(a)
        for a in uhist.get(cid, [])[:5]:
            if a in isim:
                for rel, _ in isim[a][:10]:
                    cands.add(rel)
        for a in plist:
            cands.add(a)
        out[cid] = list(cands)
    return out

all_preds = {}
with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        bnum = start // INFER_BATCH_SIZE + 1
        batch_cands = gen_pop_cands(user_hist_full, item_sim_full, art_pop_full, batch_cids)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)
        inf_df["score"] = final_model.predict(inf_df[FEAT_COLS_CLEAN].values)
        batch_preds = (inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
                       .groupby("customer_id").head(12)
                       .groupby("customer_id")["article_id"].apply(list).to_dict())
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {bnum}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(lambda x: " ".join(all_preds.get(x, pop12_full)))
sub_path = os.path.join(OUT_DIR, "submission_neg_sampling.csv")
sub.to_csv(sub_path, index=False)
print(f"\n提交: {sub_path} ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"结果汇总: 负采样 1:{NEG_RATIO}")
print(f"{'='*55}")
print(f"  Holdout MAP@12:      {score_holdout:.5f}")
print(f"  Train MAP@12:        {score_train_eval:.5f}")
print(f"  流行度 Baseline:      {score_pop_ho:.5f}")
print(f"  模型提升:            +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:         {best_iter}")
print(f"  负采样比:            1:{NEG_RATIO}")
print(f"  特征维度:             {len(FEAT_COLS_CLEAN)}")
print(f"  冷启动:              {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*55}")
print("\n完成!")
