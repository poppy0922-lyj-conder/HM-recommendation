"""
Step 5 Expanded: 多通道扩大召回 + 负采样排序训练
=================================================
与 step5_train.py 的区别:
  1. 候选生成: 从 step4 的 ~30 候选 → 多通道 ~55 候选
     新增类目召回 + 复购召回, 原有通道参数扩大
  2. 训练时负采样: 1:5, 控制正负比
  3. 23维精简特征, lr=0.02, 早停训练

输出: output/model_expanded.txt / output/submission_expanded.csv
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE,
    CUS_COLS, ART_COLS, INTER_COLS,
    SEED,
)
from utils import timer, mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 5 Expanded: 多通道扩大召回 + 负采样")
print("=" * 60)

# ============ 特征列 (23维精简, 与最原始版本一致) ============
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
print(f"  特征: {len(FEAT_COLS_CLEAN)} 维 (23维精简)")

# ============ 扩大召回参数 ============
N_HIST = 20          # 历史商品 (原 12)
N_CF_SEED = 8        # CF 种子 (原 5)
N_CF_TOP = 15        # 每种子相似 (原 10)
N_POP = 25           # 流行度 (原 12)
N_CATE = 3           # 类目数
N_CATE_TOP = 10      # 每类目商品
MAX_CANDS = 55       # 候选上限 (原 ~30)
NEG_RATIO = 5        # 训练负采样比

print(f"  召回: hist={N_HIST} cf={N_CF_SEED}x{N_CF_TOP} "
      f"cate={N_CATE}x{N_CATE_TOP} pop={N_POP}")
print(f"  候选上限: {MAX_CANDS}/用户  负采样: 1:{NEG_RATIO}")

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
        candidates_orig = pickle.load(f)

orig_avg = sum(len(v) for v in candidates_orig.values()) / max(len(candidates_orig), 1)
print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")
print(f"  原始候选(step4): 平均 {orig_avg:.1f}/用户")

# ============================================================
# 构建类目 + 复购索引
# ============================================================
with timer("构建辅助索引"):
    art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
    art_raw = pd.read_parquet(f"{PROCESSED_DIR}/articles.parquet")
    art_info = art_feat[["article_id"]].copy()
    for col in ["product_group_name", "garment_group_name"]:
        art_info[col] = art_raw[col].values if col in art_raw.columns else ""
    art_info["cate_key"] = list(zip(
        art_info["product_group_name"].fillna(""),
        art_info["garment_group_name"].fillna(""),
    ))
    del art_raw
    cate_map = defaultdict(list)
    for _, row in art_info.iterrows():
        cate_map[row["cate_key"]].append(row["article_id"])
    cate_top = {
        k: sorted(v, key=lambda x: -art_pop.get(x, 0))[:N_CATE_TOP]
        for k, v in cate_map.items()
    }

    train_txn = pd.read_parquet(f"{PROCESSED_DIR}/train_txn.parquet")
    train_txn["t_dat"] = pd.to_datetime(train_txn["t_dat"])
    m = train_txn[["customer_id", "article_id"]].merge(
        art_info[["article_id", "cate_key"]], on="article_id", how="left"
    )
    ucc = m.groupby(["customer_id", "cate_key"]).size().reset_index(name="cnt")
    ucc = ucc.sort_values(["customer_id", "cnt"], ascending=[True, False])
    user_cate = {}
    for cid, grp in ucc.groupby("customer_id"):
        user_cate[cid] = [
            k for k in grp["cate_key"].tolist()
            if k in cate_top
        ][:N_CATE]
    del m, ucc

    rebuy = train_txn.groupby(["customer_id", "article_id"]).size().reset_index(name="cnt")
    user_rebuy = defaultdict(list)
    for _, row in rebuy[rebuy["cnt"] >= 2].iterrows():
        user_rebuy[row["customer_id"]].append(row["article_id"])
    del rebuy, train_txn
    gc.collect()
    print(f"  类目数: {len(cate_top):,}  复购用户: {len(user_rebuy):,}")

# ============================================================
# 多通道候选生成
# ============================================================
def gen_cands(uhist, isim, apop, cids, ucate, ctop, urebuy, mx=MAX_CANDS):
    plist = sorted(apop, key=lambda x: -apop[x])[:N_POP]
    out = {}
    for cid in cids:
        sc = {}
        hist = uhist.get(cid, [])
        cold = len(hist) == 0
        # 通道1: 历史
        for r, a in enumerate(hist[:N_HIST]):
            sc[a] = max(sc.get(a, 0), 100.0 - r)
        # 通道2: Item-CF
        cfs = defaultdict(float)
        for a in hist[:N_CF_SEED]:
            if a in isim:
                for rel, s in isim[a][:N_CF_TOP]:
                    cfs[rel] += s
        for r, a in enumerate(sorted(cfs, key=lambda x: -cfs[x])[:30]):
            sc[a] = max(sc.get(a, 0), 80.0 - r * 0.5)
        # 通道3: 类目
        if not cold:
            for ci, ck in enumerate(ucate.get(cid, [])):
                for ii, a in enumerate(ctop.get(ck, [])[:N_CATE_TOP]):
                    sc[a] = max(sc.get(a, 0), 60.0 - ci * 3 - ii * 0.3)
        # 通道4: 复购
        if not cold:
            for r, a in enumerate(urebuy.get(cid, [])[:10]):
                sc[a] = max(sc.get(a, 0), 90.0 - r)
        # 通道5: 流行度
        for r, a in enumerate(plist):
            if a not in sc:
                sc[a] = 20.0 - r * 0.3
        # 冷启动兜底: 用全局热门类目
        if cold:
            for ci, ck in enumerate(list(ctop.keys())[:15]):
                for ii, a in enumerate(ctop[ck][:5]):
                    if a not in sc:
                        sc[a] = 15.0 - ci * 0.5 - ii * 0.2
        sorted_cands = sorted(sc.items(), key=lambda x: -x[1])
        out[cid] = [a for a, _ in sorted_cands[:mx]]
    return out

val_cids = val_txn["customer_id"].unique().tolist()
print(f"\n验证集用户: {len(val_cids):,}")

with timer("多通道候选生成"):
    candidates = gen_cands(user_hist, item_sim, art_pop, val_cids,
                           user_cate, cate_top, user_rebuy)

tot = sum(len(v) for v in candidates.values())
avg_new = tot / max(len(candidates), 1)
print(f"  候选: {len(candidates):,} 用户  总 {tot:,}  "
      f"平均 {avg_new:.1f}/用户 (原 {orig_avg:.1f}, {avg_new/orig_avg:.1f}x)")

# ============================================================
# 时间切分: val → train(前6天) + holdout(后1天)
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"\n时间切分: train={len(val_train_txn):,}  holdout={len(val_holdout_txn):,}")
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
# build_ltr_data (支持负采样)
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist, neg_ratio=None):
    """neg_ratio=None → 全量(推理/评估). neg_ratio=N → 负采样(训练)"""
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

    df = df.merge(cus_feat_df[CUS_COLS + ["customer_id"]], on="customer_id", how="left")
    df = df.merge(art_feat_df[ART_COLS + ["article_id"]], on="article_id", how="left")
    df = df.merge(inter_feat_df, on=["customer_id", "article_id"], how="left")

    df["buy_count"] = df["buy_count"].fillna(0)
    df["last_buy_days"] = df["last_buy_days"].fillna(999)
    df["first_buy_days"] = df["first_buy_days"].fillna(999)

    ca = df["customer_id"].values
    aa = df["article_id"].values
    df["cf_score"] = np.float32([
        cf_map.get(c, {}).get(a, 0.0) for c, a in zip(ca, aa)
    ])
    df["price_match"] = (
        -np.abs(df["avg_price"].values - df["avg_price_user"].values)
    ).astype(np.float32)
    del ca, aa, cf_map; gc.collect()

    pos = df["label"].sum()
    print(f"  LTR pairs: {len(df):,}  pos: {pos:,}  neg: {len(df)-pos:,}  "
          f"ratio: {pos/max(len(df)-pos,1):.3f}")
    return df

# ============================================================
# 构建 LTR 数据
# ============================================================
print("\n[Train LTR]")
with timer("构建 LTR 数据 (训练)"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist, neg_ratio=NEG_RATIO)
print("\n[Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist, neg_ratio=None)

# ============================================================
# GPU 检测 (回退 CPU)
# ============================================================
params = LGB_PARAMS.copy()
params["learning_rate"] = 0.02  # 更低学习率
try:
    test_X = np.zeros((10, len(FEAT_COLS_CLEAN)), dtype=np.float32)
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
# 准备训练数据 (23维精简特征)
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

train_ds = lgb.Dataset(X_train, label=y_train, group=groups_train,
                       feature_name=FEAT_COLS_CLEAN)
valid_ds = lgb.Dataset(X_valid, label=y_valid, group=groups_valid,
                       feature_name=FEAT_COLS_CLEAN, reference=train_ds)
print(f"\n训练样本: {len(X_train):,}  |  验证样本: {len(X_valid):,}")
print(f"特征维度: {len(FEAT_COLS_CLEAN)}")

# ============================================================
# Phase 1 训练 + 早停
# ============================================================
callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(50),
]

with timer("LightGBM 训练 (含早停)"):
    model = lgb.train(
        params, train_ds,
        num_boost_round=2000,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

best_iter = model.best_iteration if model.best_iteration > 0 else 500
print(f"\n最佳迭代轮数: {best_iter}")

del train_ds, valid_ds; gc.collect()

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
# Holdout + Train 评估
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]
train_cids = [c for c in common_users if c in candidates]

with timer("验证评估"):
    ltr_ho = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist, neg_ratio=None)
    for c in FEAT_COLS_CLEAN:
        if ltr_ho[c].dtype == "float64":
            ltr_ho[c] = ltr_ho[c].astype(np.float32)
    ltr_ho["score"] = model.predict(ltr_ho[FEAT_COLS_CLEAN].values)
    preds_ho = (
        ltr_ho.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals_ho = [val_holdout_labels[c] for c in holdout_cids]
    preds_ho_l = [preds_ho.get(c, []) for c in holdout_cids]
    score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

    ltr_tr = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                            inter_feat, item_sim, user_hist, neg_ratio=None)
    for c in FEAT_COLS_CLEAN:
        if ltr_tr[c].dtype == "float64":
            ltr_tr[c] = ltr_tr[c].astype(np.float32)
    ltr_tr["score"] = model.predict(ltr_tr[FEAT_COLS_CLEAN].values)
    preds_tr = (
        ltr_tr.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals_tr = [val_train_labels[c] for c in train_cids]
    preds_tr_l = [preds_tr.get(c, []) for c in train_cids]
    score_train_eval = mapk(actuals_tr, preds_tr_l, k=12)

pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  Val 评估对比 (23维 + 扩大召回 + 负采样):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 23维 Expanded      │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    候选: {avg_new:.1f}/用户 (原{orig_avg:.1f})  负采样: 1:{NEG_RATIO}")
print(f"    过拟合: train-holdout = {score_train_eval-score_holdout:.5f}")
print(f"{'='*55}")

# ============================================================
# 保存模型
# ============================================================
model.save_model(f"{OUTPUT_DIR}/model_expanded.txt")
print(f"\n模型已保存: {OUTPUT_DIR}/model_expanded.txt")

# ============================================================
# Phase 2: 全量 val 数据重新训练固定轮数
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 2: 全量训练 (val 全部 {VAL_DAYS} 天, iterations={best_iter})")
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
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist,
        neg_ratio=NEG_RATIO,
    )

for c in FEAT_COLS_CLEAN:
    if ltr_full[c].dtype == "float64":
        ltr_full[c] = ltr_full[c].astype(np.float32)

X_full = ltr_full[FEAT_COLS_CLEAN].values
y_full = ltr_full["label"].values
groups_full = ltr_full.groupby("customer_id").size().values
full_ds = lgb.Dataset(X_full, label=y_full, group=groups_full,
                      feature_name=FEAT_COLS_CLEAN)

final_params = params.copy()
final_params.pop("num_boost_round", None)

with timer(f"Phase 2 训练 (固定 {best_iter} 轮)"):
    final_model = lgb.train(
        final_params, full_ds,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(50)],
    )

final_model.save_model(f"{OUTPUT_DIR}/model_expanded.txt")
print(f"最终模型已保存: {OUTPUT_DIR}/model_expanded.txt")

del full_ds, X_full, y_full, ltr_full, model
gc.collect()

# ============================================================
# 全量推理
# ============================================================
print(f"\n{'='*60}")
print("全量推理")
print(f"{'='*60}")

with timer("读取全量数据"):
    cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
    art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
    inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
    with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
        item_sim_full = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
        user_hist_full = pickle.load(f)

    # 全量商品的类目/复购索引 (需要 articles_full + train_txn_full)
    art_full_raw = pd.read_parquet(f"{PROCESSED_DIR}/articles.parquet")
    # 全量商品的类目信息与 val 一致, 用相同的 cate_top
    # 读取 full train_txn 构建用户类目
    full_train = pd.read_parquet(f"{PROCESSED_DIR}/train_txn.parquet")
    full_val = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
    full_txn_all = pd.concat([full_train, full_val], ignore_index=True)
    full_txn_all["t_dat"] = pd.to_datetime(full_txn_all["t_dat"])
    del full_train, full_val; gc.collect()

    art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()

    # 全量复购索引
    rbf = full_txn_all.groupby(["customer_id", "article_id"]).size().reset_index(name="cnt")
    user_rebuy_full = defaultdict(list)
    for _, row in rbf[rbf["cnt"] >= 2].iterrows():
        user_rebuy_full[row["customer_id"]].append(row["article_id"])
    del rbf; gc.collect()

    # 全量用户类目
    ma = (
        full_txn_all[["customer_id", "article_id"]]
        .merge(art_info[["article_id", "cate_key"]], on="article_id", how="left")
    )
    uccf = ma.groupby(["customer_id", "cate_key"]).size().reset_index(name="cnt")
    uccf = uccf.sort_values(["customer_id", "cnt"], ascending=[True, False])
    user_cate_full = {}
    for cid, grp in uccf.groupby("customer_id"):
        user_cate_full[cid] = [
            k for k in grp["cate_key"].tolist()
            if k in cate_top
        ][:N_CATE]
    del ma, uccf, full_txn_all; gc.collect()

    print(f"  全量特征已加载")

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()
known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} "
      f"({len(sub_cold)/len(sub_cids)*100:.1f}%)")

pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]
all_preds = {}

with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        bnum = start // INFER_BATCH_SIZE + 1

        batch_cands = gen_cands(user_hist_full, item_sim_full, art_pop_full, batch_cids,
                                user_cate_full, cate_top, user_rebuy_full)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full,
                                neg_ratio=None)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)
        inf_df["score"] = final_model.predict(inf_df[FEAT_COLS_CLEAN].values)
        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {bnum}: {len(batch_cids):,} 用户完成")

# ============================================================
# 生成提交文件
# ============================================================
sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_expanded.csv", index=False)
print(f"\n提交文件: {OUTPUT_DIR}/submission_expanded.csv ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"结果汇总: 23维 + 扩大召回 + 负采样")
print(f"{'='*55}")
print(f"  Holdout MAP@12:            {score_holdout:.5f}")
print(f"  Train MAP@12:              {score_train_eval:.5f}")
print(f"  过拟合:                    {score_train_eval-score_holdout:.5f}")
print(f"  流行度 Baseline (holdout): {score_pop_ho:.5f}")
print(f"  模型提升:                  +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:              {best_iter}")
print(f"  候选/用户:                 {avg_new:.1f} (原{orig_avg:.1f})")
print(f"  负采样:                    1:{NEG_RATIO}")
print(f"  冷启动比例:                {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*55}")
print(f"\n完成!")
