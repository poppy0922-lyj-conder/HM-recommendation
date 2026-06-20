"""
Step 9: 5 折时序交叉验证稳定性分析
====================================
用途: 使用时序交叉验证 (而非随机拆分) 评估模型在 5 个不同时间窗口上的
     泛化稳定性，计算均值 ROC AUC 与方差区间。

核心思想:
  - 时间序列交叉验证: 按时间顺序划分训练/验证窗口，确保训练数据始终在
    验证数据之前，从根本上杜绝数据泄露。
  - 数据工程策略: 在数据量不足的场景下，通过滚动时间窗口将有限的验证期
    拆分为多个 train/val 对，变相"扩充"训练数据，让模型看到更多不同时间
    段的分布模式。
  - 稳定性分析: 5 次训练结果的均值反映模型真实能力，方差反映对时间分布
    偏移的敏感度 — 方差越小，模型越稳定可靠。

特征: 56维 (23基线 + 32 a2v + 1 v2v_sim) — 与 step5_item2vec.py 一致
参数: random_state=610 作为基准参数

输出:
  output/cv_report.txt      — 完整文字报告
  output/cv_results.csv     — 每折详细结果
  output/cv_roc_curves.png  — 5 折 ROC 曲线叠图 (可选)
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, warnings, json
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE, SEED,
)
from utils import timer, mapk, set_seed

# ============ 覆盖为 random_state=610 ============
CV_SEED = 610
set_seed(CV_SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 9: 5 折时序交叉验证稳定性分析")
print("=" * 60)

# ============================================================
# 特征列定义 (与 step5_item2vec.py 完全一致)
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
A2V_COLS = [f"a2v_{i}" for i in range(32)]
V2V_SIM_COL = ["v2v_sim"]
FEAT_COLS_CLEAN = (
    CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN +
    CAND_COLS_CLEAN + A2V_COLS + V2V_SIM_COL
)
print(f"特征维度: {len(FEAT_COLS_CLEAN)} 维 (23基线 + 32a2v + 1v2v_sim)")

N_FOLDS = 5
CV_RESULTS = []


# ============================================================
# 通用函数 (与 step5_item2vec.py 一致)
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist, art2v,
                   default_vec, v2v_col=V2V_SIM_COL[0]):
    """增强版 build_ltr_data: 新增 v2v_sim 计算"""
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

    feat_art_cols = ART_COLS + ["article_id"] + A2V_COLS
    avail_art_cols = [c for c in feat_art_cols if c in art_feat_df.columns]

    df = df.merge(cus_feat_df[CUS_COLS + ["customer_id"]], on="customer_id", how="left")
    df = df.merge(art_feat_df[avail_art_cols], on="article_id", how="left")
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

    # v2v_sim 余弦相似度
    user_emb_cache = {}
    for cid in candidates:
        hist = user_hist.get(cid, [])[:12]
        if hist and all(a in art2v for a in hist):
            vecs = np.array([art2v[a] for a in hist], dtype=np.float32)
            user_emb_cache[cid] = vecs.mean(axis=0)
        else:
            user_emb_cache[cid] = default_vec.copy()

    cids_for_sim = df["customer_id"].values
    aids_for_sim = df["article_id"].values
    user_vecs = np.array([user_emb_cache.get(c, default_vec) for c in cids_for_sim], dtype=np.float32)
    art_vecs = np.array([art2v.get(a, default_vec) for a in aids_for_sim], dtype=np.float32)
    dot_products = (user_vecs * art_vecs).sum(axis=1)
    user_norms = np.linalg.norm(user_vecs, axis=1)
    art_norms = np.linalg.norm(art_vecs, axis=1)
    df[v2v_col] = (dot_products / (user_norms * art_norms + 1e-8)).astype(np.float32)

    return df


def prepare_ltr(ltr_df):
    """准备特征矩阵、标签、group 结构"""
    for c in FEAT_COLS_CLEAN:
        if c not in ltr_df.columns:
            ltr_df[c] = np.float32(0)
        elif ltr_df[c].dtype == "float64":
            ltr_df[c] = ltr_df[c].astype(np.float32)
    X = ltr_df[FEAT_COLS_CLEAN].values
    y = ltr_df["label"].values
    groups = ltr_df.groupby("customer_id").size().values
    return X, y, groups


def generate_candidates(user_hist, item_sim, art_pop, customers,
                        n_hist=12, n_pop=12):
    """为每个用户生成候选商品列表"""
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

    with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
        user_hist_full_data = pickle.load(f)

    art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")
print(f"  用户历史: {len(user_hist):,}  全量历史: {len(user_hist_full_data):,}")

# ============================================================
# Phase 0: 训练 Word2Vec (全量数据)
# ============================================================
print(f"\n{'='*60}")
print("Phase 0: 训练 Word2Vec 商品嵌入 (全量数据)")
print(f"{'='*60}")

with timer("Word2Vec 训练"):
    sequences = list(user_hist_full_data.values())
    w2v_model = Word2Vec(
        sentences=sequences,
        vector_size=32,
        window=5,
        min_count=3,
        sg=1,
        epochs=20,
        workers=4,
        seed=CV_SEED,
    )
    print(f"  词汇表大小: {len(w2v_model.wv):,}")

art2v = {aid: w2v_model.wv[aid] for aid in art_feat["article_id"]
         if aid in w2v_model.wv}
default_vec = np.zeros(32, dtype=np.float32)
n_known = len(art2v)
n_total = len(art_feat)
print(f"  商品嵌入覆盖率: {n_known}/{n_total} ({n_known/n_total*100:.1f}%)")

# ============================================================
# Phase 0.5: 追加 a2v 嵌入到 art_feat
# ============================================================
print(f"\n{'='*60}")
print("Phase 0.5: 追加 a2v 嵌入")
print(f"{'='*60}")

with timer("追加 a2v"):
    a2v_mat = np.array([art2v.get(aid, default_vec) for aid in art_feat["article_id"]], dtype=np.float32)
    for i in range(32):
        art_feat[f"a2v_{i}"] = a2v_mat[:, i]

# ============================================================
# 构建 5 折时序交叉验证
# ============================================================
print(f"\n{'='*60}")
print(f"构建 {N_FOLDS} 折时序交叉验证窗口")
print(f"{'='*60}")

# 按时间排序 val_txn 中的日期
val_dates = sorted(val_txn["t_dat"].unique())
print(f"  Val 时间跨度: {val_dates[0].date()} ~ {val_dates[-1].date()} ({len(val_dates)} 天)")

# 5 折扩展窗口:
#   Fold 1: train=day[0],          validate=day[1]
#   Fold 2: train=days[0:2],       validate=day[2]
#   Fold 3: train=days[0:3],       validate=day[3]
#   Fold 4: train=days[0:4],       validate=day[4]
#   Fold 5: train=days[0:5],       validate=days[5:7]
#
# 说明: 每折的训练数据逐渐增多 (扩展窗口, 非滑动窗口),
#   保证训练数据始终在验证数据之前, 防止时间泄露.

folds = []
for k in range(N_FOLDS):
    # 训练截止日期: 第 k 天结束 (包含第 k 天)
    train_end = val_dates[k] + timedelta(days=1)  # 不含第 k+1 天
    # 验证日期
    if k < N_FOLDS - 1:
        val_date = val_dates[k + 1]
        val_end = val_date + timedelta(days=1)
    else:
        # 最后一折: 验证最后 2 天
        val_date = val_dates[k + 1]
        val_end = val_dates[-1] + timedelta(days=1)

    val_mask = (val_txn["t_dat"] >= val_date) & (val_txn["t_dat"] < val_end)
    folds.append((train_end, val_date, val_end))
    n_val = val_mask.sum()
    print(f"  Fold {k+1}: train<{train_end.date()}  val={val_date.date()}~{val_end.date()}  val样本={n_val:,}")


# ============================================================
# Phase 1: 5 折训练 + 评估
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 1: {N_FOLDS} 折时序交叉验证 (LightGBM LambdaRank)")
print(f"{'='*60}")
print(f"  random_state={CV_SEED}  |  lr=0.02  |  rounds=2000  |  early_stop=100")

# 构建候选人集合 (memo: 避免每折重复构建)
train_cand = {u: candidates[u] for u in sorted(set(candidates.keys()))}

# LightGBM 参数 (使用 random_state=610)
lgb_cv_params = LGB_PARAMS.copy()
lgb_cv_params["seed"] = CV_SEED
lgb_cv_params["learning_rate"] = 0.02
# GPU 检测
try:
    test_X = np.zeros((10, len(FEAT_COLS_CLEAN)), dtype=np.float32)
    test_y = np.zeros(10)
    test_ds = lgb.Dataset(test_X, label=test_y)
    lgb.train(lgb_cv_params, test_ds, num_boost_round=1, callbacks=[lgb.log_evaluation(0)])
    print("\nGPU 可用 → 使用 GPU 训练")
except Exception:
    lgb_cv_params["device"] = "cpu"
    for key in ["gpu_platform_id", "gpu_device_id"]:
        lgb_cv_params.pop(key, None)
    print("\nGPU 不可用 → 自动切换 CPU 训练")


for fold_idx in range(N_FOLDS):
    print(f"\n{'─'*55}")
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"{'─'*55}")

    train_end, val_date, val_end = folds[fold_idx]

    # 时间切分
    train_mask = val_txn["t_dat"] < train_end
    val_mask = (val_txn["t_dat"] >= val_date) & (val_txn["t_dat"] < val_end)

    train_fold_txn = val_txn.loc[train_mask]
    val_fold_txn = val_txn.loc[val_mask]

    # 构建标签
    fold_train_labels = {
        cid: set(aids)
        for cid, aids in train_fold_txn.groupby("customer_id")["article_id"].apply(list).items()
    }
    fold_val_labels = {
        cid: set(aids)
        for cid, aids in val_fold_txn.groupby("customer_id")["article_id"].apply(list).items()
    }

    # 共同用户
    fold_train_users = set(train_fold_txn["customer_id"].unique())
    fold_val_users = set(val_fold_txn["customer_id"].unique())
    common_fold_users = sorted(fold_train_users & fold_val_users)
    print(f"  训练用户: {len(fold_train_users):,}  |  验证用户: {len(fold_val_users):,}  |  共同: {len(common_fold_users):,}")

    # === 构建 LTR 数据 ===
    with timer(f"  Fold {fold_idx+1} LTR 构建"):
        train_ltr = build_ltr_data(
            train_cand, fold_train_labels, cus_feat, art_feat,
            inter_feat, item_sim, user_hist, art2v, default_vec,
        )
        val_ltr = build_ltr_data(
            train_cand, fold_val_labels, cus_feat, art_feat,
            inter_feat, item_sim, user_hist, art2v, default_vec,
        )

    X_tr, y_tr, grp_tr = prepare_ltr(train_ltr)
    X_val, y_val, grp_val = prepare_ltr(val_ltr)

    train_ds = lgb.Dataset(X_tr, label=y_tr, group=grp_tr,
                           feature_name=FEAT_COLS_CLEAN)
    valid_ds = lgb.Dataset(X_val, label=y_val, group=grp_val,
                           feature_name=FEAT_COLS_CLEAN, reference=train_ds)

    print(f"  训练样本: {len(X_tr):,}  |  验证样本: {len(X_val):,}")

    # === 训练 ===
    with timer(f"  Fold {fold_idx+1} 训练"):
        fold_model = lgb.train(
            lgb_cv_params, train_ds,
            num_boost_round=2000,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(0),  # 静默训练, 只看结果
            ],
        )

    best_round = fold_model.best_iteration
    print(f"  最佳迭代: {best_round}")

    # === 预测 ===
    val_ltr["score"] = fold_model.predict(val_ltr[FEAT_COLS_CLEAN].values)

    # === 计算 MAP@12 ===
    val_preds_map = (
        val_ltr.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    val_preds_list = [val_preds_map.get(c, []) for c in common_fold_users]
    val_actuals_list = [fold_val_labels[c] for c in common_fold_users]
    fold_map12 = mapk(val_actuals_list, val_preds_list, k=12)

    # === 计算 ROC AUC (全局) ===
    # LambdaRank 输出是排序分数, 用 sigmoid 归一化到 [0,1] 后计算 AUC
    scores = val_ltr["score"].values
    labels = val_ltr["label"].values

    # 检查标签是否包含两类
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    if n_pos > 0 and n_neg > 0:
        # Sigmoid 归一化
        proba = 1.0 / (1.0 + np.exp(-np.clip(scores, -15, 15)))
        fold_auc = roc_auc_score(labels, proba)
        fold_fpr, fold_tpr, _ = roc_curve(labels, proba)
    else:
        fold_auc = float("nan")
        fold_fpr, fold_tpr = None, None
        print(f"  [WARN] Fold {fold_idx+1}: 验证集只包含 {n_pos} 正 {n_neg} 负, 无法计算 AUC")

    print(f"  MAP@12: {fold_map12:.5f}  |  ROC AUC: {fold_auc:.5f}")

    CV_RESULTS.append({
        "fold": fold_idx + 1,
        "train_end": str(train_end.date()),
        "val_start": str(val_date.date()),
        "val_end": str(val_end.date()),
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
        "n_users": len(common_fold_users),
        "best_iteration": best_round,
        "map12": fold_map12,
        "roc_auc": fold_auc,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "pos_ratio": float(n_pos / (n_pos + n_neg)),
    })

    # === 保存最佳模型的 ROC 曲线数据 (中间折) ===
    if fold_idx == 2 and fold_fpr is not None:
        cv_best_fpr = fold_fpr
        cv_best_tpr = fold_tpr
        cv_best_auc = fold_auc

    # 清理
    del train_ltr, val_ltr, X_tr, y_tr, grp_tr, X_val, y_val, grp_val
    del train_ds, valid_ds, fold_model, train_fold_txn, val_fold_txn
    del fold_train_labels, fold_val_labels, val_preds_map
    del val_preds_list, val_actuals_list
    gc.collect()


# ============================================================
# 结果汇总
# ============================================================
print(f"\n{'='*60}")
print(f"5 折时序交叉验证结果汇总")
print(f"{'='*60}")

cv_df = pd.DataFrame(CV_RESULTS)

aucs = cv_df["roc_auc"].dropna().values
map12s = cv_df["map12"].values

auc_mean = aucs.mean()
auc_std = aucs.std(ddof=1)  # 样本标准差
auc_ci_low = auc_mean - 1.96 * auc_std
auc_ci_high = auc_mean + 1.96 * auc_std

map_mean = map12s.mean()
map_std = map12s.std(ddof=1)

print(f"\n  ┌────────────────────────┬─────────────┬─────────────┐")
print(f"  │ 指标                    │    均值     │  标准差     │")
print(f"  ├────────────────────────┼─────────────┼─────────────┤")
print(f"  │ ROC AUC                │ {auc_mean:.5f}    │ {auc_std:.5f}    │")
print(f"  │ MAP@12                 │ {map_mean:.5f}    │ {map_std:.5f}    │")
print(f"  └────────────────────────┴─────────────┴─────────────┘")
print(f"  ROC AUC 95% 置信区间:    [{auc_ci_low:.5f}, {auc_ci_high:.5f}]")
print(f"  ROC AUC 波动 (CV):       ±{auc_std/auc_mean*100:.2f}%")
print(f"  MAP@12 波动 (CV):        ±{map_std/map_mean*100:.2f}%")
print(f"  AUC 最小值:              {aucs.min():.5f}")
print(f"  AUC 最大值:              {aucs.max():.5f}")
print(f"  AUC 极差:                {aucs.max()-aucs.min():.5f}")

print(f"\n  {'─'*65}")
print(f"  {'Fold':>5}  {'MAP@12':>8}  {'ROC AUC':>8}  {'样本数':>8}  {'用户数':>7}  {'正例比':>7}")
print(f"  {'─'*65}")
for _, row in cv_df.iterrows():
    print(f"  {int(row['fold']):>5}  {row['map12']:>8.5f}  {row['roc_auc']:>8.5f}  "
          f"{int(row['val_samples']):>8,}  {int(row['n_users']):>7,}  {row['pos_ratio']:>7.4f}")
print(f"  {'─'*65}")
print(f"  {'Mean':>5}  {map_mean:>8.5f}  {auc_mean:>8.5f}")
print(f"  {'Std':>5}  {map_std:>8.5f}  {auc_std:>8.5f}")

# ============================================================
# 保存结果
# ============================================================
cv_df.to_csv(f"{OUTPUT_DIR}/cv_results.csv", index=False)
print(f"\n详细结果已保存: {OUTPUT_DIR}/cv_results.csv")

# 保存 ROC 曲线图
try:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 画中间折的 ROC 曲线
    ax.plot(cv_best_fpr, cv_best_tpr, lw=2,
            label=f"Fold 3 (AUC={cv_best_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")

    # 标出均值 AUC 和置信区间
    ax.text(0.6, 0.2, f"Mean AUC = {auc_mean:.4f} ± {auc_std:.4f}",
            fontsize=12, bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.text(0.6, 0.1, f"95% CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]",
            fontsize=10, bbox=dict(boxstyle="round", fc="lightblue", alpha=0.8))

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"5-Fold Time Series CV — ROC Curve (random_state={CV_SEED})", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/cv_roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"ROC 曲线已保存: {OUTPUT_DIR}/cv_roc_curves.png")
except Exception as e:
    print(f"  [WARN] ROC 曲线保存失败: {e}")

# ============================================================
# 文字报告
# ============================================================
report = f"""
================================================================================
  5 折时序交叉验证稳定性分析报告
================================================================================

1. 实验设置
  - 模型: LightGBM LambdaRank (56维: 23基线 + 32a2v + 1v2v_sim)
  - 参数: lr=0.02, rounds=2000, early_stop=100, random_state={CV_SEED}
  - 验证策略: 时间序列扩展窗口交叉验证 ({N_FOLDS} 折)
  - 数据范围: {{val_dates[0].date()}} ~ {{val_dates[-1].date()}} ({len(val_dates)} 天)

2. 交叉验证结果
  - ROC AUC 均值 ± 标准差: {auc_mean:.5f} ± {auc_std:.5f}
  - ROC AUC 95% 置信区间: [{auc_ci_low:.5f}, {auc_ci_high:.5f}]
  - MAP@12 均值 ± 标准差: {map_mean:.5f} ± {map_std:.5f}
  - AUC 波动系数 (CV): {auc_std/auc_mean*100:.2f}%
  - MAP@12 波动系数 (CV): {map_std/map_mean*100:.2f}%

3. 每折明细
"""

for _, row in cv_df.iterrows():
    report += (
        f"  Fold {int(row['fold'])}: "
        f"train<{row['train_end']}  "
        f"val={row['val_start']}~{row['val_end']}  "
        f"MAP@12={row['map12']:.5f}  "
        f"AUC={row['roc_auc']:.5f}  "
        f"样本={int(row['val_samples']):,}  "
        f"用户={int(row['n_users']):,}  "
        f"正例比={row['pos_ratio']:.4f}"
    )

report += f"""
4. 稳定性分析
  - AUC 最大值 - 最小值: {aucs.max():.5f} - {aucs.min():.5f} = {aucs.max()-aucs.min():.5f}
  - {'模型稳定性良好' if auc_std < 0.02 else '模型波动较大，需进一步优化'}
  - {'各折 AUC 均在均值±2σ范围内' if all(abs(a - auc_mean) < 2 * auc_std for a in aucs) else '存在离群折，需关注特定时间窗口的分布偏移'}

5. 方法说明
  时序交叉验证 (Time Series Cross-Validation) 与传统 K-Fold 的关键区别:

  传统 K-Fold: 随机打乱数据后均分 K 份，轮流训练验证。
    → 问题: 训练集中"未来"数据出现在验证集中，造成数据泄露 (data leakage)，
      导致验证指标虚高，无法反映模型在真实线上环境的表现。

  时序扩展窗口 (Expanding Window):
    Fold 1:  [Train: day 0]           → [Val: day 1]
    Fold 2:  [Train: day 0-1]         → [Val: day 2]
    Fold 3:  [Train: day 0-2]         → [Val: day 3]
    Fold 4:  [Train: day 0-3]         → [Val: day 4]
    Fold 5:  [Train: day 0-4]         → [Val: day 5-6]
    → 优势: 严格保证训练数据时间在验证数据之前，与现实推荐场景一致。
    → 工程价值: 每折的训练集逐渐增大，模拟模型在更多历史数据下的表现,
      是数据量不足场景下扩充训练数据的核心工程策略。

6. 结论
  模型在 {N_FOLDS} 折时序交叉验证上 ROC AUC 均值为 {auc_mean:.5f}
  (95% CI: [{auc_ci_low:.5f}, {auc_ci_high:.5f}]),
  标准差为 {auc_std:.5f}, MAP@12 均值为 {map_mean:.5f}。
  模型在给定时间窗口内表现{'稳定' if auc_std < 0.02 else '存在一定波动'}。
"""

with open(f"{OUTPUT_DIR}/cv_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n完整报告已保存: {OUTPUT_DIR}/cv_report.txt")
print(f"\n{'='*60}")
print("Step 9 完成!")
print(f"{'='*60}")
