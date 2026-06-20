"""
Step 7: SHAP 模型可解释性分析
=========================================
对 step5_item2vec.py 训练好的 LightGBM 模型进行可解释性分析:
  - Phase 1: 全局 SHAP 特征重要性
  - Phase 2: SHAP 依赖图 (Top-8 特征)
  - Phase 3: 典型用户推荐归因 (force plot)
  - Phase 4: 高/低分样本特征贡献对比

用法:
  python step7_shap.py                              # 使用默认参数
  python step7_shap.py --vector_size 64             # 匹配对应模型
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import argparse
import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import lightgbm as lgb
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from config import DATA_DIR, PROCESSED_DIR, LGB_PARAMS
from utils import timer

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = f"{OUTPUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# ============================================================
# Word2Vec 参数 (默认: vector_size=32)
# ============================================================
W2V_DEFAULTS = {
    'vector_size': 32,
    'window': 5,
    'min_count': 3,
    'sg': 1,
    'epochs': 20,
    'negative': 5,
}
parser = argparse.ArgumentParser(description='SHAP 可解释性分析')
for k, v in W2V_DEFAULTS.items():
    parser.add_argument(f'--{k}', type=type(v), default=v, help=f'{k} (default: {v})')
args = parser.parse_args()

A2V_COLS = [f'a2v_{i}' for i in range(args.vector_size)]
V2V_SIM_COL = ['v2v_sim']
EXP_TAG = f"vs{args.vector_size}_w{args.window}_sg{args.sg}_e{args.epochs}_n{args.negative}_mc{args.min_count}"

print("=" * 60)
print("Step 7: SHAP 模型可解释性分析")
print("=" * 60)

# ============================================================
# 特征列 (同 step5_item2vec.py)
# ============================================================
CUS_COLS = ['age', 'postal_le', 'R_days', 'n_unique_articles']
ART_COLS = [
    'popularity_score', 'price_log', 'sales_log',
    'product_group_name_le', 'product_type_name_le',
    'colour_group_name_le', 'index_name_le',
]
ART_COLS += [f'text_emb_{i}' for i in [0, 1, 6, 7, 15, 16, 18]]
INTER_COLS = ['buy_count', 'last_buy_days', 'first_buy_days']
CAND_COLS = ['cf_score', 'price_match']

FEAT_COLS = (CUS_COLS + ART_COLS + INTER_COLS + CAND_COLS +
             A2V_COLS + V2V_SIM_COL)

# 特征分组 (用于分块显示)
FEAT_GROUPS = {
    "客户特征": CUS_COLS,
    "商品特征": ART_COLS,
    "交互特征": INTER_COLS,
    "候选特征": CAND_COLS,
    "a2v嵌入": A2V_COLS,
    "v2v相似度": V2V_SIM_COL,
}

# ============================================================
# 数据读取 & LTR 构建 (复用 step5_item2vec 逻辑)
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

# 时间切分
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)
val_train_txn = val_txn[val_txn["t_dat"] < holdout_start]
val_holdout_txn = val_txn[val_txn["t_dat"] >= holdout_start]

val_train_labels = {
    cid: set(aids)
    for cid, aids in val_train_txn.groupby("customer_id")["article_id"].apply(list).items()
}
val_holdout_labels = {
    cid: set(aids)
    for cid, aids in val_holdout_txn.groupby("customer_id")["article_id"].apply(list).items()
}

# ============================================================
# 加载 Word2Vec 嵌入 (跟 step5_item2vec 一致)
# ============================================================
phase0_model_path = f"{OUTPUT_DIR}/word2vec_article_{EXP_TAG}.model"
if not os.path.exists(phase0_model_path):
    # 回退到默认路径
    phase0_model_path = f"{OUTPUT_DIR}/word2vec_article.model"

from gensim.models import Word2Vec as W2V
import gc

if os.path.exists(phase0_model_path):
    print(f"\n  加载 Word2Vec: {os.path.basename(phase0_model_path)}")
    w2v_model = W2V.load(phase0_model_path)
    art2v = {aid: w2v_model.wv[aid] for aid in art_feat["article_id"]
             if aid in w2v_model.wv}
    default_vec = np.zeros(args.vector_size, dtype=np.float32)

    # 追加 a2v 嵌入到 art_feat
    for i in range(args.vector_size):
        col = f"a2v_{i}"
        art_feat[col] = art_feat["article_id"].map(
            lambda x, i=i: art2v.get(x, default_vec)[i]
        ).astype(np.float32)
    print(f"  商品嵌入覆盖率: {len(art2v)}/{len(art_feat)} ({len(art2v)/len(art_feat)*100:.1f}%)")
else:
    print("\n  [警告] 未找到 Word2Vec 模型, a2v 嵌入将全为 0")
    default_vec = np.zeros(args.vector_size, dtype=np.float32)
    for i in range(args.vector_size):
        art_feat[f"a2v_{i}"] = np.float32(0)
    art2v = {}


# ============================================================
# 构建 LTR 数据 (简洁版)
# ============================================================
def build_ltr_data_simple(candidates, labels, cus_feat_df, art_feat_df,
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

    feat_art_cols = [c for c in (ART_COLS + ["article_id"] + A2V_COLS) if c in art_feat_df.columns]
    df = df.merge(cus_feat_df[CUS_COLS + ["customer_id"]], on="customer_id", how="left")
    df = df.merge(art_feat_df[feat_art_cols], on="article_id", how="left")
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

    # v2v_sim
    if art2v:
        user_emb_cache = {}
        for cid in candidates:
            hist = user_hist.get(cid, [])[:12]
            if hist and all(a in art2v for a in hist):
                vecs = np.array([art2v[a] for a in hist], dtype=np.float32)
                user_emb_cache[cid] = vecs.mean(axis=0)
            else:
                user_emb_cache[cid] = default_vec.copy()
        cids_sim = df["customer_id"].values
        aids_sim = df["article_id"].values
        user_vecs = np.array([user_emb_cache.get(c, default_vec) for c in cids_sim], dtype=np.float32)
        art_vecs = np.array([art2v.get(a, default_vec) for a in aids_sim], dtype=np.float32)
        dot = (user_vecs * art_vecs).sum(axis=1)
        u_norm = np.linalg.norm(user_vecs, axis=1)
        a_norm = np.linalg.norm(art_vecs, axis=1)
        df["v2v_sim"] = (dot / (u_norm * a_norm + 1e-8)).astype(np.float32)
    else:
        df["v2v_sim"] = np.float32(0)

    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = np.float32(0)
        elif df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)
    return df


# ============================================================
# 加载模型
# ============================================================
MODEL_PATH = f"{OUTPUT_DIR}/model_item2vec_{EXP_TAG}.txt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = f"{OUTPUT_DIR}/model_item2vec.txt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = f"{OUTPUT_DIR}/model_lowlr.txt"

print(f"\n加载模型: {os.path.basename(MODEL_PATH)}")
model = lgb.Booster(model_file=MODEL_PATH)
print(f"  特征数: {model.num_feature()}  |  树数: {model.num_trees()}")

# ============================================================
# 构建 holdout 集上的 LTR 数据 (用于 SHAP 分析)
# ============================================================
common_users = sorted(
    set(val_train_txn["customer_id"].unique()) &
    set(val_holdout_txn["customer_id"].unique())
)
holdout_cids = [c for c in common_users if c in candidates]

print(f"\n构建 LTR 数据 (holdout, 随机采样 2000 用户)...")
import random
random.seed(42)
sample_size = min(2000, len(holdout_cids))
sample_cids = random.sample(holdout_cids, sample_size)
sample_cands = {c: candidates[c] for c in sample_cids}

with timer("构建 LTR 数据"):
    ltr_df = build_ltr_data_simple(
        sample_cands, val_holdout_labels, cus_feat, art_feat,
        inter_feat, item_sim, user_hist
    )

X = ltr_df[FEAT_COLS].values.astype(np.float32)
y = ltr_df["label"].values
# 保存 customer_id 供 Phase 3 使用
customer_ids = ltr_df["customer_id"].values
print(f"  SHAP 数据: {len(X):,} 行 × {X.shape[1]} 维  |  正样本率: {y.mean()*100:.2f}%")

del cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist, candidates, ltr_df
gc.collect()

# ============================================================
# Phase 1: 全局 SHAP 特征重要性
# ============================================================
print(f"\n{'='*60}")
print("Phase 1: SHAP 全局特征重要性")
print(f"{'='*60}")

with timer("计算 SHAP 值 (TreeExplainer)"):
    explainer = shap.TreeExplainer(model)
    # 使用 2000 个样本作为背景 (降低计算量)
    background = X[:2000]
    # 对全部样本计算 SHAP
    shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    # 多分类时取正类
    shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    shap_arr = shap_values

# 全局特征重要性 (mean |SHAP|)
shap_importance = np.abs(shap_arr).mean(axis=0)
imp_idx = np.argsort(shap_importance)[::-1]

print(f"\nTop 20 特征 (SHAP importance):")
print(f"{'排名':>4}  {'特征名':<25}  {'|SHAP|':>10}")
print("-" * 45)
for rank, idx in enumerate(imp_idx[:20], 1):
    print(f"{rank:>4}  {FEAT_COLS[idx]:<25}  {shap_importance[idx]:>10.6f}")

# --- 1a. SHAP 条形图 (Top 20) ---
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20
top_idx = imp_idx[:top_n][::-1]
ax.barh(range(top_n), shap_importance[top_idx], color="#1f77b4")
ax.set_yticks(range(top_n))
ax.set_yticklabels([FEAT_COLS[i] for i in top_idx], fontsize=9)
ax.set_xlabel("mean |SHAP value|")
ax.set_title("SHAP Feature Importance (Top 20)")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/shap_importance_bar.png", dpi=150)
plt.close()
print(f"  已保存: {FIG_DIR}/shap_importance_bar.png")

# --- 1b. SHAP 特征分组贡献 ---
print(f"\n特征分组贡献:")
for group_name, cols in FEAT_GROUPS.items():
    if not cols:
        continue
    valid = [c for c in cols if c in FEAT_COLS]
    if not valid:
        continue
    idxs = [FEAT_COLS.index(c) for c in valid]
    group_imp = shap_importance[idxs].sum()
    print(f"  {group_name:<12}  ({len(valid):>2}维)  mean |SHAP| = {group_imp:.6f}")

# ============================================================
# Phase 2: SHAP 依赖图 (Top 8)
# ============================================================
print(f"\n{'='*60}")
print("Phase 2: SHAP 依赖图 (Top 8 特征)")
print(f"{'='*60}")

top8_idx = imp_idx[:8]
with timer("绘制依赖图"):
    for rank, idx in enumerate(top8_idx, 1):
        feat_name = FEAT_COLS[idx]
        fig, ax = plt.subplots(figsize=(8, 5))

        # SHAP 依赖图 + 自动着色交互特征
        shap.dependence_plot(
            idx, shap_arr, X,
            feature_names=FEAT_COLS,
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP Dependence: {feat_name} (Top {rank})")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("SHAP value")
        plt.tight_layout()
        fig.savefig(f"{FIG_DIR}/shap_dependence_{rank:02d}_{feat_name}.png", dpi=150)
        plt.close()

print(f"  已保存 {len(top8_idx)} 张依赖图到 {FIG_DIR}/")

# ============================================================
# Phase 3: 典型用户推荐归因 (force plot)
# ============================================================
print(f"\n{'='*60}")
print("Phase 3: 典型用户推荐归因")
print(f"{'='*60}")

# 找 3 类用户: 高频 / 多样 / 低频
# 用 holdout 用户数据
val_txn_full = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
val_txn_full["t_dat"] = pd.to_datetime(val_txn_full["t_dat"])
user_freq = val_txn_full.groupby("customer_id").size().to_dict()
user_uniq = val_txn_full.groupby("customer_id")["article_id"].nunique().to_dict()

example_cids = []
# 高频复购: 购买次数最多
high_freq = sorted(user_freq.items(), key=lambda x: -x[1])
for cid, _ in high_freq:
    if cid in sample_cids:
        example_cids.append(("高频复购用户", cid))
        break
# 多样性探索: 购买商品种类最多
high_div = sorted(user_uniq.items(), key=lambda x: -x[1])
for cid, _ in high_div:
    if cid in sample_cids and cid not in [c[1] for c in example_cids]:
        example_cids.append(("多样性用户", cid))
        break
# 低频用户: 购买次数最少
low_freq = sorted(user_freq.items(), key=lambda x: x[1])
for cid, _ in low_freq:
    if cid in sample_cids and cid not in [c[1] for c in example_cids]:
        example_cids.append(("低频用户", cid))
        break

del val_txn_full
gc.collect()

# 为每个示例用户生成 force plot
for label, cid in example_cids:
    user_mask = customer_ids == cid
    if user_mask.sum() == 0:
        continue
    user_X = X[user_mask]
    user_shap = shap_arr[user_mask]

    # 取 Top-12 推荐商品 (取分数最高的 12 个)
    user_scores = model.predict(user_X)
    top_k = min(12, len(user_scores))
    top_indices = np.argsort(user_scores)[::-1][:top_k]
    # 取第 1 个推荐商品做详细归因
    idx0 = top_indices[0]

    fig = plt.figure(figsize=(12, 2.5))
    shap.force_plot(
        explainer.expected_value,
        user_shap[idx0],
        user_X[idx0],
        feature_names=FEAT_COLS,
        matplotlib=True,
        show=False,
    )
    plt.title(f"{label} (customer_id={cid}) — Top-1 推荐归因")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/shap_force_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top-5 整体特征贡献
    top5_shap = user_shap[top_indices[:5]]
    top5_mean = np.abs(top5_shap).mean(axis=0)
    top5_feat_idx = np.argsort(top5_mean)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(top5_feat_idx)),
        top5_mean[top5_feat_idx][::-1],
        color="#2ca02c",
    )
    ax.set_yticks(range(len(top5_feat_idx)))
    ax.set_yticklabels([FEAT_COLS[i] for i in top5_feat_idx[::-1]], fontsize=9)
    ax.set_xlabel("mean |SHAP| (Top-5 推荐)")
    ax.set_title(f"{label} — Top-5 推荐平均特征贡献")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/shap_user_top5_{label}.png", dpi=150)
    plt.close()

print(f"  已保存 {len(example_cids)} 组用户归因图到 {FIG_DIR}/")

# ============================================================
# Phase 4: 高/低分样本特征贡献对比
# ============================================================
print(f"\n{'='*60}")
print("Phase 4: 高分 vs 低分样本特征贡献对比")
print(f"{'='*60}")

with timer("计算全量预测"):
    all_scores = model.predict(X)

score_high = np.percentile(all_scores, 90)
score_low = np.percentile(all_scores, 10)

high_mask = all_scores >= score_high
low_mask = all_scores <= score_low

print(f"  高分样本 (>{score_high:.4f}): {high_mask.sum():,}")
print(f"  低分样本 (<{score_low:.4f}): {low_mask.sum():,}")

high_shap_mean = np.abs(shap_arr[high_mask]).mean(axis=0)
low_shap_mean = np.abs(shap_arr[low_mask]).mean(axis=0)

# 对比 Top-15 特征
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, title, imp_data in [
    (axes[0], "High-score samples (Top 90%)", high_shap_mean),
    (axes[1], "Low-score samples (Bottom 10%)", low_shap_mean),
]:
    top_idx_h = np.argsort(imp_data)[::-1][:15][::-1]
    ax.barh(range(15), imp_data[top_idx_h], color="#d62728" if "High" in title else "#1f77b4")
    ax.set_yticks(range(15))
    ax.set_yticklabels([FEAT_COLS[i] for i in top_idx_h], fontsize=8)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)

plt.suptitle("High-score vs Low-score 特征贡献对比", fontsize=14)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/shap_high_vs_low.png", dpi=150)
plt.close()
print(f"  已保存: {FIG_DIR}/shap_high_vs_low.png")

# 对比差值 Top-10
diff = high_shap_mean - low_shap_mean
diff_idx = np.argsort(np.abs(diff))[::-1][:10]
print(f"\n高分 vs 低分 差异最大的特征:")
print(f"{'特征名':<25}  {'高分|SHAP|':>10}  {'低分|SHAP|':>10}  {'差值':>10}")
print("-" * 60)
for idx in diff_idx:
    print(f"{FEAT_COLS[idx]:<25}  {high_shap_mean[idx]:>10.6f}  {low_shap_mean[idx]:>10.6f}  {diff[idx]:>+10.6f}")

# ============================================================
# 清理 & 总结
# ============================================================
print(f"\n{'='*60}")
print("SHAP 可解释性分析完成!")
print(f"{'='*60}")
print(f"\n输出目录: {FIG_DIR}/")
print(f"生成文件:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("shap_"):
        size = os.path.getsize(f"{FIG_DIR}/{f}")
        print(f"  {f:<50} {size/1024:.1f} KB")
print(f"\n总计: {sum(1 for f in os.listdir(FIG_DIR) if f.startswith('shap_'))} 张图")
