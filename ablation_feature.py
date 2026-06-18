"""
消融实验 (Ablation Study) — 特征消融
====================================
独立运行，自动读取 F:/H&M_data 原始数据。
测试 3 个新特征对 MAP@12 的影响：
  - price_ratio    商品价格 / 用户均价
  - recency_decay  平滑时间衰减 exp(-last_buy_days/30)
  - category_match 品类是否匹配用户最常买品类

原则: 一次只加一个变量，对比 Base 模型 vs Base+新特征 的 MAP@12 变化

使用方法:
  cd F:/HM-ablation-study
  python ablation_feature.py
"""

import sys, os

# 引用 F:/HM-recommendation 项目中的 config / utils
PROJECT_DIR = "F:/HM-recommendation"
sys.path.insert(0, PROJECT_DIR)

import polars as pl
import pandas as pd
import numpy as np
import gc, time, warnings
from datetime import timedelta
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from config import DATA_DIR, SEED
from utils import mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")
print(f"LightGBM {lgb.__version__}  |  Polars {pl.__version__}  |  SEED={SEED}")

# ============================================================
# 0. 配置
# ============================================================
VAL_DAYS = 7

# 输出目录 (当前目录)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_at": [12],
    "learning_rate": 0.05,
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
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
}

# ============================================================
# 1. 数据加载 + 时间顺序切分 (防穿越)
# ============================================================
print("\n" + "=" * 60)
print("第1部分: 时间顺序本地验证 (防穿越)")
print("=" * 60)

t0 = time.time()
txn_pl = pl.read_csv(
    f"{DATA_DIR}/transactions_train.csv",
    dtypes={"customer_id": pl.Utf8, "article_id": pl.Utf8, "price": pl.Float32},
)
txn_pl = txn_pl.with_columns(pl.col("t_dat").str.to_date("%Y-%m-%d"))

art_pl = pl.read_csv(f"{DATA_DIR}/articles.csv", dtypes={"article_id": pl.Utf8})
cus_pl = pl.read_csv(f"{DATA_DIR}/customers.csv", dtypes={"customer_id": pl.Utf8})

txn = txn_pl.to_pandas()
txn["t_dat"] = pd.to_datetime(txn["t_dat"])
del txn_pl; gc.collect()

art = art_pl.to_pandas()
del art_pl; gc.collect()

cus = cus_pl.to_pandas()
del cus_pl; gc.collect()

txn["price"] = txn["price"].astype(np.float32)
t1 = time.time()
print(f"[读取数据] {t1-t0:.1f}s  |  "
      f"交易: {len(txn):,} 行  |  日期: {txn['t_dat'].min().date()} ~ {txn['t_dat'].max().date()}")

# 时间顺序切分
max_date = txn["t_dat"].max()
val_start = max_date - timedelta(days=VAL_DAYS - 1)

train_txn = txn.loc[txn["t_dat"] < val_start].copy()
val_txn = txn.loc[txn["t_dat"] >= val_start].copy()

assert train_txn["t_dat"].max() < val_txn["t_dat"].min(), "❌ 数据穿越!"
assert len(set(train_txn["customer_id"]) & set(val_txn["customer_id"])) > 0, "❌ 无共同用户"

print(f"  Train: {len(train_txn):,} 行  ({train_txn['t_dat'].min().date()} ~ {train_txn['t_dat'].max().date()})")
print(f"  Val:   {len(val_txn):,} 行  ({val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()})")

val_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_cids = list(val_gt.keys())
print(f"  验证集用户: {len(val_cids):,}")

# ============================================================
# 2. 特征工程 (仅用 train_txn)
# ============================================================
print("\n" + "=" * 60)
print("第2部分: 特征工程 (仅用训练集)")
print("=" * 60)


def build_customer_features(txn_df, cus_df):
    ref = txn_df["t_dat"].max() + timedelta(days=1)
    df = txn_df[["customer_id", "t_dat", "article_id", "price"]].copy()
    df["R_days"] = (ref - df["t_dat"]).dt.days
    rfm = df.groupby("customer_id", sort=False).agg(
        R_days=("R_days", "min"),
        F_count=("article_id", "count"),
        M_spend=("price", "sum"),
        avg_price_user=("price", "mean"),
        n_unique_articles=("article_id", "nunique"),
    ).reset_index()

    cus_feat = cus_df[["customer_id", "age", "club_member_status", "postal_code"]].copy()
    cus_feat["age"] = cus_feat["age"].fillna(cus_feat["age"].median())
    cus_feat["club_member_status"] = cus_feat["club_member_status"].fillna("UNKNOWN")
    cus_feat = cus_feat.merge(rfm, on="customer_id", how="left")
    for c in ["R_days", "F_count", "M_spend", "avg_price_user", "n_unique_articles"]:
        cus_feat[c] = cus_feat[c].fillna(0)
    cus_feat["club_member_status_le"] = cus_feat["club_member_status"].astype("category").cat.codes
    cus_feat["postal_le"] = cus_feat["postal_code"].astype("category").cat.codes
    return cus_feat


def build_article_features(txn_df, art_df):
    stats = txn_df.groupby("article_id", sort=False).agg(
        avg_price=("price", "mean"),
        sales_count=("article_id", "count"),
        n_buyers=("customer_id", "nunique"),
    ).reset_index()

    max_d = txn_df["t_dat"].max()
    _td = txn_df[["article_id", "t_dat"]].copy()
    _td["days"] = (max_d - _td["t_dat"]).dt.days
    _td["w"] = np.exp(-_td["days"] / 14)
    pop = _td.groupby("article_id", sort=False)["w"].sum().reset_index()
    pop.rename(columns={"w": "popularity_score"}, inplace=True)
    del _td; gc.collect()

    df = art_df[["article_id"]].copy()
    df = df.merge(stats, on="article_id", how="left")
    df = df.merge(pop, on="article_id", how="left")
    for c in ["avg_price", "sales_count", "n_buyers", "popularity_score"]:
        df[c] = df[c].fillna(0)
    df["price_log"] = np.log1p(df["avg_price"])
    df["sales_log"] = np.log1p(df["sales_count"])

    cat_cols = [
        "product_group_name", "product_type_name",
        "graphical_appearance_name", "colour_group_name",
        "index_name", "section_name", "garment_group_name"
    ]
    for col in cat_cols:
        if col in art_df.columns:
            df[col] = art_df[col].values
            df[col + "_le"] = df[col].astype("category").cat.codes
            df.drop(columns=[col], inplace=True)

    desc = art_df["detail_desc"].fillna(art_df.get("product_type_name", "")).fillna("")
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    svd = TruncatedSVD(n_components=20, random_state=SEED)
    emb = svd.fit_transform(tfidf.fit_transform(desc.str.lower().fillna("")))
    for i in range(20):
        df[f"text_emb_{i}"] = emb[:, i]
    del desc, emb; gc.collect()
    return df


def build_interaction_features(txn_df):
    ref = txn_df["t_dat"].max() + timedelta(days=1)
    df = txn_df[["customer_id", "article_id", "t_dat"]].copy()
    df["days_to_ref"] = (ref - df["t_dat"]).dt.days
    return (
        df.groupby(["customer_id", "article_id"], sort=False)
        .agg(
            buy_count=("article_id", "count"),
            last_buy_days=("days_to_ref", "min"),
            first_buy_days=("days_to_ref", "max"),
        )
        .reset_index()
    )


def build_item_cf(train_df, top_k=30, min_cnt=3):
    df = train_df[["customer_id", "article_id", "t_dat"]].copy()
    df["uw"] = (
        df["customer_id"] + "_" +
        df["t_dat"].dt.year.astype(str) + "_" +
        df["t_dat"].dt.isocalendar().week.astype(str)
    )
    vc = df["article_id"].value_counts()
    keep = set(vc[vc >= min_cnt].index)
    df = df.loc[df["article_id"].isin(keep)]
    cooc = defaultdict(lambda: defaultdict(int))
    for uw, grp in df.groupby("uw", sort=False):
        items = sorted(set(grp["article_id"].tolist()))
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, min(len(items), i + 5)):
                cooc[items[i]][items[j]] += 1
                cooc[items[j]][items[i]] += 1
    del df; gc.collect()
    return {a: sorted(r.items(), key=lambda x: -x[1])[:top_k] for a, r in cooc.items()}


def build_user_history(train_df, max_items=30):
    df = train_df[["customer_id", "article_id", "t_dat"]].sort_values("t_dat", ascending=False)
    df = df.drop_duplicates(subset=["customer_id", "article_id"], keep="first")
    df = df.groupby("customer_id").head(max_items)
    return df.groupby("customer_id")["article_id"].apply(list).to_dict()


t_feat = time.time()
cus_feat = build_customer_features(train_txn, cus)
art_feat = build_article_features(train_txn, art)
inter_feat = build_interaction_features(train_txn)
item_sim = build_item_cf(train_txn)
user_hist = build_user_history(train_txn)
print(f"[特征工程] {time.time()-t_feat:.1f}s")
print(f"  cus_feat: {cus_feat.shape}  art_feat: {art_feat.shape}  inter_feat: {inter_feat.shape}")

# ============================================================
# 3. 特征列名定义 (46维全量)
# ============================================================
CUS_COLS = [
    "age", "club_member_status_le", "postal_le",
    "R_days", "F_count", "M_spend", "avg_price_user", "n_unique_articles"
]
ART_COLS = [
    "avg_price", "sales_count", "n_buyers", "popularity_score",
    "price_log", "sales_log",
    "product_group_name_le", "product_type_name_le",
    "graphical_appearance_name_le", "colour_group_name_le",
    "index_name_le", "section_name_le", "garment_group_name_le"
]
ART_COLS += [f"text_emb_{i}" for i in range(20)]
INTER_COLS = ["buy_count", "last_buy_days", "first_buy_days"]
CAND_COLS = ["cf_score", "price_match"]
FEAT_COLS = CUS_COLS + ART_COLS + INTER_COLS + CAND_COLS
print(f"  Base 特征: {len(FEAT_COLS)} 维")

# ============================================================
# 4. Baseline - 全量流行度
# ============================================================
print("\n" + "=" * 60)
print("第3部分: Baseline (流行度)")
print("=" * 60)

art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]

actuals = [val_gt[c] for c in val_cids]
preds_pop = [pop12] * len(val_cids)
score_pop = mapk(actuals, preds_pop, k=12)
print(f"  Local CV MAP@12: {score_pop:.5f}")

# ============================================================
# 5. 候选生成 + LTR 构建
# ============================================================
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


def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist, extra_fn=None):
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

    new_feat_names = []
    if extra_fn:
        extra_dict = extra_fn(df)
        for fname, fvalues in extra_dict.items():
            df[fname] = np.float32(fvalues)
            new_feat_names.append(fname)

    all_feats = FEAT_COLS + new_feat_names
    for c in all_feats:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)

    return df, all_feats


# ============================================================
# 6. 训练+评估函数
# ============================================================
def train_and_eval(ltr_df, feat_cols, params, val_cids, val_gt):
    groups = ltr_df.groupby("customer_id").size().values
    ds = lgb.Dataset(
        ltr_df[feat_cols].values, label=ltr_df["label"].values,
        group=groups, feature_name=feat_cols
    )

    try:
        model = lgb.train(params, ds, num_boost_round=500,
                          callbacks=[lgb.log_evaluation(0)])
    except Exception:
        cpu_params = {**params, "device": "cpu"}
        for k in ["gpu_platform_id", "gpu_device_id"]:
            cpu_params.pop(k, None)
        model = lgb.train(cpu_params, ds, num_boost_round=500,
                          callbacks=[lgb.log_evaluation(0)])

    ltr_df["score"] = model.predict(ltr_df[feat_cols].values)
    preds = (
        ltr_df.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals = [val_gt[c] for c in val_cids]
    preds_l = [preds.get(c, []) for c in val_cids]
    return mapk(actuals, preds_l, k=12), model


# ============================================================
# 7. 候选生成 + Base 模型训练
# ============================================================
print("\n" + "=" * 60)
print("第4部分: 候选生成 + Base 模型训练")
print("=" * 60)

val_labels = {cid: set(aids) for cid, aids in val_gt.items()}
candidates = generate_candidates(user_hist, item_sim, art_pop, val_cids)
tot = sum(len(v) for v in candidates.values())
print(f"  候选: {len(candidates):,} users, avg {tot/len(candidates):.1f} cands/user")

ltr_base, base_feats = build_ltr_data(
    candidates, val_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist,
    extra_fn=None
)
pos = ltr_base["label"].sum()
print(f"  LTR pairs: {len(ltr_base):,}  pos: {pos:,}  neg: {len(ltr_base)-pos:,}  "
      f"ratio: {pos/len(ltr_base):.3f}")

t_train = time.time()
score_base, model_base = train_and_eval(ltr_base, base_feats, LGB_PARAMS, val_cids, val_gt)
print(f"[Base 模型] MAP@12 = {score_base:.5f}  (训练耗时 {time.time()-t_train:.1f}s)")

# ============================================================
# 8. 三个新特征
# ============================================================
print("\n" + "=" * 60)
print("第5部分: 消融实验")
print("=" * 60)
print("原则: 一次只加一个变量, 与 Base 模型对比")


def make_extra_fn_1():
    """price_ratio = 商品均价 / 用户均价"""
    def fn(df):
        ratio = df["avg_price"].values / (df["avg_price_user"].values + 1e-6)
        ratio = np.clip(ratio, 0, 10)
        return {"price_ratio": ratio}
    return fn


def make_extra_fn_2():
    """recency_decay = exp(-last_buy_days / 30)"""
    def fn(df):
        decay = np.exp(-df["last_buy_days"].values.clip(0, 999) / 30)
        return {"recency_decay": decay}
    return fn


def make_extra_fn_3(user_hist, art_feat_df, candidates):
    """category_match = 商品品类是否匹配用户最常购买品类"""
    art_lookup = art_feat_df.set_index("article_id")
    user_top_cat = {}
    for cid in candidates:
        hist = user_hist.get(cid, [])
        if not hist:
            user_top_cat[cid] = -1
            continue
        cat_counts = defaultdict(int)
        for aid in hist:
            if aid in art_lookup.index:
                pg = art_lookup.loc[aid, "product_group_name_le"]
                if hasattr(pg, "iloc"):
                    pg = pg.iloc[0] if len(pg) > 0 else -1
                cat_counts[pg] += 1
        user_top_cat[cid] = max(cat_counts, key=cat_counts.get) if cat_counts else -1

    def fn(df):
        a_arr = df["article_id"].values
        c_arr = df["customer_id"].values
        match = np.zeros(len(df), dtype=np.float32)
        for i, (cid, aid) in enumerate(zip(c_arr, a_arr)):
            if cid in user_top_cat and aid in art_lookup.index:
                art_cat = art_lookup.loc[aid, "product_group_name_le"]
                if hasattr(art_cat, "iloc"):
                    art_cat = art_cat.iloc[0] if len(art_cat) > 0 else -1
                match[i] = 1.0 if float(art_cat) == float(user_top_cat[cid]) else 0.0
        return {"category_match": match}
    return fn


# ============================================================
# 9. 逐个特征消融
# ============================================================

# 特征1: price_ratio
print("\n[实验1] Base + price_ratio")
ltr1, feats1 = build_ltr_data(
    candidates, val_labels, cus_feat, art_feat, inter_feat,
    item_sim, user_hist, extra_fn=make_extra_fn_1()
)
score_f1, _ = train_and_eval(ltr1, feats1, LGB_PARAMS, val_cids, val_gt)
delta1 = score_f1 - score_base
keep1 = "保留" if delta1 > 1e-5 else "删除"
print(f"  Base: {score_base:.5f}  →  +price_ratio: {score_f1:.5f}  (Δ={delta1:+.5f})  →  {keep1}")
del ltr1; gc.collect()

# 特征2: recency_decay
print("\n[实验2] Base + recency_decay")
ltr2, feats2 = build_ltr_data(
    candidates, val_labels, cus_feat, art_feat, inter_feat,
    item_sim, user_hist, extra_fn=make_extra_fn_2()
)
score_f2, _ = train_and_eval(ltr2, feats2, LGB_PARAMS, val_cids, val_gt)
delta2 = score_f2 - score_base
keep2 = "保留" if delta2 > 1e-5 else "删除"
print(f"  Base: {score_base:.5f}  →  +recency_decay: {score_f2:.5f}  (Δ={delta2:+.5f})  →  {keep2}")
del ltr2; gc.collect()

# 特征3: category_match
print("\n[实验3] Base + category_match")
ltr3, feats3 = build_ltr_data(
    candidates, val_labels, cus_feat, art_feat, inter_feat,
    item_sim, user_hist, extra_fn=make_extra_fn_3(user_hist, art_feat, candidates)
)
score_f3, _ = train_and_eval(ltr3, feats3, LGB_PARAMS, val_cids, val_gt)
delta3 = score_f3 - score_base
keep3 = "保留" if delta3 > 1e-5 else "删除"
print(f"  Base: {score_base:.5f}  →  +category_match: {score_f3:.5f}  (Δ={delta3:+.5f})  →  {keep3}")
del ltr3; gc.collect()

# ============================================================
# 10. 结果汇总
# ============================================================
print("\n" + "=" * 65)
print("消融实验结果汇总")
print("=" * 65)

print(f"\n  {'特征名称':<22s} {'添加后MAP@12':>14s} {'分数变化(Δ)':>14s} {'决定':>8s}")
print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*8}")
print(f"  {'Base (46维)':<22s} {score_base:>14.5f} {'—':>14s} {'—':>8s}")
print(f"  {'+ price_ratio':<22s} {score_f1:>14.5f} {delta1:>+14.5f} {keep1:>8s}")
print(f"  {'+ recency_decay':<22s} {score_f2:>14.5f} {delta2:>+14.5f} {keep2:>8s}")
print(f"  {'+ category_match':<22s} {score_f3:>14.5f} {delta3:>+14.5f} {keep3:>8s}")

# Markdown 表格
print("\nMarkdown 表格:")
print(f"| 特征名称 | 添加后 MAP@12 | 分数变化 (Δ) | 决定 |")
print(f"|:---|:---:|:---:|:---:|")
print(f"| Base (46维) | {score_base:.5f} | — | — |")
print(f"| + price_ratio | {score_f1:.5f} | {delta1:+.5f} | {keep1} |")
print(f"| + recency_decay | {score_f2:.5f} | {delta2:+.5f} | {keep2} |")
print(f"| + category_match | {score_f3:.5f} | {delta3:+.5f} | {keep3} |")

# ============================================================
# 11. 保存结果
# ============================================================
results_path = os.path.join(OUT_DIR, "feature_ablation_results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("特征消融实验汇总\n")
    f.write(f"流行度 Baseline MAP@12: {score_pop:.5f}\n")
    f.write(f"Base 模型 MAP@12: {score_base:.5f}\n\n")
    f.write(f"{'特征名称':<22s} {'添加后MAP@12':>14s} {'分数变化(Δ)':>14s} {'决定':>8s}\n")
    f.write(f"{'-'*22} {'-'*14} {'-'*14} {'-'*8}\n")
    f.write(f"{'Base (46维)':<22s} {score_base:>14.5f} {'—':>14s} {'—':>8s}\n")
    f.write(f"{'+ price_ratio':<22s} {score_f1:>14.5f} {delta1:>+14.5f} {keep1:>8s}\n")
    f.write(f"{'+ recency_decay':<22s} {score_f2:>14.5f} {delta2:>+14.5f} {keep2:>8s}\n")
    f.write(f"{'+ category_match':<22s} {score_f3:>14.5f} {delta3:>+14.5f} {keep3:>8s}\n")

print(f"\n结果已保存: {results_path}")

# ============================================================
# 12. 总结
# ============================================================
print("\n" + "=" * 65)
print("总结")
print("=" * 65)
print(f"  流行度 Baseline: MAP@12 = {score_pop:.5f}")
print(f"  LightGBM Base:    MAP@12 = {score_base:.5f}")
print(f"  模型提升:         {(score_base/score_pop - 1)*100:.1f}%\n")

for name, delta, keep in [
    ("price_ratio", delta1, keep1),
    ("recency_decay", delta2, keep2),
    ("category_match", delta3, keep3),
]:
    status = "✓ 有效, 建议保留" if keep == "保留" else "✗ 无效, 建议移除"
    print(f"  {name:<20s} Δ={delta:+.5f}  {status}")

print(f"\n✅ 完成!\n")
