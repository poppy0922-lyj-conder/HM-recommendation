"""
Step 2: 特征工程 (5类特征)
===========================
仅使用 train_txn 构建特征，避免数据泄露。

输出 (保存到 PROCESSED_DIR/):
  cus_feat.parquet     客户特征   (8维)
  art_feat.parquet     商品特征   (33维)
  inter_feat.parquet   交互特征   (3维)
  item_sim.pkl         Item-CF 共现矩阵
  user_hist.pkl        用户历史购买列表
"""

import pandas as pd
import numpy as np
import pickle
import gc
from collections import defaultdict
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config import (
    PROCESSED_DIR, TFIDF_MAX_FEATURES, SVD_N_COMPONENTS,
    ITEM_CF_TOP_K, ITEM_CF_MIN_CNT, USER_HIST_MAX, SEED
)
from utils import timer, set_seed

set_seed(SEED)

print("=" * 60)
print("Step 2: 特征工程")
print("=" * 60)

# 读数据
with timer("读取 Parquet"):
    train_txn = pd.read_parquet(f"{PROCESSED_DIR}/train_txn.parquet")
    art = pd.read_parquet(f"{PROCESSED_DIR}/articles.parquet")
    cus = pd.read_parquet(f"{PROCESSED_DIR}/customers.parquet")
    train_txn["t_dat"] = pd.to_datetime(train_txn["t_dat"])
    print(f"  train_txn: {len(train_txn):,}  art: {len(art):,}  cus: {len(cus):,}")


# ============================================================
# 2.1 客户特征 (8维)
# ============================================================
def build_customer_features(txn_df, cus_df):
    """构建客户特征: age + 会员/邮编编码 + RFM"""
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


with timer("2.1 客户特征"):
    cus_feat = build_customer_features(train_txn, cus)
    print(f"  维度: {cus_feat.shape}")


# ============================================================
# 2.2 商品特征 (33维)
# ============================================================
def build_article_features(txn_df, art_df):
    """构建商品特征: 价格/销量/流行度/类别编码/TF-IDF嵌入"""
    # 基本统计
    stats = txn_df.groupby("article_id", sort=False).agg(
        avg_price=("price", "mean"),
        sales_count=("article_id", "count"),
        n_buyers=("customer_id", "nunique"),
    ).reset_index()

    # 时间衰减流行度
    max_d = txn_df["t_dat"].max()
    _td = txn_df[["article_id", "t_dat"]].copy()
    _td["days"] = (max_d - _td["t_dat"]).dt.days
    _td["w"] = np.exp(-_td["days"] / 14)
    pop = _td.groupby("article_id", sort=False)["w"].sum().reset_index()
    pop.rename(columns={"w": "popularity_score"}, inplace=True)
    del _td, txn_df
    gc.collect()

    df = art_df[["article_id"]].copy()
    df = df.merge(stats, on="article_id", how="left")
    df = df.merge(pop, on="article_id", how="left")
    for c in ["avg_price", "sales_count", "n_buyers", "popularity_score"]:
        df[c] = df[c].fillna(0)
    df["price_log"] = np.log1p(df["avg_price"])
    df["sales_log"] = np.log1p(df["sales_count"])

    # 7个类别列 LabelEncode
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

    # TF-IDF + SVD 文本嵌入
    desc = art_df["detail_desc"].fillna(art_df.get("product_type_name", "")).fillna("")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=SEED)
    emb = svd.fit_transform(tfidf.fit_transform(desc.str.lower().fillna("")))
    for i in range(SVD_N_COMPONENTS):
        df[f"text_emb_{i}"] = emb[:, i]
    del desc, emb
    gc.collect()
    print(f"  TF-IDF+SVD 解释方差比: {svd.explained_variance_ratio_.sum():.2%}")
    return df


with timer("2.2 商品特征"):
    art_feat = build_article_features(train_txn.copy(), art)
    print(f"  维度: {art_feat.shape}")


# ============================================================
# 2.3 交互特征 (3维)
# ============================================================
def build_interaction_features(txn_df):
    """构建用户-商品交互特征"""
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


with timer("2.3 交互特征"):
    inter_feat = build_interaction_features(train_txn)
    print(f"  维度: {inter_feat.shape}")


# ============================================================
# 2.4 Item-CF 共现矩阵
# ============================================================
def build_item_cf(train_df, top_k=ITEM_CF_TOP_K, min_cnt=ITEM_CF_MIN_CNT):
    """构建 Item-CF 共现矩阵"""
    df = train_df[["customer_id", "article_id", "t_dat"]].copy()
    df["uw"] = (
        df["customer_id"] + "_" +
        df["t_dat"].dt.year.astype(str) + "_" +
        df["t_dat"].dt.isocalendar().week.astype(str)
    )

    # 过滤低频商品
    vc = df["article_id"].value_counts()
    keep = set(vc[vc >= min_cnt].index)
    df = df.loc[df["article_id"].isin(keep)]

    # 共现计数
    cooc = defaultdict(lambda: defaultdict(int))
    for uw, grp in df.groupby("uw", sort=False):
        items = sorted(set(grp["article_id"].tolist()))
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, min(len(items), i + 5)):
                cooc[items[i]][items[j]] += 1
                cooc[items[j]][items[i]] += 1
    del df
    gc.collect()

    # 每商品保留 top_k 相似
    sim = {a: sorted(r.items(), key=lambda x: -x[1])[:top_k] for a, r in cooc.items()}
    print(f"  Item-CF: {len(sim):,} 个商品")
    return sim


with timer("2.4 Item-CF 共现矩阵"):
    item_sim = build_item_cf(train_txn)


# ============================================================
# 2.5 用户历史购买
# ============================================================
def build_user_history(train_df, max_items=USER_HIST_MAX):
    """每个用户最近购买的去重商品列表"""
    df = train_df[["customer_id", "article_id", "t_dat"]].sort_values(
        "t_dat", ascending=False
    )
    df = df.drop_duplicates(subset=["customer_id", "article_id"], keep="first")
    df = df.groupby("customer_id").head(max_items)
    return df.groupby("customer_id")["article_id"].apply(list).to_dict()


with timer("2.5 用户历史"):
    user_hist = build_user_history(train_txn)
    print(f"  用户数: {len(user_hist):,}")

# ============================================================
# 保存所有特征
# ============================================================
with timer("保存特征文件"):
    cus_feat.to_parquet(f"{PROCESSED_DIR}/cus_feat.parquet", index=False)
    art_feat.to_parquet(f"{PROCESSED_DIR}/art_feat.parquet", index=False)
    inter_feat.to_parquet(f"{PROCESSED_DIR}/inter_feat.parquet", index=False)

    with open(f"{PROCESSED_DIR}/item_sim.pkl", "wb") as f:
        pickle.dump(item_sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{PROCESSED_DIR}/user_hist.pkl", "wb") as f:
        pickle.dump(user_hist, f, protocol=pickle.HIGHEST_PROTOCOL)

# 打印汇总
print(f"\n特征文件已保存到 {PROCESSED_DIR}/")
print(f"  cus_feat.parquet     — {cus_feat.shape}")
print(f"  art_feat.parquet     — {art_feat.shape}")
print(f"  inter_feat.parquet   — {inter_feat.shape}")
print(f"  item_sim.pkl         — {len(item_sim):,} 商品")
print(f"  user_hist.pkl        — {len(user_hist):,} 用户")
print("\nStep 2 完成!")
