"""
Step 6: 全量特征重建 + 分批推理 + 生成提交文件
==============================================
使用 Step 5 训练好的模型，在全量数据上重建特征并推理。

输出: output/submission.csv
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR,
    CUS_COLS, ART_COLS, INTER_COLS, FEAT_COLS,
    INFER_BATCH_SIZE, TFIDF_MAX_FEATURES, SVD_N_COMPONENTS,
    USER_HIST_MAX, ITEM_CF_TOP_K, ITEM_CF_MIN_CNT,
    SEED,
)
from utils import timer, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 6: 全量推理 + 生成提交文件")
print("=" * 60)

# ============================================================
# 读取原始全量数据
# ============================================================
with timer("读取全量数据"):
    train_txn = pd.read_parquet(f"{PROCESSED_DIR}/train_txn.parquet")
    val_txn = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
    art = pd.read_parquet(f"{PROCESSED_DIR}/articles.parquet")
    cus = pd.read_parquet(f"{PROCESSED_DIR}/customers.parquet")
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

    train_txn["t_dat"] = pd.to_datetime(train_txn["t_dat"])
    val_txn["t_dat"] = pd.to_datetime(val_txn["t_dat"])
    full_txn = pd.concat([train_txn, val_txn], ignore_index=True)
    print(f"  全量交易: {len(full_txn):,} 行  |  提交用户: {len(sub):,}")

# ============================================================
# 加载 Step 5 训练好的模型
# ============================================================
with timer("加载模型"):
    model = lgb.Booster(model_file=f"{OUTPUT_DIR}/model.txt")
    print(f"  模型已加载  |  迭代轮数: {model.current_iteration()}")

# ============================================================
# 全量特征重建 (与 step2 相同逻辑，应用于全量数据)
# ============================================================
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
    del _td
    gc.collect()

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
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=SEED)
    emb = svd.fit_transform(tfidf.fit_transform(desc.str.lower().fillna("")))
    for i in range(SVD_N_COMPONENTS):
        df[f"text_emb_{i}"] = emb[:, i]
    del desc, emb
    gc.collect()
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


def build_item_cf(train_df, top_k=ITEM_CF_TOP_K, min_cnt=ITEM_CF_MIN_CNT):
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
    del df
    gc.collect()
    return {a: sorted(r.items(), key=lambda x: -x[1])[:top_k] for a, r in cooc.items()}


def build_user_history(train_df, max_items=USER_HIST_MAX):
    df = train_df[["customer_id", "article_id", "t_dat"]].sort_values(
        "t_dat", ascending=False
    )
    df = df.drop_duplicates(subset=["customer_id", "article_id"], keep="first")
    df = df.groupby("customer_id").head(max_items)
    return df.groupby("customer_id")["article_id"].apply(list).to_dict()


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
    cf_col = np.array([cf_map.get(c, {}).get(a, 0.0)
                       for c, a in zip(c_arr, a_arr)], dtype=np.float32)
    df["cf_score"] = cf_col
    df["price_match"] = (
        -np.abs(df["avg_price"].values - df["avg_price_user"].values)
    ).astype(np.float32)
    del c_arr, a_arr, cf_col, cf_map
    gc.collect()
    return df


# ============================================================
# 全量特征重建
# ============================================================
with timer("全量特征重建"):
    cus_feat_full = build_customer_features(full_txn, cus)
    print(f"  客户特征: {cus_feat_full.shape}")

    art_feat_full = build_article_features(full_txn, art)
    print(f"  商品特征: {art_feat_full.shape}")

    inter_feat_full = build_interaction_features(full_txn)
    print(f"  交互特征: {inter_feat_full.shape}")

    user_hist_full = build_user_history(full_txn)
    print(f"  用户历史: {len(user_hist_full):,}")

    item_sim_full = build_item_cf(full_txn)
    print(f"  Item-CF: {len(item_sim_full):,} 商品")

    art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()

# 释放全量交易数据
del full_txn, train_txn, val_txn
gc.collect()

# ============================================================
# 全量候选生成
# ============================================================
sub_cids = sub["customer_id"].tolist()

with timer("全量候选生成"):
    cands_full = generate_candidates(user_hist_full, item_sim_full, art_pop_full, sub_cids)
    tot = sum(len(v) for v in cands_full.values())
    print(f"  候选: {len(cands_full):,} 用户, 平均 {tot/len(cands_full):.1f} 候选/用户")

# ============================================================
# 分批推理
# ============================================================
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]
all_preds = {}

# 冷启动诊断
known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"\n  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_cands = {c: cands_full[c] for c in batch_cids}

        inf_df = build_ltr_data(
            batch_cands, {}, cus_feat_full, art_feat_full,
            inter_feat_full, item_sim_full, user_hist_full
        )

        # 特征转 float32
        for c in FEAT_COLS:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        inf_df["score"] = model.predict(inf_df[FEAT_COLS].values)

        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)

        del inf_df, batch_cands
        gc.collect()
        print(f"  Batch {start // INFER_BATCH_SIZE + 1}: {len(batch_cids):,} 用户完成")

# ============================================================
# 生成提交文件
# ============================================================
with timer("生成提交文件"):
    sub["prediction"] = sub["customer_id"].map(
        lambda x: " ".join(all_preds.get(x, pop12_full))
    )
    sub.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
    print(f"  提交行数: {len(sub):,}")
    print(f"  已保存: {OUTPUT_DIR}/submission.csv")

# 打印前5行预览
print("\n提交文件预览:")
print(sub.head().to_string(index=False))
print("\nStep 6 完成!")
