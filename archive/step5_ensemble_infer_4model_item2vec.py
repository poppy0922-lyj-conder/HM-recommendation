"""
Step 5 Ensemble Infer (4-Model with Item2Vec): lowlr + catboost + xgboost + item2vec
=================================================================================
基于 step5_ensemble_infer_3model.py, 添加 Item2Vec(56维) 作为第4个异构模型

四模型等权平均:
  - LightGBM LambdaRank (lowlr)        - 23维特征
  - CatBoost YetiRank (catboost)       - 23维特征
  - XGBoost rank:ndcg (xgboost)        - 23维特征
  - LightGBM LambdaRank (item2vec)     - 56维特征(32 a2v + 1 v2v_sim + 23基线)

异构性来源:
  - 四种不同模型/损失函数 → 误差模式互不相关 → 集成增益最大

输入: model_lowlr.txt / model_catboost.cbm / model_xgboost.json / model_item2vec.txt
输入: word2vec_article.model (用于构建 a2v 和 v2v_sim 特征)
输出: submission_ensemble_item2vec.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle, gc, time, warnings
from collections import defaultdict
from catboost import CatBoost
from gensim.models import Word2Vec as W2V

import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR,
    INFER_BATCH_SIZE,
)
from utils import timer

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
print("=" * 60)
print("Step 5 Ensemble Infer (4-Model): lowlr + catboost + xgboost + item2vec")
print("=" * 60)

# ============================================================
# 特征列定义
# ============================================================
# 23维基线特征 (lowlr / catboost / xgboost)
CUS_COLS_CLEAN = ['age', 'postal_le', 'R_days', 'n_unique_articles']
ART_COLS_CLEAN = [
    'popularity_score', 'price_log', 'sales_log',
    'product_group_name_le', 'product_type_name_le',
    'colour_group_name_le', 'index_name_le',
]
ART_COLS_CLEAN += [f'text_emb_{i}' for i in [0, 1, 6, 7, 15, 16, 18]]
INTER_COLS_CLEAN = ['buy_count', 'last_buy_days', 'first_buy_days']
CAND_COLS_CLEAN = ['cf_score', 'price_match']
FEAT_COLS_23 = CUS_COLS_CLEAN + ART_COLS_CLEAN + INTER_COLS_CLEAN + CAND_COLS_CLEAN

# 56维特征 (item2vec: 23基线 + 32 a2v + 1 v2v_sim)
A2V_COLS = [f'a2v_{i}' for i in range(32)]
V2V_SIM_COL = ['v2v_sim']
FEAT_COLS_56 = FEAT_COLS_23 + A2V_COLS + V2V_SIM_COL

print(f"  模型1-3: {len(FEAT_COLS_23)}维 (lowlr/cat/xgb)")
print(f"  模型4:   {len(FEAT_COLS_56)}维 (item2vec)")

from config import CUS_COLS, ART_COLS, INTER_COLS

# ============================================================
# 加载模型
# ============================================================
with timer("加载模型"):
    model_lowlr    = lgb.Booster(model_file=f"{OUTPUT_DIR}/model_lowlr.txt")
    model_catboost = CatBoost()
    model_catboost.load_model(f"{OUTPUT_DIR}/model_catboost.cbm")
    model_xgboost  = xgb.Booster()
    model_xgboost.load_model(f"{OUTPUT_DIR}/model_xgboost.json")
    model_item2vec = lgb.Booster(model_file=f"{OUTPUT_DIR}/model_item2vec.txt")
print("  lowlr(LambdaRank) + catboost(YetiRank) + xgboost(rank:ndcg) + item2vec(56dim) 加载完成")

# ============================================================
# 加载 Word2Vec (用于构建 a2v + v2v_sim)
# ============================================================
with timer("加载 Word2Vec 模型"):
    w2v_model = W2V.load(f"{OUTPUT_DIR}/word2vec_article.model")
    art2v_full = {aid: w2v_model.wv[aid] for aid in w2v_model.wv.index_to_key}
    default_vec = np.zeros(32, dtype=np.float32)
    print(f"  Word2Vec 词表大小: {len(art2v_full):,}")

# ============================================================
# 候选生成 & LTR 构建 (推理模式, 含 Item2Vec 特征)
# ============================================================
def generate_candidates(user_hist, item_sim, art_pop, customers,
                        n_hist=12, n_pop=12, w2v_model=None, n_w2v=5):
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
        # Word2Vec 语义相似召回 (与 item2vec 模型正交)
        if w2v_model is not None:
            for aid in user_hist.get(cid, [])[:5]:
                if aid in w2v_model.wv:
                    for sim_aid, _ in w2v_model.wv.most_similar(aid, topn=n_w2v):
                        cands.add(sim_aid)
        for aid in pop_list:
            cands.add(aid)
        out[cid] = list(cands)
    return out


def build_ltr_data_item2vec(candidates, labels, cus_feat_df, art_feat_df,
                            inter_feat_df, item_sim, user_hist,
                            art2v, default_vec):
    """增强版 build_ltr_data: 新增 a2v 列 + v2v_sim 计算"""
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

    # 确保 art_feat_df 包含 a2v 列
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

    # === 计算 v2v_sim (余弦相似度) ===
    # 1. 预计算用户嵌入 (历史最近12个商品的平均)
    user_emb_cache = {}
    for cid in candidates:
        hist = user_hist.get(cid, [])[:12]
        if hist and all(a in art2v for a in hist):
            vecs = np.array([art2v[a] for a in hist], dtype=np.float32)
            user_emb_cache[cid] = vecs.mean(axis=0)
        else:
            user_emb_cache[cid] = default_vec.copy()

    # 2. 向量化计算余弦相似度
    cids_for_sim = df["customer_id"].values
    aids_for_sim = df["article_id"].values
    user_vecs = np.array([user_emb_cache.get(c, default_vec) for c in cids_for_sim], dtype=np.float32)
    art_vecs = np.array([art2v.get(a, default_vec) for a in aids_for_sim], dtype=np.float32)
    dot_products = (user_vecs * art_vecs).sum(axis=1)
    user_norms = np.linalg.norm(user_vecs, axis=1)
    art_norms = np.linalg.norm(art_vecs, axis=1)
    df[V2V_SIM_COL[0]] = (dot_products / (user_norms * art_norms + 1e-8)).astype(np.float32)

    return df


# ============================================================
# 加载全量特征 + 追加 a2v 嵌入
# ============================================================
with timer("加载全量特征 + a2v 嵌入"):
    cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
    art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
    inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
    with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
        item_sim_full = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
        user_hist_full = pickle.load(f)

    # 追加 a2v 嵌入到全量商品特征
    print("  追加 a2v 嵌入到全量商品特征...")
    for i in range(32):
        col = f"a2v_{i}"
        art_feat_full[col] = art_feat_full["article_id"].map(
            lambda x: art2v_full.get(x, default_vec)[i]
        ).astype(np.float32)

art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()

known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

# ============================================================
# 分批集成推理 (四模型等权平均)
# ============================================================
all_preds = {}
with timer(f"分批集成推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(user_hist_full, item_sim_full, art_pop_full, batch_cids, w2v_model=w2v_model)

        # 使用 Item2Vec 版 build_ltr_data (含 a2v + v2v_sim)
        inf_df = build_ltr_data_item2vec(
            batch_cands, {}, cus_feat_full, art_feat_full,
            inter_feat_full, item_sim_full, user_hist_full,
            art2v_full, default_vec,
        )

        # 确保所有特征列存在且类型正确
        for c in FEAT_COLS_56:
            if c not in inf_df.columns:
                inf_df[c] = np.float32(0)
            elif inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        X_23 = inf_df[FEAT_COLS_23].values
        X_56 = inf_df[FEAT_COLS_56].values

        # 四个模型分别预测 → 等权平均
        s_lowlr    = model_lowlr.predict(X_23)
        s_cat      = model_catboost.predict(X_23)
        dinf       = xgb.DMatrix(X_23)
        s_xgb      = model_xgboost.predict(dinf)
        s_item2vec = model_item2vec.predict(X_56)

        inf_df["score"] = (np.float64(s_lowlr) + np.float64(s_cat) +
                          np.float64(s_xgb) + np.float64(s_item2vec)) / 4.0

        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds, X_23, X_56, dinf; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_ensemble_item2vec.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_ensemble_item2vec.csv  ({len(sub):,} 行)")
print(f"\n{'='*60}")
print(f"四模型集成: lowlr + catboost + xgboost + item2vec")
print(f"{'='*60}")
print(f"  模型1 lowlr:     {len(FEAT_COLS_23)}维 LightGBM LambdaRank")
print(f"  模型2 catboost:  {len(FEAT_COLS_23)}维 CatBoost YetiRank")
print(f"  模型3 xgboost:   {len(FEAT_COLS_23)}维 XGBoost rank:ndcg")
print(f"  模型4 item2vec:  {len(FEAT_COLS_56)}维 LightGBM + Word2Vec")
print(f"  集成方式:        等权平均")
print(f"  冷启动比例:      {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*60}")
print(f"\n完成!")
