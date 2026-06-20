"""
Step 5 CatBoost: CatBoost YetiRank 排序
=========================================
基于 step5_lowlr.py 模板, 唯一改动:
  - 模型: LightGBM LambdaRank → CatBoost YetiRank
  - 训练/推理 API 适配 CatBoost
  - 参数对齐: lr=0.02, iterations=2000, early_stop=100, depth=8

YetiRank 是 CatBoost 自研排序损失, 误差模式与 LambdaRank 互补。

输出: output/model_catboost.cbm / output/submission_catboost.csv
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, warnings
from collections import defaultdict
from datetime import timedelta
from catboost import CatBoost, Pool
from gensim.models import Word2Vec as W2V

from config import (
    DATA_DIR, PROCESSED_DIR, OUTPUT_DIR,
    VAL_DAYS, INFER_BATCH_SIZE,
    CUS_COLS, ART_COLS, INTER_COLS,
    SEED,
)
from utils import timer, mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 60)
print("Step 5 CatBoost: 23维 + YetiRank + lowlr 基底")
print("=" * 60)

# ============ 特征列 (23维, 与 lowlr / xgboost 完全一致) ============
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

print(f"  特征: {len(FEAT_COLS_CLEAN)} 维  model=CatBoost YetiRank")

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

# Word2Vec 语义候选扩展
w2v_model_path = f"{OUTPUT_DIR}/word2vec_article.model"
if os.path.exists(w2v_model_path):
    w2v_model = W2V.load(w2v_model_path)
    print(f"  Word2Vec 词表大小: {len(w2v_model.wv):,}")
    art_pop_val = art_feat.set_index("article_id")["popularity_score"].to_dict()
    val_users = sorted(set(candidates.keys()))
    candidates_orig = candidates
    candidates = generate_candidates(user_hist, item_sim, art_pop_val, val_users, w2v_model=w2v_model)
    n_orig = sum(len(v) for v in candidates_orig.values())
    n_exp = sum(len(v) for v in candidates.values())
    print(f"  原始候选集: {n_orig:,}  /  扩展后: {n_exp:,}  (+{(n_exp-n_orig)/n_orig*100:.1f}%)")
    del candidates_orig, art_pop_val, val_users; gc.collect()
else:
    print("  Word2Vec 模型未找到, 跳过语义候选扩展")
    w2v_model = None

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分: val → train(前6天) + holdout(后1天)
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"  val_train: {len(val_train_txn):,}  |  val_holdout: {len(val_holdout_txn):,}")
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
# 通用函数 (与 lowlr 完全一致)
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


def prepare_ltr_cat(ltr_df):
    """转为 float32, 构建 CatBoost Pool (需要 group_id)"""
    for c in FEAT_COLS_CLEAN:
        if ltr_df[c].dtype == "float64":
            ltr_df[c] = ltr_df[c].astype(np.float32)

    ltr_df = ltr_df.sort_values("customer_id")
    X = ltr_df[FEAT_COLS_CLEAN].values
    y = ltr_df["label"].values

    # CatBoost uses group_id (int, from 0) for each query
    cid_codes = ltr_df["customer_id"].astype("category").cat.codes.values
    group_boundaries = []
    for i in range(1, len(cid_codes)):
        if cid_codes[i] != cid_codes[i - 1]:
            group_boundaries.append(i)
    group_boundaries.append(len(cid_codes))

    group_id = np.zeros(len(X), dtype=np.int32)
    start = 0
    for gid, end in enumerate(group_boundaries):
        group_id[start:end] = gid
        start = end

    return X, y, group_id, ltr_df


# 全量候选生成 (同 step6 的候选逻辑)
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
        if w2v_model is not None:
            for aid in user_hist.get(cid, [])[:5]:
                if aid in w2v_model.wv:
                    for sim_aid, _ in w2v_model.wv.most_similar(aid, topn=n_w2v):
                        cands.add(sim_aid)
        for aid in pop_list:
            cands.add(aid)
        out[cid] = list(cands)
    return out


# ============================================================
# 模型路径检测 (存在则跳过训练, 直接推理)
# ============================================================
model_path = f"{OUTPUT_DIR}/model_catboost.cbm"

if os.path.exists(model_path):
    print(f"\n检测到已存在的模型: {model_path}")
    print("跳过 Phase 1 & 2 训练，直接加载模型进行推理")

    with timer("加载已训练模型"):
        final_model = CatBoost()
        final_model.load_model(model_path)
        best_iter = final_model.tree_count_

    score_holdout = score_train_eval = score_pop_train = score_pop_ho = 0.0
    print(f"  模型加载完成, 迭代轮数: {best_iter}")

else:
    # ============================================================
    # Phase 1: 构建 LTR 数据 + 训练 + 评估
    # ============================================================
    print("\n[Phase 1 Train LTR]")
    with timer("构建 LTR 数据"):
        ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                                   inter_feat, item_sim, user_hist)

    print("\n[Phase 1 Holdout LTR]")
    ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                                  inter_feat, item_sim, user_hist)

    X_train, y_train, gid_train, ltr_train = prepare_ltr_cat(ltr_train)
    X_valid, y_valid, gid_valid, ltr_holdout = prepare_ltr_cat(ltr_holdout)

    train_pool = Pool(X_train, label=y_train, group_id=gid_train)
    valid_pool = Pool(X_valid, label=y_valid, group_id=gid_valid)

    print(f"\n  训练样本: {len(X_train):,}  |  验证样本: {len(X_valid):,}")
    print(f"  特征维度: {len(FEAT_COLS_CLEAN)}")

    # ============================================================
    # CatBoost YetiRank 参数
    # ============================================================
    cat_params = {
        "loss_function":        "YetiRank",
        "custom_metric":        ["NDCG:top=12"],
        "learning_rate":         0.02,
        "iterations":            2000,
        "depth":                 8,
        "l2_leaf_reg":           1.0,
        "random_seed":           SEED,
        "verbose":               50,
        "early_stopping_rounds": 100,
        "task_type":             "CPU",
        "thread_count":           -1,
        "allow_writing_files":   False,
    }

    print(f"\n使用 CatBoost YetiRank  lr=0.02  depth=8  iterations=2000")

    with timer("Phase 1 CatBoost 训练 (YetiRank + early_stop)"):
        model = CatBoost(cat_params)
        model.fit(train_pool, eval_set=valid_pool, plot=False)

    best_iter = model.get_best_iteration() or 2000
    print(f"\n  最佳迭代轮数: {best_iter}")

    # ============================================================
    # Phase 1 评估
    # ============================================================
    holdout_cids = [c for c in common_users if c in candidates]
    train_cids = [c for c in common_users if c in candidates]

    with timer("Phase 1 验证评估"):
        ltr_holdout["score"] = model.predict(valid_pool)
        preds_ho = (
            ltr_holdout.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        actuals_ho = [val_holdout_labels[c] for c in holdout_cids]
        preds_ho_l = [preds_ho.get(c, []) for c in holdout_cids]
        score_holdout = mapk(actuals_ho, preds_ho_l, k=12)

        ltr_train["score"] = model.predict(train_pool)
        preds_tr = (
            ltr_train.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        actuals_tr = [val_train_labels[c] for c in train_cids]
        preds_tr_l = [preds_tr.get(c, []) for c in train_cids]
        score_train_eval = mapk(actuals_tr, preds_tr_l, k=12)

    art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
    pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
    score_pop_train = mapk(actuals_tr, [pop12] * len(train_cids), k=12)
    score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

    print(f"\n{'='*55}")
    print(f"  Phase 1 Val 评估 (CatBoost YetiRank):")
    print(f"    ┌────────────────────┬──────────┬──────────┐")
    print(f"    │                    │  Train   │ Holdout  │")
    print(f"    ├────────────────────┼──────────┼──────────┤")
    print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
    print(f"    │ 23维 CatBoost       │ {score_train_eval:.5f}  │ {score_holdout:.5f}  │")
    print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
    print(f"    └────────────────────┴──────────┴──────────┘")
    print(f"    过拟合: train-holdout = {score_train_eval-score_holdout:.5f}")
    print(f"    最佳迭代轮数: {best_iter}")
    print(f"{'='*55}")

    # ============================================================
    # Phase 2: 全量 val 数据训练最终模型
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
            val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist
        )
    X_full, y_full, gid_full, ltr_full = prepare_ltr_cat(ltr_full)
    full_pool = Pool(X_full, label=y_full, group_id=gid_full)

    final_params = cat_params.copy()
    final_params["iterations"] = best_iter
    final_params.pop("early_stopping_rounds", None)

    with timer(f"Phase 2 训练 (固定 {best_iter} 轮)"):
        final_model = CatBoost(final_params)
        final_model.fit(full_pool, plot=False, verbose=50)

    final_model.save_model(model_path)
    print(f"\n模型已保存: {model_path}")

    del full_pool, X_full, y_full, model, ltr_train, ltr_holdout, ltr_full
    gc.collect()

del candidates, cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist
gc.collect()

# ============================================================
# Phase 3: 全量推理 + 提交
# ============================================================
print(f"\n{'='*60}")
print("Phase 3: 全量推理")
print(f"{'='*60}")

with timer("读取全量数据"):
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
known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} "
      f"({len(sub_cold)/len(sub_cids)*100:.1f}%)")


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
        if w2v_model is not None:
            for aid in user_hist.get(cid, [])[:5]:
                if aid in w2v_model.wv:
                    for sim_aid, _ in w2v_model.wv.most_similar(aid, topn=n_w2v):
                        cands.add(sim_aid)
        for aid in pop_list:
            cands.add(aid)
        out[cid] = list(cands)
    return out


with timer("全量候选生成"):
    cands_full = generate_candidates(user_hist_full, item_sim_full, art_pop_full, sub_cids, w2v_model=w2v_model)
    tot = sum(len(v) for v in cands_full.values())
    print(f"  候选: {len(cands_full):,} 用户, 平均 {tot/len(cands_full):.1f} 候选/用户")

del w2v_model; gc.collect()

all_preds = {}
with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        bnum = start // INFER_BATCH_SIZE + 1
        batch_cands = {c: cands_full[c] for c in batch_cids}

        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        X_inf = inf_df[FEAT_COLS_CLEAN].values
        inf_df["score"] = final_model.predict(X_inf)
        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {bnum}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_catboost.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_catboost.csv  ({len(sub):,} 行)")

del cands_full, all_preds; gc.collect()

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*55}")
print(f"  结果汇总: CatBoost YetiRank")
print(f"{'='*55}")
print(f"  loss / lr / depth / stop:        YetiRank / 0.02 / 8 / 100")
print(f"  Phase 1 Holdout MAP@12:          {score_holdout:.5f}")
print(f"  Phase 1 Train MAP@12:            {score_train_eval:.5f}")
print(f"  过拟合:                           {score_train_eval-score_holdout:.5f}")
print(f"  流行度 Baseline (holdout):        {score_pop_ho:.5f}")
print(f"  模型提升:                          +{score_holdout-score_pop_ho:.5f}")
print(f"  最佳迭代轮数:                     {best_iter}")
print(f"  特征维度:                          {len(FEAT_COLS_CLEAN)}")
print(f"  冷启动比例:                       {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*55}")
print(f"\n完成!")
