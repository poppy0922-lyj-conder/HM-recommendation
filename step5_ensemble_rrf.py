"""
Step 5 Ensemble RRF: 分数归一化 + 加权融合 集成
=================================================
相比 step5_ensemble_infer (直接分数平均), 改进:
  - 每个用户内做 min-max 归一化, 消除不同模型分数尺度差异
  - 支持按模型质量设置不同权重
  - 同时保留 RRF 方案可切换对比

融合方式 (FUSION_METHOD):
  "norm"  — 用户内 min-max 归一化 + 加权平均 (推荐)
  "rrf"   — Reciprocal Rank Fusion, 纯排名融合

输入: MODEL_CONFIG 指定的模型文件列表
输出: submission_ensemble_rrf.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, warnings
from collections import defaultdict

from config import (
    DATA_DIR, PROCESSED_DIR,
    INFER_BATCH_SIZE,
)
from utils import timer

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# ============================================================
# 模型配置 — 在此添加/删除模型、调权重
# ============================================================
FUSION_METHOD = "norm"   # "norm"=归一化+加权  |  "rrf"=排名融合
RRF_K = 60               # RRF 平滑常数 (仅 method="rrf" 时有效)

MODEL_CONFIG = [
    {
        "name": "lowlr",
        "type": "lgb",
        "path": f"{OUTPUT_DIR}/model_lowlr.txt",
        "weight": 1.0,
    },
    {
        "name": "catboost",
        "type": "catboost",
        "path": f"{OUTPUT_DIR}/model_catboost.cbm",
        "weight": 0.3,   # 如果 catboost 单模型较低, 降低权重
    },
]

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

from config import CUS_COLS, ART_COLS, INTER_COLS

# ============================================================
# 加载模型
# ============================================================
with timer("加载模型"):
    models = []
    for cfg in MODEL_CONFIG:
        if cfg["type"] == "lgb":
            import lightgbm as lgb
            m = lgb.Booster(model_file=cfg["path"])
        elif cfg["type"] == "catboost":
            from catboost import CatBoost
            m = CatBoost()
            m.load_model(cfg["path"])
        else:
            raise ValueError(f"未知模型类型: {cfg['type']}")
        models.append({"model": m, "weight": cfg["weight"], "name": cfg["name"],
                        "type": cfg["type"]})
        print(f"  [{cfg['name']}] {cfg['type']} 加载完成  (权重={cfg['weight']})")

print(f"  共 {len(models)} 个模型  fusion={FUSION_METHOD}")
if FUSION_METHOD == "norm":
    print(f"  融合方式: 用户内 min-max 归一化 + 加权平均")
elif FUSION_METHOD == "rrf":
    print(f"  融合方式: RRF 排名融合  K={RRF_K}")

# ============================================================
# 候选生成 & LTR构建
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

    return df


# ============================================================
# 加载全量特征
# ============================================================
with timer("加载全量特征"):
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
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

# ============================================================
# RRF 分批推理
# ============================================================
all_preds = {}
with timer(f"分批 RRF 推理 (batch={INFER_BATCH_SIZE}, {len(models)}模型)"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num = start // INFER_BATCH_SIZE + 1

        # 构建推理数据
        batch_cands = generate_candidates(user_hist_full, item_sim_full, art_pop_full, batch_cids)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        X_inf = inf_df[FEAT_COLS_CLEAN].values

        if FUSION_METHOD == "norm":
            # ★ 方案1: 用户内 min-max 归一化 + 加权平均
            # 消除分数尺度差异, 同时保留模型对商品区分度的信息
            inf_df["final_score"] = 0.0
            for m in models:
                raw_score = m["model"].predict(X_inf)
                tmp_col = f"_s_{m['name']}"
                inf_df[tmp_col] = raw_score
                # 每个用户内 min-max 归一化到 [0, 1]
                inf_df[f"_n_{m['name']}"] = inf_df.groupby("customer_id")[tmp_col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
                )
                inf_df["final_score"] += m["weight"] * inf_df[f"_n_{m['name']}"]
                del inf_df[tmp_col], inf_df[f"_n_{m['name']}"]

        elif FUSION_METHOD == "rrf":
            # ★ 方案2: RRF 排名融合
            inf_df["final_score"] = 0.0
            for m in models:
                raw_score = m["model"].predict(X_inf)
                tmp_col = f"_tmp_{m['name']}"
                rank_col = f"_rank_{m['name']}"
                inf_df[tmp_col] = raw_score
                inf_df[rank_col] = inf_df.groupby("customer_id")[tmp_col].transform(
                    lambda x: x.rank(ascending=False, method="first")
                )
                inf_df["final_score"] += m["weight"] / (RRF_K + inf_df[rank_col])
                del inf_df[tmp_col], inf_df[rank_col]

        # 按融合分数排序取 top-12
        batch_preds = (
            inf_df.sort_values(["customer_id", "final_score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds, X_inf; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_ensemble_rrf.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_ensemble_rrf.csv  ({len(sub):,} 行)")

print(f"\n{'='*55}")
print(f"集成结果:")
print(f"{'='*55}")
model_info = ", ".join(f"{m['name']}(w={m['weight']})" for m in models)
print(f"  模型: {model_info}")
print(f"  融合方式: {FUSION_METHOD}")
print(f"{'='*55}")
print(f"\n完成!")
