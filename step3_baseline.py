"""
Step 3: 基线模型评估
=====================
流行度 + Item-CF 简单推荐 → 验证集 MAP@12 评估 + Kaggle 提交文件

输出:
  output/baseline_pop.csv   流行度 Baseline 提交
  output/baseline_cf.csv    Item-CF Baseline 提交
"""

import pandas as pd
import numpy as np
import pickle

from config import DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, SEED
from utils import timer, mapk, set_seed

set_seed(SEED)

print("=" * 60)
print("Step 3: 基线模型评估")
print("=" * 60)

# 读取特征和数据
with timer("读取特征文件"):
    art_feat = pd.read_parquet(f"{PROCESSED_DIR}/art_feat.parquet")
    val_txn = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
    val_txn["t_dat"] = pd.to_datetime(val_txn["t_dat"])

    with open(f"{PROCESSED_DIR}/item_sim.pkl", "rb") as f:
        item_sim = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist.pkl", "rb") as f:
        user_hist = pickle.load(f)

# 构建验证集 ground truth
val_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_cids = list(val_gt.keys())
print(f"  验证集用户数: {len(val_cids):,}")

# 流行度分数
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]

# ============================================================
# 基线1 — 全量流行度
# ============================================================
with timer("基线1: 流行度"):
    actuals_pop = [val_gt[c] for c in val_cids]
    preds_pop = [pop12] * len(val_cids)
    score_pop = mapk(actuals_pop, preds_pop, k=12)
    print(f"  MAP@12 = {score_pop:.5f}")

    # 生成 Kaggle 提交文件 (全部客户)
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    pop_str = " ".join(pop12)
    sub["prediction"] = pop_str
    sub.to_csv(f"{OUTPUT_DIR}/baseline_pop.csv", index=False)
    print(f"  已保存 {OUTPUT_DIR}/baseline_pop.csv ({len(sub):,} 行)")

# ============================================================
# 基线2 — Item-CF 简单推荐
# ============================================================
def item_cf_recommend(user_hist, item_sim, art_pop, pop12, k=12):
    """基于 Item-CF 的简单推荐"""
    preds = {}
    for cid, hist_items in user_hist.items():
        scores = {}
        # 从历史商品出发，通过相似矩阵累加分数
        for aid in hist_items[:5]:
            if aid in item_sim:
                for rel, sc in item_sim[aid][:10]:
                    scores[rel] = scores.get(rel, 0) + sc
        recs = sorted(scores, key=lambda x: -scores[x])[:k]
        # 不足 k 个则用流行度填充
        if len(recs) < k:
            for pop_item in pop12:
                if pop_item not in recs:
                    recs.append(pop_item)
                if len(recs) >= k:
                    break
        preds[cid] = recs
    return preds


with timer("基线2: Item-CF"):
    cf_preds = item_cf_recommend(user_hist, item_sim, art_pop, pop12, k=12)
    preds_cf = [cf_preds.get(c, pop12) for c in val_cids]
    score_cf = mapk(actuals_pop, preds_cf, k=12)
    print(f"  MAP@12 = {score_cf:.5f}")

    # 生成 Kaggle 提交文件 (全部客户, 冷启动用户用流行度兜底)
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    pop_str = " ".join(pop12)
    sub["prediction"] = sub["customer_id"].apply(
        lambda cid: " ".join(cf_preds.get(cid, pop12)) if cid in cf_preds else pop_str
    )
    sub.to_csv(f"{OUTPUT_DIR}/baseline_cf.csv", index=False)
    print(f"  已保存 {OUTPUT_DIR}/baseline_cf.csv ({len(sub):,} 行)")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*45}")
print(f"  流行度 Baseline:  {score_pop:.5f}")
print(f"  Item-CF Baseline: {score_cf:.5f}")
print(f"{'='*45}")
print("\nStep 3 完成!")
