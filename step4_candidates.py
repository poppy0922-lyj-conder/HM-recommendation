"""
Step 4: LTR 候选生成
=====================
为每个验证集用户生成候选商品列表 (历史购买 + Item-CF + 流行度)

输出: PROCESSED_DIR/val_candidates.pkl
"""

import pandas as pd
import pickle

from config import PROCESSED_DIR, SEED
from utils import timer, set_seed

set_seed(SEED)

print("=" * 60)
print("Step 4: 候选生成")
print("=" * 60)

with timer("读取特征和验证集"):
    art_feat = pd.read_parquet(f"{PROCESSED_DIR}/art_feat.parquet")
    val_txn = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
    val_txn["t_dat"] = pd.to_datetime(val_txn["t_dat"])

    with open(f"{PROCESSED_DIR}/item_sim.pkl", "rb") as f:
        item_sim = pickle.load(f)
    with open(f"{PROCESSED_DIR}/user_hist.pkl", "rb") as f:
        user_hist = pickle.load(f)

# 验证集用户
val_cids = val_txn["customer_id"].unique().tolist()
print(f"  验证集用户数: {len(val_cids):,}")

# 流行度列表
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()


def generate_candidates(user_hist, item_sim, art_pop, customers,
                        n_hist=12, n_cf=12, n_pop=12):
    """为每个用户生成候选商品"""
    pop_list = sorted(art_pop, key=lambda x: -art_pop[x])[:n_pop]
    out = {}
    for cid in customers:
        cands = set()
        # 历史购买商品
        for aid in user_hist.get(cid, [])[:n_hist]:
            cands.add(aid)
        # Item-CF 协同过滤
        for aid in user_hist.get(cid, [])[:5]:
            if aid in item_sim:
                for rel, _ in item_sim[aid][:10]:
                    cands.add(rel)
        # 流行度商品
        for aid in pop_list:
            cands.add(aid)
        out[cid] = list(cands)
    return out


with timer("候选生成"):
    candidates = generate_candidates(user_hist, item_sim, art_pop, val_cids)

tot = sum(len(v) for v in candidates.values())
print(f"  候选用户数: {len(candidates):,}")
print(f"  平均候选数: {tot / len(candidates):.1f}")

# 保存
with open(f"{PROCESSED_DIR}/val_candidates.pkl", "wb") as f:
    pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n已保存 {PROCESSED_DIR}/val_candidates.pkl")
print("\nStep 4 完成!")
