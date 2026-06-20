"""
Step 5 Two-Tower: 神经网络双塔排序模型
======================================
基于 step5_item2vec.py 改造:
  - 双塔 MLP (用户塔 + 商品塔 + 上下文偏置)
  - 点积打分的神经网络替代 LightGBM LambdaRank
  - 与树模型误差模式互补, 用于后续异构集成

特征: 36(用户塔) + 43(商品塔) + 6(上下文) = 85维输入
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec

from config import (
    DATA_DIR, PROCESSED_DIR, LGB_PARAMS,
    VAL_DAYS, INFER_BATCH_SIZE,
)
from utils import timer, mapk

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("Step 5 Two-Tower: 双塔神经网络排序")
print(f"  Device: {device}")
print("=" * 60)

# ============================================================
# 特征列定义
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

A2V_COLS = [f'a2v_{i}' for i in range(32)]

# 用户塔输入: 用户特征 + 用户历史 a2v 均值
USER_TOWER_COLS = CUS_COLS_CLEAN + A2V_COLS  # 4 + 32 = 36
# 商品塔输入: 商品特征 + 商品 a2v
ITEM_TOWER_COLS = ART_COLS_CLEAN + A2V_COLS   # 11 + 32 = 43
# 上下文特征 (用户-商品对级别)
CTX_TOWER_COLS = INTER_COLS_CLEAN + CAND_COLS_CLEAN + ['v2v_sim']  # 3 + 2 + 1 = 6

FEAT_COLS_CLEAN = USER_TOWER_COLS + ITEM_TOWER_COLS + CTX_TOWER_COLS

print(f"  用户塔: {len(USER_TOWER_COLS)}维 (用户特征4 + a2v均值32)")
print(f"  商品塔: {len(ITEM_TOWER_COLS)}维 (商品特征11 + a2v32)")
print(f"  上下文: {len(CTX_TOWER_COLS)}维 (交互3 + 候选2 + v2v_sim1)")

from config import CUS_COLS, ART_COLS, INTER_COLS

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 双塔模型定义
# ============================================================
class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, ctx_dim, emb_dim=16):
        super().__init__()
        # 用户塔: 36 → 64 → 32 → emb_dim
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, emb_dim),
        )
        # 商品塔: 43 → 64 → 32 → emb_dim
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, emb_dim),
        )
        # 上下文偏置: 6 → 16 → 8 → 1
        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, user_feat, item_feat, ctx_feat):
        u = self.user_tower(user_feat)
        i = self.item_tower(item_feat)
        dot = (u * i).sum(dim=1, keepdim=True)
        bias = self.ctx_net(ctx_feat)
        return dot + bias


# ============================================================
# PyTorch Dataset (BPR 三元组采样)
# ============================================================
class BPRDataset(Dataset):
    """BPR 排序损失: 每行 = (用户, 正样本, 负样本) 三元组"""
    def __init__(self, df, user_cols, item_cols, ctx_cols, uid_col='customer_id'):
        self.user_feat = torch.FloatTensor(df[user_cols].values)
        self.item_feat = torch.FloatTensor(df[item_cols].values)
        self.ctx_feat = torch.FloatTensor(df[ctx_cols].values)
        self.uids = df[uid_col].values

        # 构建 用户 → 正样本索引 映射
        self.pos_idx = defaultdict(list)
        for idx, (uid, label) in enumerate(zip(self.uids, df["label"].values)):
            if label == 1:
                self.pos_idx[uid].append(idx)

        # 所有负样本索引
        self.neg_idx_all = np.where(df["label"].values == 0)[0]

        # 生成 (uid, pos_idx) 对
        self.pairs = []
        for uid, idxs in self.pos_idx.items():
            for p_idx in idxs:
                self.pairs.append((uid, p_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, pos_idx = self.pairs[idx]
        neg_idx = np.random.choice(self.neg_idx_all)
        return (self.user_feat[pos_idx], self.item_feat[pos_idx], self.ctx_feat[pos_idx],
                self.user_feat[neg_idx], self.item_feat[neg_idx], self.ctx_feat[neg_idx])


# ============================================================
# 通用函数 (复用 item2vec 的 build_ltr_data)
# ============================================================
def build_ltr_data(candidates, labels, cus_feat_df, art_feat_df,
                   inter_feat_df, item_sim, user_hist, art2v,
                   default_vec, v2v_col='v2v_sim'):
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

    # === 计算 v2v_sim (余弦相似度) ===
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

    # === 额外: 用户塔需要用户历史 a2v 均值(而非候选的) ===
    # 在 build_ltr_data 中, user_emb_cache 已经计算了每用户的历史 a2v 均值
    # 但 USER_TOWER_COLS 的 a2v_0..a2v_31 是候选商品的嵌入,
    # 所以需要额外把用户历史 a2v 均值拼进去
    for i in range(32):
        col = f"a2v_{i}"
        if col not in df.columns:
            df[col] = np.float32(0)

    # 将用户历史 a2v 均值写入单独列 (用户塔用)
    hist_a2v_cols = [f"hist_a2v_{i}" for i in range(32)]
    user_hist_emb = np.array([
        user_emb_cache.get(c, default_vec) for c in df["customer_id"].values
    ], dtype=np.float32)
    for i in range(32):
        df[hist_a2v_cols[i]] = user_hist_emb[:, i]

    return df


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


def prepare_twotower_features(df):
    """从 LTR DataFrame 中提取双塔特征"""
    # 用户塔: 用 hist_a2v 代替 a2v
    hist_a2v_cols = [f"hist_a2v_{i}" for i in range(32)]
    user_feat_cols = CUS_COLS_CLEAN + hist_a2v_cols
    for c in user_feat_cols:
        if c not in df.columns:
            df[c] = np.float32(0)
        elif df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)

    # 商品塔: 商品特征 + 商品 a2v
    for c in ITEM_TOWER_COLS:
        if c not in df.columns:
            df[c] = np.float32(0)
        elif df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)

    # 上下文特征
    for c in CTX_TOWER_COLS:
        if c not in df.columns:
            df[c] = np.float32(0)
        elif df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)

    return df


def train_epoch_bpr(model, loader, optimizer, device):
    """BPR 排序损失训练"""
    model.train()
    total_loss = 0
    for batch in loader:
        u_pos, i_pos, c_pos, u_neg, i_neg, c_neg = [x.to(device) for x in batch]

        optimizer.zero_grad()
        pos_scores = model(u_pos, i_pos, c_pos).squeeze()
        neg_scores = model(u_neg, i_neg, c_neg).squeeze()
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_holdout(model, df, user_cols, item_cols, ctx_cols, holdout_gt, holdout_cids, device):
    model.eval()
    df = prepare_twotower_features(df)
    # 只评估 holdout 用户, 缩小数据量
    holdout_set = set(holdout_cids)
    df = df[df["customer_id"].isin(holdout_set)].copy()
    if len(df) == 0:
        return 0.0

    user_feat = torch.FloatTensor(df[user_cols].values).to(device)
    item_feat = torch.FloatTensor(df[item_cols].values).to(device)
    ctx_feat = torch.FloatTensor(df[ctx_cols].values).to(device)
    scores = model(user_feat, item_feat, ctx_feat).cpu().numpy().flatten()
    del user_feat, item_feat, ctx_feat

    df["score"] = scores
    preds = (
        df.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals = [holdout_gt[c] for c in holdout_cids]
    preds_l = [preds.get(c, []) for c in holdout_cids]
    return mapk(actuals, preds_l, k=12)


# ============================================================
# 读取基础数据
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

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")
print(f"  用户历史数量: {len(user_hist):,}")

# 加载全量用户历史 (用于 Word2Vec)
with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
    user_hist_full_data = pickle.load(f)
print(f"  全量用户历史: {len(user_hist_full_data):,}")

# ============================================================
# Phase 0: 训练 Word2Vec
# ============================================================
print("\n" + "=" * 60)
print("Phase 0: 训练 Word2Vec 商品嵌入 (32维, 全量数据)")
print("=" * 60)

w2v_model_path = f"{OUTPUT_DIR}/word2vec_article.model"
if os.path.exists(w2v_model_path):
    with timer("加载已有 Word2Vec"):
        w2v_model = Word2Vec.load(w2v_model_path)
    print(f"  词汇表大小: {len(w2v_model.wv):,}")
else:
    with timer("Phase 0 Word2Vec 训练 (全量)"):
        sequences = list(user_hist_full_data.values())
        total_seqs = len(sequences)
        total_articles = sum(len(s) for s in sequences)
        print(f"  序列数: {total_seqs:,}  |  总购买事件: {total_articles:,}")
        w2v_model = Word2Vec(
            sentences=sequences,
            vector_size=32, window=5, min_count=3,
            sg=1, epochs=20, workers=4, seed=42,
        )
        print(f"  词汇表大小: {len(w2v_model.wv):,}")
    w2v_model.save(w2v_model_path)

art2v = {aid: w2v_model.wv[aid] for aid in art_feat["article_id"]
         if aid in w2v_model.wv}
default_vec = np.zeros(32, dtype=np.float32)
n_known = len(art2v)
n_total = len(art_feat)
print(f"  商品嵌入覆盖率: {n_known}/{n_total} ({n_known/n_total*100:.1f}%)")

# ============================================================
# Phase 0.5: 将 a2v 嵌入追加到 art_feat
# ============================================================
with timer("追加 a2v 嵌入到 art_feat"):
    for i in range(32):
        col = f"a2v_{i}"
        art_feat[col] = art_feat["article_id"].map(
            lambda x: art2v.get(x, default_vec)[i]
        ).astype(np.float32)

# ============================================================
# 时间切分
# ============================================================
val_max_date = val_txn["t_dat"].max()
holdout_start = val_max_date - timedelta(days=1)

val_train_txn = val_txn.loc[val_txn["t_dat"] < holdout_start].copy()
val_holdout_txn = val_txn.loc[val_txn["t_dat"] >= holdout_start].copy()

print(f"\n  时间划分: val_train={len(val_train_txn):,}  |  val_holdout={len(val_holdout_txn):,}")

train_users = set(val_train_txn["customer_id"].unique())
holdout_users = set(val_holdout_txn["customer_id"].unique())
common_users = sorted(train_users & holdout_users)
print(f"  共同用户: {len(common_users):,}")

val_train_gt = val_train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_train_labels = {cid: set(aids) for cid, aids in val_train_gt.items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

# ============================================================
# Phase 1: 构建 LTR 数据 + 双塔训练
# ============================================================
print("\n" + "=" * 60)
print("Phase 1: 双塔神经网络训练")
print("=" * 60)

print("\n[构建数据]")
with timer("构建 LTR 数据"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist, art2v, default_vec)
    ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                                  inter_feat, item_sim, user_hist, art2v, default_vec)

ltr_train = prepare_twotower_features(ltr_train)
ltr_holdout = prepare_twotower_features(ltr_holdout)

hist_a2v_cols = [f"hist_a2v_{i}" for i in range(32)]
USER_TOWER_FEAT = CUS_COLS_CLEAN + hist_a2v_cols

train_dataset = BPRDataset(ltr_train, USER_TOWER_FEAT, ITEM_TOWER_COLS, CTX_TOWER_COLS)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

# ============================================================
# 初始化模型
# ============================================================
model = TwoTower(
    user_dim=len(USER_TOWER_FEAT),
    item_dim=len(ITEM_TOWER_COLS),
    ctx_dim=len(CTX_TOWER_COLS),
    emb_dim=16,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

print(f"\n  用户塔特征: {len(USER_TOWER_FEAT)}维")
print(f"  商品塔特征: {len(ITEM_TOWER_COLS)}维")
print(f"  上下文特征: {len(CTX_TOWER_COLS)}维")
print(f"  嵌入维度: 16")
print(f"  训练样本: {len(train_dataset):,}")
print(f"  验证样本: {len(ltr_holdout):,}")

# ============================================================
# 训练循环
# ============================================================
holdout_cids = [c for c in common_users if c in candidates]
best_map = 0.0
best_epoch = 0
patience = 5
no_improve = 0
n_epochs = 30

print(f"\n{'='*60}")
print(f"训练开始 (最多 {n_epochs} epochs, early_stop={patience})")
print(f"{'='*60}")

for epoch in range(1, n_epochs + 1):
    t0 = time.time()
    loss = train_epoch_bpr(model, train_loader, optimizer, device)
    gc.collect()
    torch.cuda.empty_cache()
    holdout_map = eval_holdout(model, ltr_holdout, USER_TOWER_FEAT, ITEM_TOWER_COLS,
                                CTX_TOWER_COLS, val_holdout_gt, holdout_cids, device)
    scheduler.step(holdout_map)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0

    print(f"  Epoch {epoch:2d}/{n_epochs} | loss={loss:.4f} | Holdout MAP@12={holdout_map:.5f} | lr={lr_now:.2e} | {elapsed:.0f}s")

    if holdout_map > best_map:
        best_map = holdout_map
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_twotower.pt")
        print(f"    → 新最佳, 模型已保存")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

# 加载最佳模型
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model_twotower.pt", map_location=device))
print(f"\n  最佳 Holdout MAP@12: {best_map:.5f}")

# ============================================================
# Phase 1 评估
# ============================================================
score_holdout = eval_holdout(model, ltr_holdout, USER_TOWER_FEAT, ITEM_TOWER_COLS,
                              CTX_TOWER_COLS, val_holdout_gt, holdout_cids, device)
score_train = eval_holdout(model, ltr_train, USER_TOWER_FEAT, ITEM_TOWER_COLS,
                            CTX_TOWER_COLS, val_train_gt, holdout_cids, device)

# 流行度基线
art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
score_pop_train = mapk([val_train_gt[c] for c in holdout_cids], [pop12] * len(holdout_cids), k=12)
score_pop_ho = mapk([val_holdout_gt[c] for c in holdout_cids], [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*60}")
print(f"  Phase 1 Val 评估 (Two-Tower):")
print(f"    ┌──────────────────────┬──────────┬──────────┐")
print(f"    │                      │  Train   │ Holdout  │")
print(f"    ├──────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline       │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ Two-Tower NN         │ {score_train:.5f}  │ {score_holdout:.5f}  │")
print(f"    │ 提升                  │ +{score_train-score_pop_train:.5f}  │ +{score_holdout-score_pop_ho:.5f}  │")
print(f"    └──────────────────────┴──────────┴──────────┘")
print(f"    最佳 Holdout MAP@12:  {best_map:.5f}")
print(f"{'='*60}")

# 释放 Phase 1 显存
del model, train_dataset, train_loader
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Phase 2: 全量 val 数据训练最终模型
# ============================================================
print(f"\n{'='*60}")
print("Phase 2: 全量训练 (val全部7天)")
print(f"{'='*60}")

val_all_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_all_labels = {cid: set(aids) for cid, aids in val_all_gt.items()}
all_val_users = sorted(set(candidates.keys()) | set(val_all_labels.keys()))
all_val_users_in_cands = [u for u in all_val_users if u in candidates]
print(f"  全量训练用户: {len(all_val_users_in_cands):,}")

with timer("构建全量 LTR 数据"):
    ltr_full = build_ltr_data(
        {u: candidates[u] for u in all_val_users_in_cands},
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist,
        art2v, default_vec,
    )
ltr_full = prepare_twotower_features(ltr_full)

full_dataset = BPRDataset(ltr_full, USER_TOWER_FEAT, ITEM_TOWER_COLS, CTX_TOWER_COLS)
full_loader = DataLoader(full_dataset, batch_size=1024, shuffle=True, num_workers=0)

# 重训模型 (用全量数据)
final_model = TwoTower(
    user_dim=len(USER_TOWER_FEAT),
    item_dim=len(ITEM_TOWER_COLS),
    ctx_dim=len(CTX_TOWER_COLS),
    emb_dim=16,
).to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"  全量训练样本: {len(full_dataset):,}")
print(f"  Phase 2 训练 (固定 {best_epoch} epochs)...")

# 用 Phase 1 最佳 epoch 数训练
for epoch in range(1, best_epoch + 1):
    loss = train_epoch_bpr(final_model, full_loader, final_optimizer, device)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}/{best_epoch} | loss={loss:.4f}")

torch.save(final_model.state_dict(), f"{OUTPUT_DIR}/model_twotower_final.pt")
print(f"\n模型已保存: {OUTPUT_DIR}/model_twotower_final.pt")

# 释放 Phase 2 显存
del final_model, ltr_full, full_dataset, full_loader
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Phase 3: 全量推理 + 提交
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: 全量推理")
print("=" * 60)

del cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist, candidates, ltr_train, ltr_holdout
gc.collect()

cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")
with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
    item_sim_full = pickle.load(f)
with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
    user_hist_full = pickle.load(f)

print("\n  追加 a2v 嵌入到全量商品特征...")
for i in range(32):
    col = f"a2v_{i}"
    art_feat_full[col] = art_feat_full["article_id"].map(
        lambda x: art2v.get(x, default_vec)[i]
    ).astype(np.float32)

art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()

known_users = set(user_hist_full.keys())
sub_cold = [c for c in sub_cids if c not in known_users]
print(f"  提交用户: {len(sub_cids):,}  冷启动: {len(sub_cold):,} ({len(sub_cold)/len(sub_cids)*100:.1f}%)")

# 重新加载模型
final_model = TwoTower(
    user_dim=len(USER_TOWER_FEAT),
    item_dim=len(ITEM_TOWER_COLS),
    ctx_dim=len(CTX_TOWER_COLS),
    emb_dim=16,
).to(device)
final_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model_twotower_final.pt", map_location=device))
final_model.eval()
all_preds = {}

with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(user_hist_full, item_sim_full, art_pop_full, batch_cids, w2v_model=w2v_model)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full,
                                art2v, default_vec)
        inf_df = prepare_twotower_features(inf_df)

        user_feat = torch.FloatTensor(inf_df[USER_TOWER_FEAT].values).to(device)
        item_feat = torch.FloatTensor(inf_df[ITEM_TOWER_COLS].values).to(device)
        ctx_feat = torch.FloatTensor(inf_df[CTX_TOWER_COLS].values).to(device)

        with torch.no_grad():
            scores = final_model(user_feat, item_feat, ctx_feat).cpu().numpy().flatten()

        inf_df["score"] = scores
        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_twotower.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_twotower.csv  ({len(sub):,} 行)")

print(f"\n{'='*60}")
print(f"结果汇总: Two-Tower 双塔排序模型")
print(f"{'='*60}")
print(f"  用户塔:                            {len(USER_TOWER_FEAT)}维 (用户特征4 + hist_a2v均值32)")
print(f"  商品塔:                            {len(ITEM_TOWER_COLS)}维 (商品特征11 + a2v32)")
print(f"  上下文:                            {len(CTX_TOWER_COLS)}维 (交互3 + 候选2 + v2v_sim1)")
print(f"  嵌入维度:                          16")
print(f"  Phase 1 Holdout MAP@12:            {best_map:.5f}")
print(f"  Word2Vec 词表大小:                  {len(w2v_model.wv):,}")
print(f"  商品嵌入覆盖率:                     {n_known}/{n_total} ({n_known/n_total*100:.1f}%)")
print(f"  冷启动比例:                         {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*60}")
print(f"\n完成!")
