"""
Step 5 Neural Network: MLP排序模型作为第四个异构模型
=====================================================
基于 step5_lowlr.py 模板, 复用所有数据管线, 使用 PyTorch MLP + ListNet loss

与树模型的异构性:
  - 模型类别: 神经网络 vs 决策树 → 误差模式完全不同
  - 损失函数: ListNet (softmax CE) vs LambdaRank/YetiRank/rank:ndcg
  - 决策边界: 平滑连续 vs 阶梯状
  - 需要特征标准化 (树模型不需要)

输出: output/model_nn.pt / output/scaler_nn.pkl / output/submission_nn.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, PROCESSED_DIR,
    VAL_DAYS, INFER_BATCH_SIZE,
)
from utils import timer, mapk

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
print("=" * 60)
print(f"Step 5 NN: 23维 MLP + ListNet loss  (device={device})")
print("=" * 60)

# ============================================================
# 特征列 (23维, 与 lowlr/catboost/xgboost 完全一致)
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

print(f"  特征: {len(FEAT_COLS_CLEAN)}维  epochs=150  lr=0.001")

from config import CUS_COLS, ART_COLS, INTER_COLS

# ============================================================
# PyTorch 模型 & 数据集
# ============================================================
class RankingMLP(nn.Module):
    """简单 MLP 排序模型: 23 -> 128 -> 64 -> 32 -> 1"""
    def __init__(self, input_dim=23, hidden_dims=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GroupDataset(Dataset):
    """每个 customer 作为一个样本, 包含其所有候选商品"""
    def __init__(self, X, y, groups):
        self.data = []
        start = 0
        for g in groups:
            g = int(g)
            self.data.append((X[start:start+g].copy(), y[start:start+g].copy()))
            start += g

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X_g, y_g = self.data[idx]
        return torch.tensor(X_g, dtype=torch.float32), torch.tensor(y_g, dtype=torch.float32)


def collate_groups(batch):
    """拼接 batch 内所有 group, 记录每个 group 的大小"""
    X_list, y_list = zip(*batch)
    X_batch = torch.cat(X_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    sizes = [len(y) for y in y_list]
    return X_batch, y_batch, sizes


def listnet_loss(scores, labels, group_sizes):
    """ListNet: softmax cross-entropy per query group"""
    total_loss = torch.tensor(0.0, device=scores.device)
    n_groups = 0
    start = 0
    eps = 1e-10

    for g_size in group_sizes:
        if g_size <= 1:
            start += g_size
            continue
        end = start + g_size
        g_scores = scores[start:end].view(-1)
        g_labels = labels[start:end]

        pos_sum = g_labels.sum()
        if pos_sum < 0.5:
            start = end
            continue

        # 稳定 softmax
        g_max = g_scores.max()
        g_exp = torch.exp(g_scores - g_max)
        g_probs = g_exp / (g_exp.sum() + eps)

        # 目标: 正样本均分概率
        target = g_labels / pos_sum

        loss = -(target * torch.log(g_probs + eps)).sum()
        total_loss = total_loss + loss
        n_groups += 1
        start = end

    return total_loss / max(n_groups, 1)


def evaluate_map12_nn(model, ltr_df, gt_dict, cids, scaler, device):
    """预测 + 计算 MAP@12"""
    model.eval()
    X = ltr_df[FEAT_COLS_CLEAN].values.astype(np.float32)
    X = scaler.transform(X)
    with torch.no_grad():
        scores = model(torch.tensor(X).to(device)).cpu().numpy().flatten()
    ltr_df = ltr_df.copy()
    ltr_df["score"] = scores
    preds = (
        ltr_df.sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id").head(12)
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    actuals = [gt_dict[c] for c in cids]
    preds_l = [preds.get(c, []) for c in cids]
    return mapk(actuals, preds_l, k=12)


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

    return df


def prepare_ltr(ltr_df):
    for c in FEAT_COLS_CLEAN:
        if ltr_df[c].dtype == "float64":
            ltr_df[c] = ltr_df[c].astype(np.float32)
    X = ltr_df[FEAT_COLS_CLEAN].values
    y = ltr_df["label"].values
    groups = ltr_df.groupby("customer_id").size().values
    return X, y, groups


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

print(f"  val_txn: {val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()}")

# ============================================================
# 时间切分: val → train(前5天) + holdout(后2天)
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

val_train_gt = val_train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_holdout_gt = val_holdout_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_train_labels = {cid: set(aids) for cid, aids in val_train_gt.items()}
val_holdout_labels = {cid: set(aids) for cid, aids in val_holdout_gt.items()}

# ============================================================
# Phase 1: 构建 LTR 数据 + 训练 + 评估
# ============================================================
print("\n[Train LTR]")
with timer("构建 LTR 数据"):
    ltr_train = build_ltr_data(candidates, val_train_labels, cus_feat, art_feat,
                               inter_feat, item_sim, user_hist)

print("\n[Holdout LTR]")
ltr_holdout = build_ltr_data(candidates, val_holdout_labels, cus_feat, art_feat,
                              inter_feat, item_sim, user_hist)

X_train, y_train, groups_train = prepare_ltr(ltr_train)
X_valid, y_valid, groups_valid = prepare_ltr(ltr_holdout)

# 特征标准化 (神经网络必须)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_valid = scaler.transform(X_valid).astype(np.float32)

print(f"\n  训练样本: {X_train.shape[0]:,}  |  验证样本: {X_valid.shape[0]:,}")
print(f"  训练 groups: {len(groups_train):,}  |  验证 groups: {len(groups_valid):,}")

# ============================================================
# 创建 DataLoader
# ============================================================
train_dataset = GroupDataset(X_train, y_train, groups_train)
valid_dataset = GroupDataset(X_valid, y_valid, groups_valid)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    collate_fn=collate_groups, num_workers=0, pin_memory=(device.type == "cuda"),
)
valid_loader = DataLoader(
    valid_dataset, batch_size=256, shuffle=False,
    collate_fn=collate_groups, num_workers=0, pin_memory=(device.type == "cuda"),
)

# ============================================================
# 模型 & 优化器
# ============================================================
model = RankingMLP(input_dim=len(FEAT_COLS_CLEAN)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6,
)

num_epochs = 150
patience = 25
best_map = -1.0
best_epoch = 0
best_state = None
no_improve = 0

holdout_cids = [c for c in common_users if c in candidates]
train_eval_cids = [c for c in common_users if c in candidates]

# ============================================================
# Phase 1 训练循环
# ============================================================
print(f"\nPhase 1 NN 训练: epochs={num_epochs}  patience={patience}")
print("-" * 55)

with timer("Phase 1 NN 训练"):
    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, sizes in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            scores = model(X_batch)
            loss = listnet_loss(scores, y_batch, sizes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- valid loss ----
        model.eval()
        valid_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for X_batch, y_batch, sizes in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                scores = model(X_batch)
                loss = listnet_loss(scores, y_batch, sizes)
                valid_loss += loss.item()
                n_vb += 1
        avg_valid_loss = valid_loss / max(n_vb, 1)

        # ---- evaluate MAP@12 ----
        score_holdout = evaluate_map12_nn(
            model, ltr_holdout, val_holdout_gt, holdout_cids, scaler, device,
        )

        scheduler.step(score_holdout)

        if score_holdout > best_map:
            best_map = score_holdout
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 1 or epoch <= 3:
            print(f"  epoch {epoch:3d}  |  train_loss={avg_train_loss:.4f}  valid_loss={avg_valid_loss:.4f}  "
                  f"holdout_MAP={score_holdout:.5f}  best={best_map:.5f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        if no_improve >= patience:
            print(f"\n  早停于 epoch {epoch} (patience={patience}), 最佳 epoch={best_epoch}")
            break

# 加载最佳模型
model.load_state_dict(best_state)
print(f"\n  最佳 epoch: {best_epoch}  最佳 Holdout MAP@12: {best_map:.5f}")

# 评估 train MAP@12
score_train_eval = evaluate_map12_nn(
    model, ltr_train, val_train_gt, train_eval_cids, scaler, device,
)

art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
actuals_tr = [val_train_gt[c] for c in train_eval_cids]
actuals_ho = [val_holdout_gt[c] for c in holdout_cids]
score_pop_train = mapk(actuals_tr, [pop12] * len(train_eval_cids), k=12)
score_pop_ho = mapk(actuals_ho, [pop12] * len(holdout_cids), k=12)

print(f"\n{'='*55}")
print(f"  Phase 1 Val 评估 (NN MLP + ListNet):")
print(f"    ┌────────────────────┬──────────┬──────────┐")
print(f"    │                    │  Train   │ Holdout  │")
print(f"    ├────────────────────┼──────────┼──────────┤")
print(f"    │ 流行度 Baseline     │ {score_pop_train:.5f}  │ {score_pop_ho:.5f}  │")
print(f"    │ 23维 NN MLP         │ {score_train_eval:.5f}  │ {best_map:.5f}  │")
print(f"    │ 提升                │ +{score_train_eval-score_pop_train:.5f}  │ +{best_map-score_pop_ho:.5f}  │")
print(f"    └────────────────────┴──────────┴──────────┘")
print(f"    过拟合程度: train-holdout = {score_train_eval-best_map:.5f}")
print(f"{'='*55}")

# ============================================================
# Phase 2: 全量 val 数据训练最终模型
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 2: 全量训练 (val全部7天, epochs={best_epoch})")
print(f"{'='*60}")

val_all_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_all_labels = {cid: set(aids) for cid, aids in val_all_gt.items()}
all_val_users = sorted(set(candidates.keys()) | set(val_all_labels.keys()))
all_val_users_in_cands = [u for u in all_val_users if u in candidates]
print(f"  全量训练用户: {len(all_val_users_in_cands):,}")

with timer("构建全量 LTR 数据"):
    ltr_full = build_ltr_data(
        {u: candidates[u] for u in all_val_users_in_cands},
        val_all_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist
    )
X_full, y_full, groups_full = prepare_ltr(ltr_full)

# 全量数据标准化
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X_full).astype(np.float32)

full_dataset = GroupDataset(X_full, y_full, groups_full)
full_loader = DataLoader(
    full_dataset, batch_size=128, shuffle=True,
    collate_fn=collate_groups, num_workers=0, pin_memory=(device.type == "cuda"),
)

# 重新初始化模型 + 训练 best_epoch 轮
final_model = RankingMLP(input_dim=len(FEAT_COLS_CLEAN)).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)

with timer(f"Phase 2 训练 (固定 {best_epoch} epochs)"):
    for epoch in range(1, best_epoch + 1):
        final_model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, sizes in full_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            final_optimizer.zero_grad()
            scores = final_model(X_batch)
            loss = listnet_loss(scores, y_batch, sizes)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 1 or epoch <= 2:
            print(f"  epoch {epoch:3d}/{best_epoch}  loss={total_loss/max(n_batches,1):.4f}")

# 保存模型 & scaler
torch.save(final_model.state_dict(), f"{OUTPUT_DIR}/model_nn.pt")
with open(f"{OUTPUT_DIR}/scaler_nn.pkl", "wb") as f:
    pickle.dump(scaler_full, f)
print(f"\n模型已保存: {OUTPUT_DIR}/model_nn.pt")
print(f"Scaler已保存: {OUTPUT_DIR}/scaler_nn.pkl")

del X_full, y_full, groups_full, train_dataset, valid_dataset, full_dataset, ltr_train, ltr_holdout
gc.collect()

# ============================================================
# Phase 3: 全量推理 + 提交
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: 全量推理")
print("=" * 60)

del cus_feat, art_feat, inter_feat, val_txn, item_sim, user_hist, candidates
gc.collect()

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

final_model.eval()
all_preds = {}
with timer(f"分批推理 (batch={INFER_BATCH_SIZE})"):
    for start in range(0, len(sub_cids), INFER_BATCH_SIZE):
        batch_cids = sub_cids[start:start + INFER_BATCH_SIZE]
        batch_num = start // INFER_BATCH_SIZE + 1

        batch_cands = generate_candidates(user_hist_full, item_sim_full, art_pop_full, batch_cids)
        inf_df = build_ltr_data(batch_cands, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        for c in FEAT_COLS_CLEAN:
            if inf_df[c].dtype == "float64":
                inf_df[c] = inf_df[c].astype(np.float32)

        X_inf = scaler_full.transform(inf_df[FEAT_COLS_CLEAN].values.astype(np.float32))
        with torch.no_grad():
            scores = final_model(torch.tensor(X_inf).to(device)).cpu().numpy().flatten()
        inf_df["score"] = scores

        batch_preds = (
            inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id").head(12)
            .groupby("customer_id")["article_id"].apply(list).to_dict()
        )
        all_preds.update(batch_preds)
        del inf_df, batch_cands, batch_preds, X_inf; gc.collect()
        print(f"  Batch {batch_num}: {len(batch_cids):,} 用户完成")

sub["prediction"] = sub["customer_id"].map(
    lambda x: " ".join(all_preds.get(x, pop12_full))
)
sub.to_csv(f"{OUTPUT_DIR}/submission_nn.csv", index=False)
print(f"\n提交已保存: {OUTPUT_DIR}/submission_nn.csv  ({len(sub):,} 行)")

print(f"\n{'='*55}")
print(f"结果汇总: NN MLP + ListNet")
print(f"{'='*55}")
print(f"  Phase 1 Holdout MAP@12:          {best_map:.5f}")
print(f"  Phase 1 Train MAP@12:            {score_train_eval:.5f}")
print(f"  过拟合:                           {score_train_eval-best_map:.5f}")
print(f"  最佳 epoch:                       {best_epoch}")
print(f"  冷启动比例:                       {len(sub_cold)/len(sub_cids)*100:.1f}%")
print(f"{'='*55}")
print(f"\n完成!")
