"""
召回通道消融实验 (Recall Ablation Study)
========================================
前提: 需先运行 F:/HM-recommendation 的 step1 → step2 → step6

每个实验: 本地 CV 评估 (Val MAP@12) + 全量推理 → 保存提交文件

实验设计 (做加法, 看增量):
  仅流行度 → +历史 → +CF → +历史+CF

使用方法:
  cd F:/HM-ablation-study
  python ablation_recall.py
"""

import sys, os

# 引用 F:/HM-recommendation 项目中的 config / utils
PROJECT_DIR = "F:/HM-recommendation"
sys.path.insert(0, PROJECT_DIR)

import pandas as pd
import numpy as np
import pickle, gc, time, warnings
from collections import defaultdict

import lightgbm as lgb

from config import (
    DATA_DIR, PROCESSED_DIR,
    LGB_PARAMS, CUS_COLS, ART_COLS, INTER_COLS, FEAT_COLS,
    SEED,
)
from utils import mapk, set_seed

set_seed(SEED)
warnings.filterwarnings("ignore")
print(f"LightGBM {lgb.__version__}  SEED={SEED}")

# 输出目录
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 0. 读取中间文件 (step2 + step6 产出)
# ============================================================
print("=" * 60)
print("读取中间文件")
print("=" * 60)

t0 = time.time()

# Val 评估用 (step2)
val_txn = pd.read_parquet(f"{PROCESSED_DIR}/val_txn.parquet")
val_txn["t_dat"] = pd.to_datetime(val_txn["t_dat"])

cus_feat = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat.parquet")
art_feat = pd.read_parquet(f"{PROCESSED_DIR}/art_feat.parquet")
inter_feat = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat.parquet")

with open(f"{PROCESSED_DIR}/item_sim.pkl", "rb") as f:
    item_sim = pickle.load(f)
with open(f"{PROCESSED_DIR}/user_hist.pkl", "rb") as f:
    user_hist = pickle.load(f)

print(f"  Val: {len(val_txn):,} 行  |  cus: {cus_feat.shape}  art: {art_feat.shape}  inter: {inter_feat.shape}")

# 全量推理用 (step6)
cus_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/cus_feat_full.parquet")
art_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/art_feat_full.parquet")
inter_feat_full = pd.read_parquet(f"{PROCESSED_DIR}/inter_feat_full.parquet")

with open(f"{PROCESSED_DIR}/item_sim_full.pkl", "rb") as f:
    item_sim_full = pickle.load(f)
with open(f"{PROCESSED_DIR}/user_hist_full.pkl", "rb") as f:
    user_hist_full = pickle.load(f)

print(f"  Full: cus: {cus_feat_full.shape}  art: {art_feat_full.shape}  inter: {inter_feat_full.shape}")

art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
art_pop_full = art_feat_full.set_index("article_id")["popularity_score"].to_dict()
pop12_full = sorted(art_pop_full, key=lambda x: -art_pop_full[x])[:12]

sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub_cids = sub["customer_id"].tolist()
print(f"[读取] {time.time()-t0:.1f}s  |  提交客户: {len(sub_cids):,}")

# ============================================================
# 1. 验证集
# ============================================================
val_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
val_cids = list(val_gt.keys())
val_labels = {cid: set(aids) for cid, aids in val_gt.items()}
print(f"  验证集用户: {len(val_cids):,}")

# ============================================================
# 2. 候选生成
# ============================================================
def generate_candidates(user_hist, item_sim, art_pop, customers,
                        use_history=True, use_cf=True, use_pop=True,
                        n_hist=12, n_pop=12):
    pop_list = sorted(art_pop, key=lambda x: -art_pop[x])[:n_pop]
    out = {}
    for cid in customers:
        cands = set()
        if use_history:
            for aid in user_hist.get(cid, [])[:n_hist]:
                cands.add(aid)
        if use_cf:
            for aid in user_hist.get(cid, [])[:5]:
                if aid in item_sim:
                    for rel, _ in item_sim[aid][:10]:
                        cands.add(rel)
        if use_pop:
            for aid in pop_list:
                cands.add(aid)
        out[cid] = list(cands)
    return out


# ============================================================
# 3. LTR 数据构建
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

    c_arr = df["customer_id"].values; a_arr = df["article_id"].values
    df["cf_score"] = np.float32([cf_map.get(c, {}).get(a, 0.0) for c, a in zip(c_arr, a_arr)])
    df["price_match"] = (-np.abs(df["avg_price"].values - df["avg_price_user"].values)).astype(np.float32)
    del c_arr, a_arr, cf_map; gc.collect()

    for c in FEAT_COLS:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype(np.float32)
    return df


# ============================================================
# 4. 训练 + 评估
# ============================================================
def train_model(ltr_df, feat_cols, params):
    groups = ltr_df.groupby("customer_id").size().values
    ds = lgb.Dataset(ltr_df[feat_cols].values, label=ltr_df["label"].values,
                     group=groups, feature_name=feat_cols)
    try:
        model = lgb.train(params, ds, num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    except Exception:
        cpu_params = {**params, "device": "cpu"}
        for k in ["gpu_platform_id", "gpu_device_id"]:
            cpu_params.pop(k, None)
        model = lgb.train(cpu_params, ds, num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    return model


def eval_mapk(model, ltr_df, feat_cols, val_cids, val_gt):
    ltr_df["score"] = model.predict(ltr_df[feat_cols].values)
    preds = (ltr_df.sort_values(["customer_id", "score"], ascending=[True, False])
             .groupby("customer_id").head(12)
             .groupby("customer_id")["article_id"].apply(list).to_dict())
    actuals = [val_gt[c] for c in val_cids]
    return mapk(actuals, [preds.get(c, []) for c in val_cids], k=12)


# ============================================================
# 5. 全量推理
# ============================================================
INFER_BATCH = 50000

def full_inference_and_save(model, sub_cids, use_hist, use_cf, use_pop, output_path):
    t0 = time.time()
    all_preds = {}
    for start in range(0, len(sub_cids), INFER_BATCH):
        batch_cids = sub_cids[start:start + INFER_BATCH]
        cands_batch = generate_candidates(user_hist_full, item_sim_full, art_pop_full,
                                          batch_cids, use_history=use_hist, use_cf=use_cf, use_pop=use_pop)
        inf_df = build_ltr_data(cands_batch, {}, cus_feat_full, art_feat_full,
                                inter_feat_full, item_sim_full, user_hist_full)
        inf_df["score"] = model.predict(inf_df[FEAT_COLS].values)
        batch_preds = (inf_df.sort_values(["customer_id", "score"], ascending=[True, False])
                       .groupby("customer_id").head(12)
                       .groupby("customer_id")["article_id"].apply(list).to_dict())
        all_preds.update(batch_preds)
        del cands_batch, inf_df, batch_preds; gc.collect()

    sub_out = pd.DataFrame({"customer_id": sub_cids})
    sub_out["prediction"] = sub_out["customer_id"].apply(
        lambda cid: " ".join(all_preds.get(cid, pop12_full)))
    sub_out.to_csv(output_path, index=False)
    print(f"  全量推理: {len(all_preds):,}/{len(sub_cids):,} 用户  ({time.time()-t0:.1f}s)  →  {output_path}")
    del all_preds; gc.collect()


# ============================================================
# 6. 流行度 Baseline
# ============================================================
print("\n" + "=" * 60)
print("流行度 Baseline")
print("=" * 60)

pop12_train = sorted(art_pop, key=lambda x: -art_pop[x])[:12]
actuals = [val_gt[c] for c in val_cids]
score_pop = mapk(actuals, [pop12_train] * len(val_cids), k=12)
print(f"  纯流行度 MAP@12: {score_pop:.5f}")

sub_pop = pd.DataFrame({"customer_id": sub_cids})
sub_pop["prediction"] = " ".join(pop12_full)
sub_pop.to_csv(os.path.join(OUT_DIR, "ablation_pop_baseline.csv"), index=False)
print(f"  全量提交: {OUT_DIR}/ablation_pop_baseline.csv")

# ============================================================
# 7. 召回消融实验
# ============================================================
print("\n" + "=" * 60)
print("召回通道消融实验")
print("=" * 60)

experiments = [
    ("仅流行度 (基准)",  False, False, True,  "ablation_pop"),
    ("+ 历史",          True,  False, True,  "ablation_hist"),
    ("+ CF",            False, True,  True,  "ablation_cf"),
    ("+ 历史+CF",       True,  True,  True,  "ablation_full"),
]

results = []
prev_score = None

for name, use_hist, use_cf, use_pop, file_tag in experiments:
    print(f"\n{'='*50}")
    print(f"[实验] {name}")
    print(f"{'='*50}")

    # Val 候选 + LTR
    val_cands = generate_candidates(user_hist, item_sim, art_pop, val_cids,
                                    use_history=use_hist, use_cf=use_cf, use_pop=use_pop)
    tot = sum(len(v) for v in val_cands.values())
    avg_cand = tot / len(val_cands) if val_cands else 0

    ltr_df = build_ltr_data(val_cands, val_labels, cus_feat, art_feat, inter_feat, item_sim, user_hist)
    pos = ltr_df["label"].sum()

    # 训练
    model = train_model(ltr_df, FEAT_COLS, LGB_PARAMS)
    score = eval_mapk(model, ltr_df, FEAT_COLS, val_cids, val_gt)
    delta = score - score_pop
    step_gain = score - prev_score if prev_score is not None else 0.0

    print(f"  Val 候选: {avg_cand:.1f}/用户 | LTR: {len(ltr_df):,} (pos={pos:,})")
    print(f"  Val MAP@12 = {score:.5f}  (Δ流行度={delta:+.5f}, 增量={step_gain:+.5f})")

    # 全量推理
    full_inference_and_save(model, sub_cids, use_hist, use_cf, use_pop,
                            os.path.join(OUT_DIR, f"{file_tag}.csv"))

    results.append((name, score, delta, avg_cand, len(ltr_df), step_gain, file_tag))
    prev_score = score
    del val_cands, ltr_df, model; gc.collect()

# ============================================================
# 8. 汇总
# ============================================================
print("\n\n" + "=" * 80)
print("召回通道消融实验 — 结果汇总")
print("=" * 80)
print(f"\n  {'实验组':<18s} {'Val MAP@12':>11s} {'Δ流行度':>9s} {'增量':>9s} {'候选/用户':>9s}")
print(f"  {'-'*18} {'-'*11} {'-'*9} {'-'*9} {'-'*9}")
print(f"  {'纯流行度 (无模型)':<18s} {score_pop:>11.5f} {'—':>9s} {'—':>9s} {'12.0':>9s}")
for name, score, delta, avg_cand, n_samples, step_gain, _ in results:
    print(f"  {name:<18s} {score:>11.5f} {delta:>+9.5f} {step_gain:>+9.5f} {avg_cand:>9.1f}")

# Markdown
print(f"\nMarkdown 表格:\n")
print(f"| 实验组 | Val MAP@12 | Δ流行度 | 增量 | 候选/用户 |")
print(f"|:---|---:|---:|---:|---:|")
print(f"| 纯流行度 (无模型) | {score_pop:.5f} | — | — | 12.0 |")
for name, score, delta, avg_cand, n_samples, step_gain, _ in results:
    print(f"| {name} | {score:.5f} | {delta:+.5f} | {step_gain:+.5f} | {avg_cand:.1f} |")

# 保存结果
results_path = os.path.join(OUT_DIR, "recall_ablation_results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("召回通道消融实验汇总\n\n")
    f.write(f"流行度 Baseline MAP@12: {score_pop:.5f}\n\n")
    f.write(f"{'实验组':<18s} {'Val MAP@12':>11s} {'Δ流行度':>9s} {'候选/用户':>9s}\n")
    f.write(f"{'-'*18} {'-'*11} {'-'*9} {'-'*9}\n")
    for name, score, delta, avg_cand, n_samples, step_gain, _ in results:
        f.write(f"{name:<18s} {score:>11.5f} {delta:>+9.5f} {avg_cand:>9.1f}\n")

print(f"\n结果保存: {results_path}")
print(f"\n✅ 完成!\n")
