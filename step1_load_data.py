"""
Step 1: 数据加载与验证集划分
============================
Polars 读取 CSV → 按时间顺序划分 train/val → 保存 Parquet

输入: DATA_DIR/transactions_train.csv, articles.csv, customers.csv
输出: PROCESSED_DIR/train_txn.parquet, val_txn.parquet, articles.parquet, customers.parquet
"""

import polars as pl
import pandas as pd
import numpy as np
import gc
from datetime import timedelta

from config import DATA_DIR, PROCESSED_DIR, VAL_DAYS, SEED
from utils import timer, set_seed

set_seed(SEED)

print("=" * 60)
print("Step 1: 数据加载与验证集划分")
print("=" * 60)

with timer("CSV 读取"):
    # 读取交易数据 (指定 dtypes 加速)
    txn_pl = pl.read_csv(
        f"{DATA_DIR}/transactions_train.csv",
        schema_overrides={"customer_id": pl.Utf8, "article_id": pl.Utf8, "price": pl.Float32},
    )
    txn_pl = txn_pl.with_columns(pl.col("t_dat").str.to_date("%Y-%m-%d"))
    print(f"  交易记录: {txn_pl.height:,} 行")

    # 读取商品和客户
    art_pl = pl.read_csv(f"{DATA_DIR}/articles.csv", schema_overrides={"article_id": pl.Utf8})
    cus_pl = pl.read_csv(f"{DATA_DIR}/customers.csv", schema_overrides={"customer_id": pl.Utf8})

    # 转为 pandas (后续 sklearn/lgb 需要)
    txn = txn_pl.to_pandas()
    txn["t_dat"] = pd.to_datetime(txn["t_dat"])
    del txn_pl
    gc.collect()

    art = art_pl.to_pandas()
    del art_pl
    gc.collect()

    cus = cus_pl.to_pandas()
    del cus_pl
    gc.collect()

    txn["price"] = txn["price"].astype(np.float32)
    print(f"  商品: {len(art):,}  |  客户: {len(cus):,}")

# 按时间顺序切分: 验证集 = 最后 VAL_DAYS 天
with timer("时间顺序切分"):
    max_date = txn["t_dat"].max()
    val_start = max_date - timedelta(days=VAL_DAYS - 1)

    train_txn = txn.loc[txn["t_dat"] < val_start].copy()
    val_txn = txn.loc[txn["t_dat"] >= val_start].copy()

    print(f"  Train: {len(train_txn):,} 行  "
          f"({train_txn['t_dat'].min().date()} ~ {train_txn['t_dat'].max().date()})")
    print(f"  Val:   {len(val_txn):,} 行  "
          f"({val_txn['t_dat'].min().date()} ~ {val_txn['t_dat'].max().date()})")

    # 验证无重叠
    assert train_txn["t_dat"].max() < val_txn["t_dat"].min(), \
        "错误: train 和 val 日期范围有重叠!"

# 保存为 Parquet
with timer("保存 Parquet"):
    train_txn.to_parquet(f"{PROCESSED_DIR}/train_txn.parquet", index=False)
    val_txn.to_parquet(f"{PROCESSED_DIR}/val_txn.parquet", index=False)
    art.to_parquet(f"{PROCESSED_DIR}/articles.parquet", index=False)
    cus.to_parquet(f"{PROCESSED_DIR}/customers.parquet", index=False)

print(f"\n已保存到 {PROCESSED_DIR}/")
print("  train_txn.parquet  val_txn.parquet  articles.parquet  customers.parquet")
print("\nStep 1 完成!")
