from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import polars as pl


def get_hm_root() -> Path:
    """获取 HM 项目根目录：cache_utils.py → backend → dashboard → HM"""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """返回 data 目录路径（数据位于项目根目录的 data/ 子目录）"""
    return get_hm_root() / "data"


def _ensure_exists(path: Path) -> None:
    """校验文件是否存在，不存在则抛出 FileNotFoundError"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")


def _read_parquet(path: Path, columns: Iterable[str] | None = None) -> pl.DataFrame:
    """按需读取 parquet 文件，可选仅加载指定列以节省内存"""
    _ensure_exists(path)
    if columns is None:
        return pl.read_parquet(str(path))
    return pl.read_parquet(str(path), columns=list(columns))


@lru_cache(maxsize=8)
def load_articles() -> pl.DataFrame:
    """加载商品基础属性表（内存缓存，重复调用不重读磁盘）"""
    data_dir = get_data_dir()
    path = data_dir / "articles.parquet"
    df = _read_parquet(path)
    if "article_id" not in df.columns:
        raise ValueError("articles.parquet 缺少必要字段: article_id")
    return df


@lru_cache(maxsize=8)
def load_customers() -> pl.DataFrame:
    """加载用户画像基础表（内存缓存，重复调用不重读磁盘）"""
    data_dir = get_data_dir()
    path = data_dir / "customers.parquet"
    df = _read_parquet(path)
    if "customer_id" not in df.columns:
        raise ValueError("customers.parquet 缺少必要字段: customer_id")
    return df


@lru_cache(maxsize=8)
def load_transactions(txn_type: str) -> pl.DataFrame:
    """按类型加载交易数据并仅保留必要列（train→train_txn / val→val_txn），内存缓存"""
    if txn_type not in {"train", "val"}:
        raise ValueError("txn_type 必须为 train 或 val")

    data_dir = get_data_dir()
    filename = "train_txn.parquet" if txn_type == "train" else "val_txn.parquet"
    path = data_dir / filename

    df = _read_parquet(path)

    required = {"t_dat", "customer_id", "article_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} 缺少必要字段: {sorted(missing)}")

    # 仅保留核心列，避免加载不必要字段
    cols = set(df.columns)
    keep = ["t_dat", "customer_id", "article_id"]
    if "price" in cols:
        keep.append("price")

    df = df.select(keep)
    return df