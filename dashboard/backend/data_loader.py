from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Literal

import polars as pl

from cache_utils import load_articles, load_customers, load_transactions

TxnType = Literal["train", "val"]


@dataclass(frozen=True)
class KPIOverview:
    """KPI 概览数据容器：总交易、总金额、独立用户/商品数"""
    txn_type: TxnType
    total_transactions: int
    total_amount: float
    unique_customers: int
    unique_articles: int


def _txn_base(txn_type: TxnType) -> pl.DataFrame:
    """读取交易数据并做基础清洗：去空值、日期解析、金额填充"""
    df = load_transactions(txn_type)
    df = df.drop_nulls(["t_dat", "customer_id", "article_id"])

    if df.schema["t_dat"] == pl.Utf8:
        df = df.with_columns(pl.col("t_dat").str.to_datetime("%Y-%m-%d", strict=False))

    if "price" in df.columns:
        df = df.with_columns(pl.col("price").cast(pl.Float64).fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("price"))

    return df


@lru_cache(maxsize=16)
def get_txn_date_range(txn_type: TxnType) -> tuple[date, date]:
    """获取指定交易集的最小/最大日期，用于日期筛选控件默认值"""
    df = _txn_base(txn_type)
    mm = df.select(
        pl.col("t_dat").dt.date().min().alias("min_day"),
        pl.col("t_dat").dt.date().max().alias("max_day"),
    ).to_dicts()[0]
    return mm["min_day"], mm["max_day"]


@lru_cache(maxsize=32)
def kpi_overview(txn_type: TxnType) -> KPIOverview:
    """计算四个大盘 KPI 指标：总交易笔数、总金额、独立用户数、独立商品数"""
    df = _txn_base(txn_type)
    total_transactions = int(df.height)
    agg = df.select(
        pl.col("price").sum().alias("total_amount"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
        pl.col("article_id").n_unique().alias("unique_articles"),
    ).to_dicts()[0]

    return KPIOverview(
        txn_type=txn_type,
        total_transactions=total_transactions,
        total_amount=float(agg["total_amount"] or 0.0),
        unique_customers=int(agg["unique_customers"] or 0),
        unique_articles=int(agg["unique_articles"] or 0),
    )


def daily_sales(
    txn_type: TxnType,
    start_day: date | None,
    end_day: date | None,
) -> dict[str, Any]:
    """按天聚合销售额和订单量，返回时序数据（趋势图 + 聚合明细表使用）"""
    min_day, max_day = get_txn_date_range(txn_type)
    sd = start_day or min_day
    ed = end_day or max_day
    if sd > ed:
        raise ValueError("start_day 不能晚于 end_day")

    df = _txn_base(txn_type).with_columns(pl.col("t_dat").dt.date().alias("day"))
    df = df.filter(pl.col("day").is_between(sd, ed, closed="both"))

    out = (
        df.group_by("day")
        .agg(
            pl.col("price").sum().alias("sales_amount"),
            pl.len().alias("orders"),
        )
        .sort("day")
    )

    return {
        "txn_type": txn_type,
        "start_day": sd.isoformat(),
        "end_day": ed.isoformat(),
        "series": [
            {
                "date": r["day"].isoformat(),
                "sales_amount": float(r["sales_amount"] or 0.0),
                "orders": int(r["orders"] or 0),
            }
            for r in out.to_dicts()
        ],
    }


def _age_bin_expr(age_col: pl.Expr) -> pl.Expr:
    """将年龄值映射到分段标签（用于年龄分布柱状图）"""
    return (
        pl.when(age_col.is_null() | (age_col < 0))
        .then(pl.lit("未知"))
        .when(age_col <= 17)
        .then(pl.lit("0-17"))
        .when(age_col <= 24)
        .then(pl.lit("18-24"))
        .when(age_col <= 34)
        .then(pl.lit("25-34"))
        .when(age_col <= 44)
        .then(pl.lit("35-44"))
        .when(age_col <= 54)
        .then(pl.lit("45-54"))
        .when(age_col <= 64)
        .then(pl.lit("55-64"))
        .otherwise(pl.lit("65+"))
    )


@lru_cache(maxsize=32)
def user_age_dist(txn_type: TxnType) -> dict[str, Any]:
    """统计交易用户年龄分段人数分布"""
    txn_users = _txn_base(txn_type).select("customer_id").unique()
    customers = load_customers()

    if "age" not in customers.columns:
        raise ValueError("customers.parquet 缺少字段: age")

    cus = customers.select(
        pl.col("customer_id"),
        pl.col("age").cast(pl.Int32, strict=False),
    )

    joined = txn_users.join(cus, on="customer_id", how="left").with_columns(
        _age_bin_expr(pl.col("age")).alias("age_bin")
    )

    agg = (
        joined.group_by("age_bin")
        .agg(pl.col("customer_id").n_unique().alias("users"))
    )

    order = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "未知"]
    records = {r["age_bin"]: int(r["users"] or 0) for r in agg.to_dicts()}

    return {
        "txn_type": txn_type,
        "bins": [{"age_bin": k, "users": records.get(k, 0)} for k in order],
    }


def _category_col(articles: pl.DataFrame) -> str:
    """自动探测商品分类字段（按优先级选择一个存在的列名）"""
    candidates = [
        "product_group_name",
        "index_group_name",
        "department_name",
        "section_name",
        "garment_group_name",
        "product_type_name",
    ]
    for c in candidates:
        if c in articles.columns:
            return c
    raise ValueError("articles.parquet 缺少可用的商品大类字段")


# 商品大类英文→中文映射表
CATEGORY_CN_MAP = {
    "Garment Upper body": "上装",
    "Garment Lower body": "下装",
    "Garment Full body": "全身装",
    "Swimwear": "泳装",
    "Underwear": "内衣",
    "Accessories": "配饰",
    "Shoes": "鞋类",
    "Socks & Tights": "袜类",
    "Nightwear": "睡衣",
    "Unknown": "其他",
    "未知": "其他",
}


def _translate_category(cat: str) -> str:
    """将英文品类名转为中文，未匹配的保持原名"""
    return CATEGORY_CN_MAP.get(cat, cat)


@lru_cache(maxsize=64)
def goods_top_category(txn_type: TxnType, top_n: int = 10) -> dict[str, Any]:
    """统计商品大类销量 TopN（返回中文品类名称）"""
    if top_n <= 0 or top_n > 100:
        raise ValueError("top_n 必须在 1~100 之间")

    txn = _txn_base(txn_type).select(["article_id"])
    articles = load_articles()
    cat_col = _category_col(articles)
    art = articles.select(
        pl.col("article_id"),
        pl.col(cat_col).cast(pl.Utf8, strict=False).fill_null("未知").alias("category"),
    )

    joined = txn.join(art, on="article_id", how="left").with_columns(
        pl.col("category").fill_null("未知")
    )

    agg = (
        joined.group_by("category")
        .agg(pl.len().alias("qty"))
        .sort("qty", descending=True)
        .head(top_n)
    )

    return {
        "txn_type": txn_type,
        "top_n": top_n,
        "items": [
            {"category": _translate_category(r["category"]), "qty": int(r["qty"] or 0)}
            for r in agg.to_dicts()
        ],
    }


def compare_train_val() -> dict[str, Any]:
    """汇总训练集 vs 验证集的四个核心 KPI 指标，供对比柱状图使用"""
    train = kpi_overview("train")
    val = kpi_overview("val")
    return {
        "train": {
            "total_transactions": train.total_transactions,
            "total_amount": train.total_amount,
            "unique_customers": train.unique_customers,
            "unique_articles": train.unique_articles,
        },
        "val": {
            "total_transactions": val.total_transactions,
            "total_amount": val.total_amount,
            "unique_customers": val.unique_customers,
            "unique_articles": val.unique_articles,
        },
    }