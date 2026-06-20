from __future__ import annotations

import sys
import random
from pathlib import Path
from datetime import date

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ABLATION_REPORT_PATH = PROJECT_ROOT / "step5_feature_ablation_report.md"

from data_loader import compare_train_val, daily_sales, goods_top_category, kpi_overview, user_age_dist

# 固定随机种子，保证重复运行结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==================== FastAPI 实例化 & CORS 跨域配置 ====================
app = FastAPI(title="HM 交互式数据看板 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源（本地 file:// 协议 + localhost 均放行）
    allow_credentials=False,
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
    expose_headers=["*"],      # 暴露所有响应头给前端
)


# ==================== 通用工具函数 ====================

def _ensure_txn_type(txn_type: str) -> str:
    """校验 txn_type 参数，仅允许 train 或 val"""
    if txn_type not in {"train", "val"}:
        raise HTTPException(status_code=400, detail="txn_type 必须为 train 或 val")
    return txn_type


def error_response(message: str, detail: str = ""):
    """统一错误返回格式：error 标记 + 可读 message + 技术 detail"""
    return {"error": True, "message": message, "detail": detail}


# ==================== 原有 5 个 API 接口（业务数据概览标签页） ====================

@app.get("/api/kpi/overview")
def api_kpi_overview(txn_type: str = Query("train")):
    """大盘 KPI 概览：总交易笔数、总消费金额、独立用户量、独立商品量"""
    try:
        t = _ensure_txn_type(txn_type)
        kpi = kpi_overview(t)
        return {
            "txn_type": kpi.txn_type,
            "total_transactions": kpi.total_transactions,
            "total_amount": kpi.total_amount,
            "unique_customers": kpi.unique_customers,
            "unique_articles": kpi.unique_articles,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/trend/daily_sales")
def api_trend_daily_sales(
    txn_type: str = Query("train"),
    start_day: date | None = Query(None),
    end_day: date | None = Query(None),
):
    """每日销售&订单量时序趋势：按日期聚合销售额和订单量，返回折线图数据"""
    try:
        t = _ensure_txn_type(txn_type)
        return daily_sales(t, start_day, end_day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/user/age_dist")
def api_user_age_dist(txn_type: str = Query("train")):
    """用户年龄分段人数分布柱状图数据"""
    try:
        t = _ensure_txn_type(txn_type)
        return user_age_dist(t)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/goods/top_category")
def api_goods_top_category(
    txn_type: str = Query("train"),
    top_n: int = Query(10, ge=1, le=100),
):
    """商品大类销量 TopN 横向条形图数据（返回中文品类名称）"""
    try:
        t = _ensure_txn_type(txn_type)
        return goods_top_category(t, top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/compare/train_val")
def api_compare_train_val():
    """训练集 & 验证集核心 KPI 指标对比"""
    try:
        return compare_train_val()
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


# ==================== 新增接口共用依赖 ====================
from functools import lru_cache
import ast
import os
import pickle
import warnings

import pandas as pd

# 特征英文名 → 中文名映射表（前端特征重要性、下拉选择器使用）
FEAT_NAME_MAP = {
    "F_count": "购买频次",
    "cf_score": "协同过滤得分",
    "R_days": "最近购买天数",
    "n_buyers": "购买人数",
    "club_member_status_le": "会员状态",
    "first_buy_days": "首次购买天数",
    "n_unique_articles": "唯一商品数",
    "product_type_name_le": "商品类型",
    "product_group_name_le": "商品组",
    "sales_count": "销量",
    "M_spend": "消费金额",
    "price_log": "价格对数",
}

CATEGORY_NAME_MAP = {
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
}


def _hm_root() -> Path:
    """返回 HM 项目根目录路径（main.py → backend → dashboard → HM）"""
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    """返回 data 数据目录路径（数据位于项目根目录的 data/ 子目录）"""
    return _hm_root() / "data"


def _orig_dir() -> Path:
    """返回新项目根目录（step系列文件及消融报告均位于根目录）"""
    return _hm_root()


def _require_file(path: Path) -> Path:
    """校验文件存在，不存在则返回 404 HTTP 错误"""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    return path


def _load_all_functions(py_path: Path) -> dict:
    """
    通过 AST 解析从原项目 .py 文件中安全提取所有函数定义并 exec 执行
    返回 {函数名: 函数对象} 字典
    注意：预填充 numpy/pandas/lightgbm 等命名空间，避免原文件导入语句缺失导致的 NameError
    """
    _require_file(py_path)
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        raise HTTPException(status_code=404, detail=f"在 {py_path} 中未找到任何函数定义")

    from collections import defaultdict
    import gc as _gc
    import time as _time
    from contextlib import contextmanager

    base_ns = {
        "np": np,
        "pd": pd,
        "gc": _gc,
        "time": _time,
        "contextmanager": contextmanager,
        "defaultdict": defaultdict,
        "warnings": warnings,
        "os": os,
        "pickle": pickle,
        "SEED": SEED,
        "set_seed": lambda s: (random.seed(s), np.random.seed(s)),
    }

    try:
        import lightgbm as lgb
        base_ns["lgb"] = lgb
    except ImportError:
        pass

    mod = ast.Module(body=func_nodes, type_ignores=[])
    code = compile(mod, filename=str(py_path), mode="exec")
    ns = dict(base_ns)
    exec(code, ns, ns)
    ns.pop("__builtins__", None)
    return ns


# ==================== 单文件粒度 lru_cache（按需加载，全局只读一次） ====================

@lru_cache(maxsize=1)
def _load_val_txn():
    """加载验证集交易数据（仅读取 customer_id, article_id 两列）"""
    return pd.read_parquet(
        _require_file(_data_dir() / "val_txn.parquet"),
        columns=["customer_id", "article_id"],
        engine="pyarrow",
    )


@lru_cache(maxsize=1)
def _load_art_feat():
    """加载商品特征表中的 popularity_score（用于流行度排序）"""
    return pd.read_parquet(
        _require_file(_data_dir() / "art_feat.parquet"),
        columns=["article_id", "popularity_score"],
        engine="pyarrow",
    )


@lru_cache(maxsize=1)
def _load_item_sim():
    """加载 Item-CF 商品相似度矩阵"""
    with open(_require_file(_data_dir() / "item_sim.pkl"), "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_user_hist():
    """加载用户历史购买记录"""
    with open(_require_file(_data_dir() / "user_hist.pkl"), "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _recall_base():
    """
    召回分析公共基数据：验证集 ground truth、用户列表、商品流行度、Top12 热门商品
    仅计算一次，后续接口复用
    """
    val_txn = _load_val_txn()
    val_gt = val_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
    val_cids = list(val_gt.keys())

    art_feat = _load_art_feat()
    art_pop = art_feat.set_index("article_id")["popularity_score"].to_dict()
    pop12 = sorted(art_pop, key=lambda x: -art_pop[x])[:12]

    return {
        "val_gt": val_gt,
        "val_cids": val_cids,
        "art_pop": art_pop,
        "pop12": pop12,
    }


# ==================== 原项目函数提取（AST 方式安全复用，不改动原文件） ====================

@lru_cache(maxsize=1)
def _mapk_func():
    """提取 utils.py 中的 mapk 函数（依赖 apk，需全量提取所有函数确保跨函数调用正常）"""
    ns = _load_all_functions(_orig_dir() / "utils.py")
    if "mapk" not in ns:
        raise HTTPException(status_code=404, detail="utils.py 中不存在函数 mapk")
    return ns["mapk"]


@lru_cache(maxsize=1)
def _item_cf_recommend_func():
    """提取 step3_baseline.py 中的 item_cf_recommend 函数"""
    ns = _load_all_functions(_orig_dir() / "step3_baseline.py")
    if "item_cf_recommend" not in ns:
        raise HTTPException(status_code=404, detail="step3_baseline.py 中不存在函数 item_cf_recommend")
    return ns["item_cf_recommend"]


@lru_cache(maxsize=1)
def _generate_candidates_switch_func():
    """提取 ablation_recall.py 中带通道开关的 generate_candidates 函数"""
    ns = _load_all_functions(_orig_dir() / "ablation_recall.py")
    if "generate_candidates" not in ns:
        raise HTTPException(status_code=404, detail="ablation_recall.py 中不存在函数 generate_candidates")
    return ns["generate_candidates"]


def _ensure_non_empty_str(value: str, name: str) -> str:
    """校验字符串参数非空"""
    v = (value or "").strip()
    if not v:
        raise HTTPException(status_code=400, detail=f"{name} 不能为空")
    return v


# ==================== 新增 6 个 API 接口（召回路径分析 + 特征分析标签页） ====================

# ---------- 召回路径分析（3 个接口） ----------

@app.get("/api/recall/channels")
@lru_cache(maxsize=1)
def api_recall_channels():
    """
    召回通道统计：分别开启/关闭历史购买、Item-CF、流行度三条通道，
    调用 ablation_recall.py 的 generate_candidates 得到各通道候选统计
    """
    try:
        base = _recall_base()
        gen_cands = _generate_candidates_switch_func()
        item_sim = _load_item_sim()
        user_hist = _load_user_hist()

        val_cids = base["val_cids"]
        art_pop = base["art_pop"]

        channels = [
            ("history", True, False, False),
            ("cf", False, True, False),
            ("pop", False, False, True),
            ("history+cf+pop", True, True, True),
        ]

        out = []
        for name, use_hist, use_cf, use_pop in channels:
            cands = gen_cands(
                user_hist, item_sim, art_pop, val_cids,
                use_history=use_hist, use_cf=use_cf, use_pop=use_pop,
            )
            sizes = [len(v) for v in cands.values()]
            out.append({
                "channel": name,
                "users": len(sizes),
                "avg_candidates": float(sum(sizes) / max(len(sizes), 1)),
                "min_candidates": int(min(sizes) if sizes else 0),
                "max_candidates": int(max(sizes) if sizes else 0),
            })

        return {"channels": out}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/recall/user_path")
def api_recall_user_path(user_id: str = Query(..., alias="user_id")):
    """
    用户召回完整链路：对指定用户分别执行单通道生成候选，
    计算三通道独立/重叠候选数，返回 Sankey 图所需的 nodes + links
    """
    try:
        uid = _ensure_non_empty_str(user_id, "user_id")
        base = _recall_base()
        gen_cands = _generate_candidates_switch_func()
        item_sim = _load_item_sim()
        user_hist = _load_user_hist()
        art_pop = base["art_pop"]

        hist_c = gen_cands(user_hist, item_sim, art_pop, [uid], use_history=True, use_cf=False, use_pop=False).get(uid, [])
        cf_c   = gen_cands(user_hist, item_sim, art_pop, [uid], use_history=False, use_cf=True, use_pop=False).get(uid, [])
        pop_c  = gen_cands(user_hist, item_sim, art_pop, [uid], use_history=False, use_cf=False, use_pop=True).get(uid, [])
        all_c  = gen_cands(user_hist, item_sim, art_pop, [uid], use_history=True, use_cf=True, use_pop=True).get(uid, [])

        hist_set = set(hist_c)
        cf_set = set(cf_c)
        pop_set = set(pop_c)
        all_set = set(all_c)

        # 计算各通道独立贡献的候选数（去重后）
        only_hist = len(hist_set - (cf_set | pop_set))
        only_cf   = len(cf_set - (hist_set | pop_set))
        only_pop  = len(pop_set - (hist_set | cf_set))
        overlap   = int(max(len(all_set) - only_hist - only_cf - only_pop, 0))

        # 构建 Sankey 图节点和连线
        nodes = [
            {"name": "历史购买"},
            {"name": "相似召回"},
            {"name": "流行度"},
            {"name": "候选集合"},
        ]
        links = [
            {"source": "历史购买", "target": "候选集合", "value": only_hist},
            {"source": "相似召回", "target": "候选集合", "value": only_cf},
            {"source": "流行度",   "target": "候选集合", "value": only_pop},
        ]
        if overlap > 0:
            nodes.append({"name": "重叠"})
            links.extend([
                {"source": "历史购买", "target": "重叠",     "value": min(len(hist_set), overlap)},
                {"source": "相似召回", "target": "重叠",     "value": min(len(cf_set), overlap)},
                {"source": "流行度",   "target": "重叠",     "value": min(len(pop_set), overlap)},
                {"source": "重叠",     "target": "候选集合", "value": overlap},
            ])

        return {
            "user_id": uid,
            "history_items": list(hist_c)[:50],
            "cf_items": list(cf_c)[:100],
            "pop_items": list(pop_c)[:50],
            "candidate_items": list(all_c)[:200],
            "counts": {
                "history": len(hist_set),
                "cf": len(cf_set),
                "pop": len(pop_set),
                "all": len(all_set),
                "overlap": overlap,
            },
            "sankey": {"nodes": nodes, "links": links},
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/recall/metrics")
@lru_cache(maxsize=1)
def api_recall_metrics():
    """
    整体召回指标：计算 Popularity 基准和 Item-CF 基准在验证集上的 MAP@12
    """
    try:
        base = _recall_base()
        mapk = _mapk_func()
        item_cf_recommend = _item_cf_recommend_func()
        item_sim = _load_item_sim()
        user_hist = _load_user_hist()

        val_gt = base["val_gt"]
        val_cids = base["val_cids"]
        art_pop = base["art_pop"]
        pop12 = base["pop12"]

        actuals = [val_gt[c] for c in val_cids]

        # Popularity 基准
        preds_pop = [pop12] * len(val_cids)
        score_pop = float(mapk(actuals, preds_pop, k=12))

        # Item-CF 基准
        user_hist_sub = {cid: user_hist.get(cid, []) for cid in val_cids}
        cf_preds = item_cf_recommend(user_hist_sub, item_sim, art_pop, pop12, k=12)
        preds_cf = [cf_preds.get(c, pop12) for c in val_cids]
        score_cf = float(mapk(actuals, preds_cf, k=12))

        return {
            "metrics": [
                {"name": "popularity_map12", "value": score_pop},
                {"name": "itemcf_map12", "value": score_cf},
            ],
            "meta": {"users": len(val_cids)},
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


# ---------- 消融报告解析（特征分析公用） ----------

@lru_cache(maxsize=1)
def _load_ablation_table():
    """
    读取 step5_feature_ablation_report.md 中的 Markdown 表格，
    返回 (文件路径, 表头列表, 数据行列表, 原始文本)
    """
    report_path = _require_file(_orig_dir() / "step5_feature_ablation_report.md")
    text = report_path.read_text(encoding="utf-8")

    table_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s)

    if len(table_lines) < 3:
        raise HTTPException(status_code=500, detail="未在 step5_feature_ablation_report.md 中找到可解析的表格")

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for ln in table_lines[2:]:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) != len(header):
            continue
        rows.append(dict(zip(header, cols)))

    return report_path, header, rows, text


# ---------- 特征分析（3 个接口） ----------

@app.get("/api/features/importance")
@lru_cache(maxsize=1)
def api_features_importance():
    """
    特征重要性：从消融报告解析 |Δ全46维| 列作为特征重要性
    不加载 lgb 模型文件（model 目录不存在），完全依赖 step5_feature_ablation_report.md
    """
    try:
        report_path, header, rows, _ = _load_ablation_table()

        delta_key = next((h for h in header if "Δ" in h), None)
        if delta_key is None:
            raise HTTPException(status_code=500, detail="消融报告表格中缺少 Δ 列")

        feat_key = next((h for h in header if "特征" in h), header[0])

        items = []
        for row in rows:
            fname = (row.get(feat_key, "") or "").replace("**", "").strip()
            if not fname or "全46维" in fname or "基线" in fname:
                continue
            try:
                delta_str = (row.get(delta_key, "0") or "0").replace("+", "").replace("−", "-")
                imp = abs(float(delta_str))
            except (ValueError, TypeError):
                imp = 0.0
            cn = FEAT_NAME_MAP.get(fname, fname)
            items.append({"feature": fname, "importance": imp, "feature_cn": cn})

        items.sort(key=lambda x: -x["importance"])
        return {
            "source": "step5_feature_ablation_report.md",
            "model_file": None,
            "items": items,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


# ==================== 特征分布数据加载缓存 ====================

@lru_cache(maxsize=1)
def _load_train_txn():
    """加载训练集交易数据（仅 customer_id, article_id）"""
    return pd.read_parquet(
        _require_file(_data_dir() / "train_txn.parquet"),
        columns=["customer_id", "article_id"],
        engine="pyarrow",
    )


@lru_cache(maxsize=1)
def _feat_column_index():
    """
    读取三个特征文件的列名集合（仅读 schema 元数据，不读数据体，毫秒级）。
    后续加载时根据此索引只读取目标列，避免全量加载数百MB文件。
    """
    import pyarrow.parquet as pq
    return {
        "cus": set(pq.read_schema(str(_data_dir() / "cus_feat.parquet")).names),
        "art": set(pq.read_schema(str(_data_dir() / "art_feat.parquet")).names),
        "inter": set(pq.read_schema(str(_data_dir() / "inter_feat.parquet")).names),
    }


@lru_cache(maxsize=32)
def _compute_feat_distribution(fname: str):
    """
    计算单个特征的 train/val 双线直方图（20-bin），结果缓存。
    cus/art 特征：去重 join key 后 merge，结果通过 map 广播回全量交易——避免
      在百万行上做全量 merge。
    inter 特征：用 Polars 做高性能多线程 join，再转为 pandas Series 用于直方图。
    """
    train_txn = _load_train_txn()
    val_txn = _load_val_txn()
    col_idx = _feat_column_index()

    if fname in col_idx["cus"]:
        # 去重 customer_id 再 merge，结果通过 map 广播回全体交易
        train_keys = train_txn[["customer_id"]].drop_duplicates()
        val_keys = val_txn[["customer_id"]].drop_duplicates()
        feat_df = pd.read_parquet(
            str(_data_dir() / "cus_feat.parquet"),
            columns=["customer_id", fname], engine="pyarrow",
        )
        train_map = train_keys.merge(feat_df, on="customer_id", how="left")
        val_map = val_keys.merge(feat_df, on="customer_id", how="left")
        train_lu = train_map.set_index("customer_id")[fname]
        val_lu = val_map.set_index("customer_id")[fname]
        train_vals = train_txn["customer_id"].map(train_lu)
        val_vals = val_txn["customer_id"].map(val_lu)
    elif fname in col_idx["art"]:
        train_keys = train_txn[["article_id"]].drop_duplicates()
        val_keys = val_txn[["article_id"]].drop_duplicates()
        feat_df = pd.read_parquet(
            str(_data_dir() / "art_feat.parquet"),
            columns=["article_id", fname], engine="pyarrow",
        )
        train_map = train_keys.merge(feat_df, on="article_id", how="left")
        val_map = val_keys.merge(feat_df, on="article_id", how="left")
        train_lu = train_map.set_index("article_id")[fname]
        val_lu = val_map.set_index("article_id")[fname]
        train_vals = train_txn["article_id"].map(train_lu)
        val_vals = val_txn["article_id"].map(val_lu)
    elif fname in col_idx["inter"]:
        # Polars 多线程 join，百万行规模远快于 pandas merge
        import polars as pl
        inter_path = str(_data_dir() / "inter_feat.parquet")
        inter_df = pl.read_parquet(inter_path, columns=["customer_id", "article_id", fname])
        train_pl = pl.from_pandas(train_txn[["customer_id", "article_id"]])
        val_pl = pl.from_pandas(val_txn[["customer_id", "article_id"]])
        train_joined = train_pl.join(inter_df, on=["customer_id", "article_id"], how="left")
        val_joined = val_pl.join(inter_df, on=["customer_id", "article_id"], how="left")
        train_vals = train_joined[fname].to_pandas()
        val_vals = val_joined[fname].to_pandas()
        del inter_df, train_pl, val_pl, train_joined, val_joined
    else:
        raise HTTPException(status_code=404, detail=f"特征不存在于 cus_feat/art_feat/inter_feat: {fname}")

    train_vals = pd.to_numeric(train_vals, errors="coerce").dropna()
    val_vals = pd.to_numeric(val_vals, errors="coerce").dropna()

    if train_vals.empty or val_vals.empty:
        raise HTTPException(status_code=500, detail=f"特征 {fname} 分布数据为空")

    vmin = float(min(train_vals.min(), val_vals.min()))
    vmax = float(max(train_vals.max(), val_vals.max()))
    if vmin == vmax:
        vmax = vmin + 1.0

    train_hist, edges = np.histogram(train_vals.values, bins=20, range=(vmin, vmax))
    val_hist, _ = np.histogram(val_vals.values, bins=20, range=(vmin, vmax))
    mids = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    return {
        "feature_name": fname,
        "bins": mids,
        "train": {"count": int(train_vals.shape[0]), "hist": train_hist.astype(int).tolist()},
        "val": {"count": int(val_vals.shape[0]), "hist": val_hist.astype(int).tolist()},
        "range": {"min": vmin, "max": vmax},
    }


@app.get("/api/features/distribution")
def api_features_distribution(feature_name: str = Query(..., alias="feature_name")):
    """
    特征分布对比：对指定特征分别计算 train 和 val 的 20-bin 直方图
    自动在三张特征表中查找该列名，结果内存缓存，秒级返回
    """
    try:
        fname = _ensure_non_empty_str(feature_name, "feature_name")
        return _compute_feat_distribution(fname)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


@app.get("/api/features/ablation")
@lru_cache(maxsize=1)
def api_features_ablation():
    """消融实验结果：返回 step5_feature_ablation_report.md 的完整表格"""
    try:
        report_path, header, rows, raw_md = _load_ablation_table()
        return {"file": str(report_path), "table": {"header": header, "rows": rows}, "raw_md": raw_md}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        return error_response("数据文件不存在", str(e))
    except ModuleNotFoundError as e:
        return error_response("代码模块导入失败", str(e))
    except Exception as e:
        return error_response("接口内部错误", str(e))


app.mount("/frontend", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "frontend"), html=True), name="frontend")