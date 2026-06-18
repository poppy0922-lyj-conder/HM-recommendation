"""
H&M 推荐模型 — 工具函数

包含: MAP@K 评估、计时器、随机种子设置。
"""

import numpy as np
import random
import gc
import time
from contextlib import contextmanager

from config import SEED


def set_seed(seed: int = SEED):
    """设置全局随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def apk(actual, predicted, k=12):
    """单用户 Average Precision@K

    actual: list[int]  用户实际购买的商品列表
    predicted: list[int]  模型推荐的商品列表 (top-k)
    """
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score, hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actuals, preds, k=12):
    """全局 Mean Average Precision@K

    actuals: list[list[int]]  每个用户的实际购买
    preds:   list[list[int]]  每个用户的推荐结果
    """
    return np.mean([apk(a, p, k) for a, p in zip(actuals, preds)])


def clean_memory(*objs):
    """删除多个对象并触发垃圾回收"""
    for obj in objs:
        del obj
    gc.collect()


@contextmanager
def timer(name="Stage"):
    """上下文管理器，打印阶段耗时

    用法:
        with timer("数据加载"):
            data = load_data()
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"[{name}] 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
