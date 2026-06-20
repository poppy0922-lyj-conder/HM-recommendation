# H&M 个性化推荐系统

基于 LightGBM LambdaRank 的商品排序推荐管线，覆盖数据处理、特征工程、模型训练、消融实验、多模型集成到提交文件生成的全流程。

## 项目结构

```
F:\HM-recommendation/
├── config.py                    # 全局配置 (路径/参数/随机种子/特征列定义)
├── utils.py                     # 工具函数 (MAP@K 评估/计时器/随机种子)
├── requirements.txt             # Python 依赖清单
├── .gitignore
│
├── step1_load_data.py           # 数据加载与验证集划分
├── step2_features.py            # 特征工程 (5类 46维特征)
├── step3_baseline.py            # 基线模型评估 (流行度/Item-CF)
├── step4_candidates.py          # LTR 候选生成
├── step5_train.py               # LightGBM LambdaRank 排序模型训练 (46维)
├── step6_infer.py               # 全量特征重建 + 分批推理 + 提交
│
├── step5_lowlr.py               # 低学习率+多树 (lr=0.02, 2000轮)
├── step5_xgboost.py             # XGBoost rank:ndcg 异构排序模型
├── step5_catboost.py            # CatBoost YetiRank 排序
├── step5_xendcg.py              # LightGBM rank_xendcg 目标函数
├── step5_ensemble_infer.py      # 四模型等权平均集成推理
├── step5_expanded.py            # 多通道扩大召回+负采样排序训练
├── step5_item2vec.py            # Word2Vec 序列嵌入增强版 (56维)
│
├── ablation_feature.py          # 特征消融实验 (price_ratio/recency_decay/category_match)
├── ablation_recall.py           # 召回通道消融实验 (仅流行度/+历史/+CF/+历史+CF)
├── train_after_ablation.py      # 消融后最终训练 (23维精简特征)
├── step_neg_sampling.py         # 负采样训练脚本 (控制正负比 1:5)
│
└── output/                      # 模型与提交文件 (自动生成)
```

## 运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 准备数据

从 [Kaggle H&M 竞赛](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 下载数据，放入 `F:/H&M_data/` 目录：

```
F:/H&M_data/
├── articles.csv
├── customers.csv
├── transactions_train.csv
└── sample_submission.csv
```

### 3. 基础管线 (按顺序运行)

```bash
python step1_load_data.py          # 加载数据 → 按时间切分 train/val
python step2_features.py           # 特征工程 → 5类特征
python step3_baseline.py           # 基线评估 → 流行度 + Item-CF MAP@12
python step4_candidates.py         # 候选生成 → 每用户候选商品列表
python step5_train.py              # LTR 训练 → Phase 1 早停 + Phase 2 全量训练
python step6_infer.py              # 全量推理 → 生成 submission.csv
```

## 管线详解

### 基础管线

| Step | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1 | 数据加载与切分 | CSV 原始数据 | train/val Parquet |
| 2 | 特征工程 | train_txn | 客户8维 + 商品33维 + 交互3维 + Item-CF + 用户历史 |
| 3 | 基线评估 | 特征 + val_txn | MAP@12 + baseline 提交文件 |
| 4 | 候选生成 | 特征 + val_txn | val_candidates.pkl |
| 5 | 排序训练 | 特征 + 候选 | model.txt + 评估报告 |
| 6 | 全量推理 | 全量数据 + model.txt | submission.csv |

### 特征维度 (46维)

- **客户特征 (8维)**: age, club_member_status_le, postal_le, R_days, F_count, M_spend, avg_price_user, n_unique_articles
- **商品特征 (33维)**: avg_price, sales_count, n_buyers, popularity_score, price_log, sales_log, 7个类别编码 + 20维文本嵌入
- **交互特征 (3维)**: buy_count, last_buy_days, first_buy_days
- **候选特征 (2维)**: cf_score, price_match

### 训练策略

- **Phase 1**: val 数据内再切分 (前5天训练 / 后2天 holdout 评估) → 早停 (patience=50) → 确定最佳迭代轮数
- **Phase 2**: 使用全部 val 数据重新训练固定轮数 → 保存最终模型

## 改进实验

### 1. 消融实验

#### 特征消融 (`ablation_feature.py`)
独立运行，测试 3 个新特征对 MAP@12 的影响：
- **price_ratio**: 商品价格 / 用户均价
- **recency_decay**: 平滑时间衰减 exp(-last_buy_days/30)
- **category_match**: 品类是否匹配用户最常买品类

原则：一次只加一个变量，对比 Base 模型 vs Base+新特征 的 MAP@12 变化。

#### 召回通道消融 (`ablation_recall.py`)
评估不同召回通道的组合效果：
- 仅流行度 → +历史 → +CF → +历史+CF

每个实验包含本地 CV 评估 + 全量推理 + 提交文件。

#### 消融后训练 (`train_after_ablation.py`)
基于消融结果使用 23 维精简特征训练：
- 客户特征从 8 维精简到 4 维
- 商品特征从 33 维精简到 11 维
- 交互 3 维 + 候选 2 维不变
- 学习率降至 0.02，轮数增至 2000

#### 负采样训练 (`step_neg_sampling.py`)
在消融后训练基础上增加负采样（正:负 = 1:5），缓解正负样本不平衡问题。

### 2. 多模型架构

| 脚本 | 模型 | 目标函数 | 特征维度 |
|------|------|----------|----------|
| `step5_lowlr.py` | LightGBM | LambdaRank | 23维 |
| `step5_xgboost.py` | XGBoost | rank:ndcg | 23维 |
| `step5_catboost.py` | CatBoost | YetiRank | 23维 |
| `step5_xendcg.py` | LightGBM | rank_xendcg | 23维 |
| `step5_ensemble_infer.py` | 四模型集成 | 等权平均 | 23维 |

四个模型使用不同的排序损失函数，误差模式互不相关，集成后可互相抵消。

### 3. 高级召回策略

| 脚本 | 特点 |
|------|------|
| `step5_expanded.py` | 多通道召回 (历史+CF+类目+复购+流行度) 约55候选/用户，1:5负采样 |
| `step5_item2vec.py` | Word2Vec 购买序列嵌入 32维 + 余弦相似度，共56维特征 |

### 4. Word2Vec 语义候选扩展

`step5_lowlr` / `step5_xgboost` / `step5_catboost` / `step5_xendcg` 均支持：
- 检测 `output/word2vec_article.model` 是否存在
- 若存在，在候选生成时通过 `w2v_model.wv.most_similar()` 扩展每用户候选集

## 配置说明

主要参数在 `config.py` 中调整：

```python
SEED = 42                              # 全局随机种子
VAL_DAYS = 7                           # 验证集天数
LGB_PARAMS = {                         # LightGBM 参数
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 8,
    ...
}
INFER_BATCH_SIZE = 50000               # 推理批次大小
```

## 随机种子

所有 step 开头均调用 `set_seed(SEED)` 保证可复现。

## 环境要求

- Python >= 3.9
- 依赖见 requirements.txt
- GPU 可选 (LightGBM/CatBoost 自动检测 GPU/CPU)

## License

MIT
