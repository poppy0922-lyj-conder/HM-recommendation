# H&M 个性化推荐系统

基于 LightGBM LambdaRank + Word2Vec 序列嵌入的商品排序推荐管线。

最优模型：**Item2Vec 56维 + 语义候选扩展**（Private 0.02908）

## 项目结构

```
F:\HM-recommendation/
├── step5_item2vec.py          ← 最优模型 (56维: 23基线 + 32 a2v + 1 v2v_sim)
├── step7_shap.py              # SHAP 模型可解释性分析
├── step9_cv.py                # 5 折时序交叉验证稳定性分析
├── step1_load_data.py         # 数据加载与验证集划分
├── step2_features.py          # 特征工程 (5类 46维特征)
├── step3_baseline.py          # 基线模型评估 (流行度/Item-CF)
├── step4_candidates.py        # LTR 候选生成
├── step6_submit.py            # 全量推理 + 提交文件
├── config.py                  # 全局配置
├── utils.py                   # 工具函数
├── requirements.txt
├── output/                    # 模型与输出文件
│   ├── model_item2vec.txt           LightGBM 模型
│   ├── cv_results.csv               5 折 CV 结果
│   ├── cv_report.txt                CV 稳定性报告
│   └── cv_roc_curves.png            ROC 曲线
│
├── history/                   # 历史改进代码
│   ├── step5_lowlr.py             低学习率+多树 (lr=0.02, 2000轮)
│   ├── step5_xgboost.py           XGBoost rank:ndcg
│   ├── step5_catboost.py          CatBoost YetiRank
│   ├── step5_xendcg.py            LightGBM rank_xendcg
│   ├── step5_clean.py / round2    特征精简 (46→34→23)
│   ├── step5_full_train.py        全量训练
│   ├── step5_ensemble_infer*.py   多模型集成 (2/3/4模型, RRF, 权重搜索)
│   ├── step5_ensemble_infer_4model_item2vec.py  含Item2Vec四模型集成
│   ├── step5_nn.py / twotower.py  神经网络尝试
│   ├── step5_two_stage.py         两阶段排序尝试
│   ├── step5_expanded.py          多通道扩大召回
│   └── step6_infer.py             旧版推理
│
├── ablation/                  # 消融实验
│   ├── ablation_feature.py        特征消融
│   ├── ablation_recall.py         召回通道消融
│   ├── train_after_ablation.py    消融后训练 (23维)
│   ├── step_neg_sampling.py       负采样训练 (1:5)
│   └── step5_feature_ablation_report.md
│
└── dashboard/                 # 数据看板 (队友贡献)
    ├── backend/
    ├── frontend/
    └── run_dashboard.bat
```

## 运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 准备数据

从 [Kaggle H&M 竞赛](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 下载原始 CSV 数据，放入 `E:/H&M_data/`：

```
E:/H&M_data/
├── articles.csv
├── customers.csv
├── transactions_train.csv
└── sample_submission.csv
```

> 中间处理后的 Parquet 特征数据 (step1~step4 输出) 可在 [data 仓库](https://github.com/poppy0922-lyj-conder/data.git) 中获取，
> 直接放置到 `E:/H&M_data/processed/` 目录下即可跳过 step1~step4 直接运行 step5。

### 3. 最优模型运行

```bash
python step1_load_data.py          # 数据加载
python step2_features.py           # 特征工程
python step3_baseline.py           # 基线评估
python step4_candidates.py         # 候选生成
python step5_item2vec.py           # Item2Vec 训练 + 推理 (最优模型)
python step6_submit.py             # 全量推理提交
python step7_shap.py               # SHAP 可解释性分析 (需先有模型)
python step9_cv.py                 # 5 折时序交叉验证稳定性分析 (可选, 需先有特征)
```

参数调优示例:

```bash
# vector_size 调优 (默认32)
python step5_item2vec.py --vector_size 64
python step5_item2vec.py --vector_size 128
# CBOW + 更多轮次
python step5_item2vec.py --sg 0 --epochs 30
```

## 改进历程

| 阶段 | 操作 | Private |
|:---:|------|:-------:|
| 基线 | 46维 + LightGBM | 0.02604 |
| 特征精简 | 46→23维 | 0.02746 |
| LowLR | lr=0.02, 2000轮 | **0.02804** |
| 双模型集成 | lowlr+cat | 0.02821 |
| 三模型集成 | lowlr+cat+xgb | **0.02825** |
| Item2Vec V1 | 首次引入 56维 | 0.02849 |
| Item2Vec V2 | 全量数据重训 | 0.02872 |
| **Item2Vec V4** | **语义候选扩增** | **0.02908** |

## 配置说明

主要参数在 `config.py` 中调整：

```python
VAL_DAYS = 7                           # 验证集天数
LGB_PARAMS = {'learning_rate': 0.05, 'num_leaves': 127, ...}
INFER_BATCH_SIZE = 50000               # 推理批次大小
```

## 环境要求

- Python >= 3.9
- 依赖见 requirements.txt
- GPU 可选 (LightGBM 自动检测 GPU/CPU)
