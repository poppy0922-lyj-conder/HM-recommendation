# HM 推荐系统可视化数据看板

HM 个性化推荐系统可视化数据看板，覆盖**业务数据概览、召回路径分析、特征消融分析**三大模块，技术栈为 **FastAPI 后端 + ECharts 前端**，采用深色科技风统一设计。

---

## 目录结构

```
dashboard/
├── backend/               # FastAPI 后端服务
│   ├── main.py            # 主入口，包含所有接口与数据加载逻辑
│   ├── cache_utils.py     # 数据缓存工具
│   ├── data_loader.py     # 数据加载封装
│   └── requirements.txt   # Python 依赖清单
├── frontend/
│   └── index.html         # 看板前端页面（单文件）
├── run_dashboard.bat      # Windows 一键启动脚本
└── README.md
```

---

## 依赖文件与数据路径

### 数据集文件

统一存放于**项目根目录 `../data/`**，包含以下文件：

| 文件 | 用途 |
|------|------|
| `train_txn.parquet` | 训练集交易数据 |
| `val_txn.parquet` | 验证集交易数据 |
| `articles.parquet` | 商品信息表 |
| `customers.parquet` | 用户信息表 |
| `cus_feat.parquet` | 用户特征（8维） |
| `art_feat.parquet` | 商品特征（33维） |
| `inter_feat.parquet` | 用户-商品交互特征（3维） |
| `item_sim.pkl` | Item-CF 商品相似度矩阵 |
| `user_hist.pkl` | 用户历史购买记录 |
| `val_candidates.pkl` | 验证集召回候选集 |

### 业务依赖代码

存放于**项目根目录 `../`**，接口通过 AST 动态加载以下模块中的函数：

- **`utils.py`** — `mapk`（MAP@12 评估指标）
- **`step3_baseline.py`** — `item_cf_recommend`（Item-CF 协同过滤推荐）
- **`ablation_recall.py`** — `generate_candidates`（多通道召回候选生成）

### 特征消融报告

存放于**项目根目录 `../step5_feature_ablation_report.md`**，解析 Markdown 表格供特征重要性排序与消融瀑布图渲染。

> **注意**：`data/` 目录为大体积数据集，**不纳入 Git 版本管理**，使用前需将数据集放置到项目根目录对应路径。

---

## 环境与启动方式

### 运行环境

- Python **3.10** 及以上

### 一键启动（推荐）

双击 `dashboard/run_dashboard.bat`，脚本自动完成「虚拟环境创建 → 依赖安装 → 服务启动」全流程：

1. **首次启动**：自动创建虚拟环境并安装 `requirements.txt` 中的全部依赖
2. **后续启动**：直接复用已有环境，秒级完成

### 手动启动（备用）

```bash
cd dashboard/backend
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 60
```

---

## 访问地址

| 入口 | 地址 |
|------|------|
| 看板页面 | `http://127.0.0.1:8000/frontend/index.html` |
| 接口文档（Swagger） | `http://127.0.0.1:8000/docs` |

---

## 注意事项

- 服务启动后**请勿关闭终端窗口**，关闭即停止服务。
- 大文件首次加载会有延迟，已内置 **lru_cache 内存缓存**，二次访问速度显著提升。
- 端口 8000 被占用时，可修改启动脚本中的 `--port` 参数调整端口。