# HM 推荐系统数据看板 —— 项目交付说明

## 1. 交付概述

本次交付为 **HM 个性化推荐系统数据看板模块**，已完成从旧项目到 `HM-recommendation` 新项目的**目录迁移、路径适配与全量问题修复**，可直接部署运行，满足**业务概览、召回路径分析、特征消融分析**三类可视化需求。

看板采用 **FastAPI（后端）+ ECharts 5.x（前端）** 技术栈，后端通过 AST 动态加载项目根目录下的业务代码模块，前端为单文件零依赖架构，整体以深色科技风统一设计。

---

## 2. 功能模块清单

看板包含三个标签页，共计 11 个 API 接口：

### 2.1 业务数据概览

- 交易 KPI 总览（总交易笔数、总消费金额、独立用户量、独立商品量）
- 每日销售 & 订单趋势（双 Y 轴折线图）
- 用户年龄分布
- 商品大类 Top10
- 交易聚合明细表（按日汇总）

支持 **train / val 数据集切换**与**日期区间筛选**，筛选条件全局联动：明细表、趋势图、KPI 卡片同步刷新。

| 接口 | 说明 |
|------|------|
| `GET /api/kpi/overview` | 大盘 KPI 概览 |
| `GET /api/trend/daily_sales` | 每日销售 & 订单趋势 |
| `GET /api/user/age_dist` | 用户年龄分段分布 |
| `GET /api/goods/top_category` | 商品大类销量 Top10 |
| `GET /api/compare/train_val` | Train / Val 核心指标对比 |

### 2.2 召回路径分析

- Popularity 与 Item-CF 两路召回 MAP@12 指标
- 召回通道占比（历史购买 / Item-CF / 流行度三通道统计）
- 单用户召回链路桑基图可视化

支持输入用户 ID 查询其完整召回链路，展示各通道独立贡献与重叠量。

| 接口 | 说明 |
|------|------|
| `GET /api/recall/channels` | 召回通道统计与占比 |
| `GET /api/recall/user_path` | 单用户召回链路桑基图 |
| `GET /api/recall/metrics` | Popularity / Item-CF MAP@12 指标 |

### 2.3 特征分析

- 特征重要性排序（Gain）
- 特征分布 Train / Val 对比（20-bin 直方图双线图）
- 消融实验 Δ 指标瀑布图

支持单特征分布查询与条形图联动：点击重要性排序条目即触发对应特征的分布对比查询；消融瀑布图直观展示各特征组对 MAP@12 的贡献方向。

| 接口 | 说明 |
|------|------|
| `GET /api/features/importance` | 特征重要性排序 |
| `GET /api/features/distribution` | 指定特征 Train / Val 分布对比 |
| `GET /api/features/ablation` | 消融实验完整表格与 Δ 瀑布图 |

---

## 3. 迁移适配说明

### 3.1 目录结构适配

原项目代码位于 `original-code-for-the-HM-competition/` 子目录，新项目代码位于根目录，已修正所有代码路径引用：

```python
def _orig_dir() -> Path:
    """返回新项目根目录（step系列文件及消融报告均位于根目录）"""
    return _hm_root()
```

### 3.2 路径计算适配

全部采用基于 `__file__` 的绝对路径计算，不依赖运行时工作目录，彻底规避相对路径漂移导致的文件找不到问题：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]    # main.py → backend → dashboard → 项目根
DATA_DIR    = PROJECT_ROOT / "data"
ABLATION_REPORT_PATH = PROJECT_ROOT / "step5_feature_ablation_report.md"
```

### 3.3 启动环境适配

内置虚拟环境自动创建与依赖安装脚本，无需手动配置 Python 环境，开箱即用：

1. 检测虚拟环境是否存在 → 不存在则自动创建
2. 自动安装 `requirements.txt` 中的全部依赖
3. 启动 uvicorn 热重载模式服务

---

## 4. 问题修复记录

### 4.1 跨域访问

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| 非 8000 端口前端请求被浏览器拦截 CORS，本地 file 协议、Live Server 等前端访问方式均无法调用接口 | 新增 `CORSMiddleware` 全局中间件，`allow_origins=["*"]`，放行所有来源、方法与请求头 | `main.py` |

### 4.2 启动失败

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| `No module named uvicorn`，后端从未成功启动 | 重写 `run_dashboard.bat`：自动创建 venv + 安装依赖 + 启动服务 | `run_dashboard.bat` |
| `.bat` 脚本中文编码导致 CMD 解析乱码 | 全部改为 ASCII 纯英文提示文本 | `run_dashboard.bat` |
| `requirements.txt` 缺少 `polars`、`numpy`、`lightgbm` 等依赖 | 补充全部缺失依赖，并钉版 `fastapi==0.115.0`、`uvicorn==0.30.6`、`pandas==2.2.2` | `requirements.txt` |

### 4.3 召回接口异常

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| `name 'SEED' is not defined` — `/api/recall/channels`、`/api/recall/user_path` 等全部召回接口返回 500 错误 | 在模块级定义 `SEED = 42` 常量，并注入 AST exec 执行所需的 `base_ns` 命名空间，同时注入 `set_seed` 函数 | `main.py` |

> **根因**：`_load_all_functions()` 通过 AST 提取函数体 `compile + exec` 方式加载原项目函数，此过程会丢失原文件中的顶层 `import` 语句与全局常量（如 `from config import SEED`），需在 `base_ns` 中预填充必要变量。

### 4.4 接口超时

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| `/api/features/distribution` 首次请求 60s 超时，无法返回数据 | **四项优化**： ① 交易数据按需读列（`columns=["customer_id", "article_id"]`）避免全量 IO；② cus/art 分支用 `drop_duplicates()` + `merge()` + `Series.map()` 广播替代全量百万行 merge；③ inter 分支用 Polars 多线程 join 替代 pandas merge；④ `_compute_feat_distribution()` 加 `@lru_cache(maxsize=32)` 缓存分箱结果，后续相同特征查询命中缓存毫秒级返回 | `main.py` |

### 4.5 前端显示问题

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| 消融实验瀑布图配色偏绿，与深色科技蓝系整体风格不一致 | 统一为蓝系渐变配色（`#0066cc → #0099ff → #00ccff`），条形方向（正/负 Δ）区分左右渐变方向 | `index.html` |
| 双 Y 轴标签重叠（销售额 / 订单量轴标签过于靠近） | 增加两侧 `axisLabel.margin` 间距，左轴 `margin: 20`、右轴 `margin: 25`，右侧 Y 轴关闭 splitLine | `index.html` |
| 召回通道饼图尺寸偏小，legend 区域挤压 canvas | 增大 canvas 绘制区域，三通道独立配色（history `#0066cc` / cf `#0099ff` / pop `#33ccff`），增强图例字号 | `index.html` |
| 下拉选择器（数据集切换、特征选择）默认浅色样式，与深色背景不协调 | 统一定义深色 `select-inline` 样式：背景 `#1a2a4a`、文字白色、`option` 深色底、蓝色发光边框 | `index.html` |
| 数据来源卡片文字 `step5_feature_ablation_report.md` 路径过长导致截断 | 添加 `white-space: nowrap; overflow: hidden; text-overflow: ellipsis` + `title` 属性，悬浮可查看完整路径 | `index.html` |
| 前端 `"Failed to fetch"` 提示笼统，无法定位是后端未启动、文件缺失还是代码异常 | 新增 `_formatFetchError()` 函数，区分「后端服务未启动」「请求超时」「HTTP 状态错误」「业务错误」四种场景并输出可读提示 | `index.html` |
| `fetchJson()` 不识别后端返回的 `{"error": true}` 业务错误标记 | 在 `fetchJson()` 中新增 `if (data.error) throw new Error(data.message)` 逻辑，将后端结构化错误透传到前端 | `index.html` |

### 4.6 数据一致性

| 问题 | 修复 | 涉及文件 |
|------|------|----------|
| 数据集切换（train/val）时明细表、趋势图不同步刷新，出现旧数据残留 | 数据集切换触发 `refreshAll({resetDate: true})` 全量重新请求 5 个业务接口并重设日期，确保筛选条件全局联动 | `index.html` |

---

## 5. 技术栈与依赖

### 后端

| 组件 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.115.0 | Web 框架，提供 11 个 REST API 接口 |
| Uvicorn | 0.30.6 | ASGI 服务器，支持 `--reload` 热重载 |
| Pandas | 2.2.2 | 数据加载、聚合、merge |
| Polars | latest | 多线程列式 join（inter 特征加速） |
| PyArrow | latest | Parquet 文件读写引擎及 schema 元数据读取 |
| NumPy | latest | 直方图计算（`np.histogram`） |
| LightGBM | latest | AST 动态加载原项目函数时的依赖环境 |
| functools.lru_cache | 内置 | 数据缓存与计算结果缓存 |

### 前端

| 组件 | 版本 | 用途 |
|------|------|------|
| ECharts | 5.x（CDN） | 全部图表渲染（折线图、柱状图、饼图、桑基图、瀑布图） |
| 原生 HTML / CSS / JS | — | 无额外构建依赖，单文件交付 |

---

## 6. 验收标准

### 6.1 启动验收

- 双击 `run_dashboard.bat` 可正常启动服务，控制台输出 `Uvicorn running on http://0.0.0.0:8000`
- 访问 `http://127.0.0.1:8000/docs` 可正常打开 Swagger 文档，11 个接口完整列出

### 6.2 接口验收

- 业务接口（`/api/kpi/overview`、`/api/trend/daily_sales` 等 5 个）全部正常返回 JSON 数据，无 500、`FileNotFoundError`
- 召回接口（`/api/recall/channels`、`/api/recall/user_path`、`/api/recall/metrics`）全部正常返回，无 `SEED is not defined` 等变量未定义异常
- 特征接口（`/api/features/importance`、`/api/features/distribution`、`/api/features/ablation`）正常返回，分布接口首次请求不超时，后续请求缓存命中
- 所有接口在数据文件缺失时返回结构化错误 `{"error": true, "message": "..."}` 而非 HTTP 500 崩溃

### 6.3 页面验收

- 三个标签页正常渲染，深色科技蓝主题风格统一
- 所有图表无布局错乱、文字截断、样式失效问题
- 下拉选择器、输入框、按钮等控件深色样式一致

### 6.4 功能验收

- 数据集切换（train/val）联动刷新全部业务概览图表与明细表
- 日期区间筛选联动刷新趋势图、明细表
- 召回用户 ID 查询正常展示桑基图召回链路
- 特征重要性条形图点击联动切换对应特征分布对比图
- 消融瀑布图正确展示正/负 Δ 方向

---

## 7. 交付文件清单

```
dashboard/
├── backend/
│   ├── main.py                    # FastAPI 主入口（748 行，11 个 API 接口）
│   ├── data_loader.py             # 业务数据聚合函数
│   ├── cache_utils.py             # Parquet/Pickle 文件读取缓存
│   └── requirements.txt           # Python 依赖清单（7 个包）
├── frontend/
│   └── index.html                 # 看板前端页面（490 行，三标签页单文件）
├── run_dashboard.bat              # Windows 一键启动脚本（32 行）
├── README.md                      # 看板部署与使用说明
└── dashboard_deliverable.md       # 本交付说明文档
```

**项目根目录依赖文件**（看板运行所需，非看板模块自身交付）：

```
HM-recommendation/
├── data/                          # 10 个 Parquet/Pickle 数据文件
├── utils.py                       # mapk 评估函数
├── step3_baseline.py              # Item-CF 推荐函数
├── ablation_recall.py             # 多通道召回候选生成
├── config.py                      # SEED 常量等全局配置
├── step5_feature_ablation_report.md  # 特征消融实验报告（46 维）
└── ...
```"}