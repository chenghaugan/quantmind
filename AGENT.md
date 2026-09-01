# AGENT.md — QuantMind

> AI 驱动的国内多资产（期货/A股/港股/期权）量化投研框架

## 项目定位

QuantMind 是一个端到端的量化投研平台，覆盖 **数据获取 → 因子研究 → 策略回测 → 模拟盘 → 实盘** 全链路。面向中国期货市场（商品/金融期货），兼顾 A 股、港股、期权。提供三种使用方式：CLI / REST API / Streamlit Web。

## 技术栈

| 层 | 技术 |
|---|---|
| 语言 | Python 3.13 |
| 后端 API | FastAPI + Uvicorn（端口 8000） |
| 前端 Web | Streamlit（端口 8501） |
| 数据库 | TimescaleDB（PostgreSQL 16 + 时序扩展） |
| 缓存 | Redis 7 |
| 调度 | APScheduler |
| LLM | OpenAI-compatible（默认 mock，可接 DeepSeek/OpenAI/Anthropic） |
| 数据源 | AKShare / efinance / yfinance / mootdx / 本地 Parquet |
| 容器化 | Docker Compose（4 服务：timescaledb + redis + api + web） |

## 目录结构

```
quantmind/
├── quantmind/                 # 主包
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings（QM_ 前缀环境变量）
│   ├── cli.py                 # Typer CLI 入口
│   │
│   ├── core/                  # vnpy 风格核心：Event/Engine/Gateway/Constant
│   │   ├── event.py           # EventType 枚举 + Event 数据类
│   │   ├── engine.py          # EventEngine 事件总线
│   │   ├── constant.py        # Exchange/Interval/Direction/Offset
│   │   ├── contracts.py       # 合约尺寸 default_size()
│   │   ├── gateway.py         # 网关抽象基类
│   │   ├── object.py          # Bar/Trade/Order/Position 等数据对象
│   │   └── profile.py         # 性能 profiling
│   │
│   ├── data/                  # 数据层
│   │   ├── manager.py         # DataManager 异步查询入口
│   │   ├── quality.py         # 数据质量体检（间隙/尖峰/换月/新鲜度）
│   │   ├── feed/              # 数据源适配器
│   │   │   ├── registry.py    # build_default_registry() 注册全部源
│   │   │   ├── base.py        # HistoryRequest / Feed 协议
│   │   │   ├── mock.py        # 离线 mock（无 key 默认回退）
│   │   │   ├── akshare_future.py   # AKShare 期货
│   │   │   ├── akshare_option.py   # AKShare 期权
│   │   │   ├── efinance_feed.py    # efinance A股/港股
│   │   │   ├── em_hk.py            # 东方财富港股
│   │   │   ├── yfinance_us.py      # yfinance 美股
│   │   │   ├── mootdx_astock.py    # mootdx A股实时
│   │   │   ├── astock_parquet.py   # A股本地 Parquet
│   │   │   ├── local_daily.py      # 本地日线 CSV/Parquet
│   │   │   ├── local_file.py       # 通用本地文件
│   │   │   ├── market_universe.py  # 全市场标的池
│   │   │   ├── seat_position_csv.py     # 席位持仓 CSV
│   │   │   └── china_futures_csv.py     # 中国期货 CSV
│   │   └── store/             # 持久化
│   │       ├── timescale.py   # TimescaleDB 存储
│   │       ├── cache.py       # InMemoryStore
│   │       ├── disk_cache.py  # DiskBarCache（Parquet 写缓存 → data_cache/）
│   │       └── fixture.py     # 测试 fixture
│   │
│   ├── research/              # 因子研究
│   │   ├── evaluator.py       # IC/IR/衰减/单调性评估
│   │   ├── eval.py            # 评估辅助
│   │   ├── judge.py           # LLM 因子判读
│   │   ├── dedup.py           # 相关性去冗余
│   │   ├── split.py           # 训练/验证/测试切分（防泄漏）
│   │   ├── combine.py         # 因子复合
│   │   ├── decay.py           # 因子衰减监控
│   │   ├── barra.py           # Barra 风险模型
│   │   ├── neutralize.py      # 中性化
│   │   ├── target.py          # 目标函数
│   │   ├── ml_factor.py       # ML 因子
│   │   ├── ml_ranker.py       # ML 排序
│   │   ├── risk_xray.py       # 风险 X-Ray
│   │   ├── cross_sectional_backtest.py  # 截面回测
│   │   ├── knowledge_loop.py  # 知识库闭环
│   │   ├── pipeline.py        # 因子挖掘流水线核心
│   │   ├── orchestrator.py    # 端到端编排
│   │   ├── factors/           # 因子库
│   │   │   ├── alpha101.py    # WorldQuant Alpha101
│   │   │   ├── alpha191.py    #国泰君安 Alpha191
│   │   │   ├── alpha_cs.py    # 截面 Alpha（Panel + DSL）
│   │   │   ├── gtja191.py     # GTJA 191
│   │   │   ├── qlib158.py     # Qlib 158
│   │   │   ├── wq.py          # WorldQuant DSL 原语
│   │   │   ├── technical.py   # 技术指标因子
│   │   │   ├── academic.py    # 学术因子
│   │   │   ├── seat_futures.py     # 期货席位因子 F1-F8
│   │   │   ├── panel_expr.py  # 面板表达式 DSL
│   │   │   ├── expression.py  # 表达式解析
│   │   │   ├── seed_pool.py   # 种子池
│   │   │   └── registry.py    # 因子注册
│   │   └── search/            # 因子搜索算法
│   │       ├── base.py        # 搜索基类
│   │       ├── cot.py         # 链式精炼（Chain-of-Thought）
│   │       ├── ea.py          # 进化算法
│   │       └── tot.py         # 树状搜索
│   │
│   ├── strategy/              # 策略层
│   │   ├── base.py            # CtaTemplate 策略基类
│   │   ├── runners.py         # run_strategy(mode) 统一入口
│   │   ├── context.py         # 策略上下文
│   │   ├── validation.py      # 策略校验
│   │   ├── mined.py           # AI 挖掘策略注册
│   │   ├── dual_ma.py         # 双均线
│   │   ├── multifactor.py     # 多因子
│   │   ├── pair.py            # 配对交易
│   │   ├── allweather.py      # 全天候
│   │   └── components/        # 可组合策略框架
│   │       ├── alpha.py       # Alpha 组件
│   │       ├── ml_alpha.py    # ML Alpha 组件
│   │       ├── universe.py    # 选股域
│   │       ├── risk.py        # 风控组件
│   │       ├── portfolio.py   # 组合构建
│   │       ├── execution.py   # 执行组件
│   │       └── composable.py  # 组合器
│   │
│   ├── backtest/              # 回测引擎
│   │   ├── engine.py          # 回测核心引擎
│   │   ├── broker.py          # 模拟经纪商
│   │   ├── cost.py            # 手续费/滑点模型
│   │   ├── analyzer.py        # 绩效分析（Sharpe/MDD/Calmar…）
│   │   ├── optimizer.py       # 参数寻优（网格 + Optuna）
│   │   ├── walkforward.py     # Walk-Forward 滚动回测
│   │   ├── validation.py      # 回测结果校验
│   │   └── diagnostics.py     # 回测诊断
│   │
│   ├── risk/                  # 风控层
│   │   ├── engine.py          # 委托风控预检
│   │   ├── limits.py          # 限额档位（default/conservative/unlimited）
│   │   ├── portfolio.py       # 组合风控
│   │   ├── turbulence.py      # 市场湍流度
│   │   └── calendar.py        # 交易日历（Asia/Shanghai）
│   │
│   ├── ai/                    # LLM 层
│   │   ├── provider.py        # build_provider()（mock/openai/deepseek/anthropic）
│   │   ├── agent.py           # ResearchAgent：idea → spec/factors/strategy
│   │   ├── factor_gen.py      # LLM 因子生成
│   │   ├── codegen.py         # 策略代码生成
│   │   ├── sandbox.py         # AST 沙箱安全校验
│   │   ├── safety.py          # 安全检查
│   │   ├── idea_parser.py     # 投资想法解析
│   │   ├── memory.py          # AI 记忆
│   │   ├── prompts.py         # 提示词模板
│   │   ├── expr_map.py        # 表达式映射
│   │   └── knowledge_enrichment.py  # 知识增强
│   │
│   ├── live/                  # 实盘交易网关
│   │   ├── ctp_gateway.py     # CTP（期货）
│   │   ├── xtp_gateway.py     # XTP（股票）
│   │   ├── ib_gateway.py      # IB（盈透）
│   │   ├── order_manager.py   # 订单管理
│   │   ├── sim.py             # 模拟网关
│   │   ├── reconcile.py       # 对账
│   │   └── runner.py          # 实盘运行器
│   │
│   ├── paper/                 # 模拟盘
│   │   ├── engine.py          # PaperEngine 历史回放
│   │   └── promotion.py       # LifecycleManager（RESEARCH→BACKTEST→PAPER→LIVE）
│   │
│   ├── knowledge/             # 知识库
│   │   ├── store.py           # KnowledgeStore（因子/策略/研究日志/方法论）
│   │   ├── schema.py          # 知识库 schema
│   │   ├── seeds.py           # 种子知识
│   │   └── web_source.py      # 网络知识源（Tavily）
│   │
│   ├── strategy_mining/       # LLM 策略挖掘
│   │   ├── architect.py       # 策略架构师
│   │   ├── auto_backtest.py   # 自动回测循环
│   │   ├── compiler.py        # 策略编译器
│   │   ├── prompts.py         # 挖掘提示词
│   │   └── schema.py          # 挖掘 schema
│   │
│   ├── monitoring/            # 监控
│   │   └── notifier.py        # 通知器
│   │
│   ├── benchmark/             # 基准测试
│   │   ├── runner.py          # 基准运行器
│   │   ├── tasks.py           # 基准任务
│   │   └── run_benchmark.py   # 入口
│   │
│   ├── models/                # 【运行时生成】ML 模型持久化（joblib，由 ml_service 自动创建）
│   │                            # 训练好的因子/排序模型存 .joblib，容器以 root 创建此目录
│   │
│   ├── api/                   # FastAPI 后端
│   │   ├── app.py             # FastAPI 应用 + lifespan + 全部路由
│   │   ├── auth.py            # JWT 认证
│   │   ├── routes_auth.py     # 认证路由
│   │   ├── routes_ml.py       # ML 路由
│   │   ├── routes_profile.py  # Profile 路由
│   │   ├── schemas.py         # Pydantic 请求/响应模型
│   │   ├── schemas_auth.py    # 认证 schema
│   │   ├── ws.py              # WebSocket 管理
│   │   ├── scheduler.py       # APScheduler 调度器
│   │   ├── logging_config.py  # 日志配置
│   │   └── services/          # Service 层（一个领域一个 service）
│   │       ├── data_service.py
│   │       ├── data_admin_service.py
│   │       ├── data_settings_service.py
│   │       ├── factor_service.py
│   │       ├── backtest_service.py
│   │       ├── research_service.py
│   │       ├── risk_service.py
│   │       ├── optimize_service.py
│   │       ├── search_service.py
│   │       ├── knowledge_service.py
│   │       ├── lifecycle_service.py
│   │       ├── seat_service.py
│   │       ├── settings_service.py
│   │       ├── strategy_mining_service.py
│   │       ├── ml_service.py
│   │       ├── profile_service.py
│   │       └── alert_settings_service.py
│   │
│   └── web/                   # Streamlit 前端
│       ├── streamlit_app.py   # 首页入口
│       ├── README.md          # 前端升级路径说明
│       ├── pages/             # 25 个功能页面
│       │   ├── 1_仪表盘.py
│       │   ├── 2_行情数据.py
│       │   ├── 3_因子研究.py
│       │   ├── 4_策略回测.py
│       │   ├── 6_生命周期.py
│       │   ├── 7_实时监控.py
│       │   ├── 8_WalkForward.py
│       │   ├── 9_FactorLibrary.py
│       │   ├── 10_风控中心.py
│       │   ├── 11_参数优化.py
│       │   ├── 13_数据质量.py
│       │   ├── 14_设置.py
│       │   ├── 16_数据管理.py
│       │   ├── 19_行情仓库总览.py
│       │   ├── 20_端到端流水线.py
│       │   ├── 21_因子衰减监控.py
│       │   ├── 22_因子组合策略.py
│       │   ├── 23_知识库.py
│       │   └── 24_LLM策略挖掘.py
│       └── utils/             # 前端工具
│           ├── api_client.py  # REST 客户端
│           ├── charts.py      # Plotly 图表
│           ├── theme.py       # 主题/布局
│           ├── layout.py      # 布局组件
│           ├── constants.py   # 常量
│           ├── universes.py   # 标的池
│           └── ws_client.py   # WebSocket 客户端
│
├── tests/                     # 测试（~90 个测试文件，全离线 mock）
├── scripts/                   # 运维脚本
├── examples/                  # 示例脚本
├── db/init.sql                # TimescaleDB 初始化（超表+连续聚合+压缩）
├── config/                    # 配置文件（seat_sources 等）
├── docs/                      # 设计文档
├── docker-compose.yml         # 容器编排
├── Dockerfile                 # Python 3.13-slim 镜像
├── pyproject.toml             # 包定义 + 依赖
└── .env.example               # 环境变量模板
```

## 核心数据流

```
投资想法(idea)
  → AI Research（LLM 解析 → 因子规格 → 策略代码）
  → 因子挖掘（seed → co/ea/tot 搜索 → 去冗余）
  → 截面评估（IC/IR/衰减/单调性）
  → OOS 回测（Walk-Forward 防过拟合）
  → 复合 Alpha → 策略代码
  → 生命周期注册（RESEARCH → BACKTEST → PAPER → LIVE）
  → 模拟盘回放 → 实盘网关
```

## 关键约定

- **标的代码**: `vt_symbol = "{symbol}.{EXCHANGE}"`（如 `rb0.SHFE`、`600519.SSE`、`00700.HKEX`）
- **合约尺寸**: `core/contracts.default_size(vt)` 查表
- **数据源优先级**: 本地 Parquet (`QM_LOCAL_*_ROOT`) > 真实源 (akshare/efinance) > mock
- **本地行情仓库**: `data_cache/` 目录，Parquet 格式，首次拉取后秒级返回
- **时区**: 全部使用 `Asia/Shanghai`
- **LLM 默认**: `mock` provider，无需 API key 即可跑通全流程

## 环境变量（QM_ 前缀）

| 变量 | 说明 | 默认 |
|---|---|---|
| `QM_DB_URL` | TimescaleDB 连接串 | `postgresql://qm:quantmind@timescaledb:5432/quantmind` |
| `QM_DB_PASSWORD` | 数据库密码 | `quantmind` |
| `QM_REDIS_URL` | Redis 连接串 | `redis://redis:6379/0` |
| `QM_LLM_PROVIDER` | LLM 提供商 | `mock` |
| `QM_LLM_API_KEY` | LLM API Key | 空 |
| `QM_LLM_BASE_URL` | LLM Base URL | 空 |
| `QM_LLM_MODEL` | LLM 模型名 | 空 |
| `QM_LOG_LEVEL` | 日志级别 | `INFO` |
| `QM_LOCAL_DATA_ROOT` | 期货数据根目录 | 空 |
| `QM_LOCAL_STOCK_ROOT` | A股数据根目录 | 空 |
| `QM_LOCAL_HK_ROOT` | 港股数据根目录 | 空 |
| `QM_LOCAL_OPTION_ROOT` | 期权数据根目录 | 空 |
| `TAVILY_API_KEY` | Tavily 联网检索 | 空（离线兜底） |

## Docker 部署

### 端口映射（已避开其他项目冲突）

| 服务 | 容器端口 | 宿主机端口 | 说明 |
|---|---|---|---|
| TimescaleDB | 5432 | **5435** | 5432 被 investment-postgres 占用，5434 被 investment-pg 占用 |
| Redis | 6379 | **6380** | 避免与 redis-newapi 内部 6379 混淆 |
| API (FastAPI) | 8000 | **8001** | 8000 被 portainer 占用 |
| Web (Streamlit) | 8501 | **8502** | 预留安全端口 |

### 热更新机制

- 源码通过 Docker volume 挂载：`./quantmind:/app/quantmind`
- API: `uvicorn --reload` 监听文件变更自动重载（已验证 ✓）
- Web: `streamlit --server.runOnSave true` 自动刷新（已验证 ✓）
- 仅修改依赖时需 `docker compose build api web`
- 行情仓库持久化：`./data_cache:/app/data_cache`

### 已修复的兼容性问题

- `bcrypt<4.1`：passlib 与 bcrypt>=4.1 不兼容，已锁定版本
- `postgresql+asyncpg://`：DB URL 需 asyncpg 驱动前缀（非默认 psycopg2）
- Docker 构建代理：使用阿里云镜像（apt + pip）绕过代理问题

### 常用命令

```bash
# 启动全栈
docker compose up -d

# 查看日志
docker compose logs -f api
docker compose logs -f web

# 重建（依赖变更时）
docker compose build --no-cache \
  --build-arg http_proxy="" --build-arg https_proxy="" \
  --build-arg HTTP_PROXY="" --build-arg HTTPS_PROXY="" \
  --build-arg no_proxy="*" --build-arg NO_PROXY="*"
docker compose up -d

# 进入容器调试
docker compose exec api bash

# 运行测试
docker compose exec api python -m pytest

# 停止
docker compose down

# 停止并清除数据
docker compose down -v
```

### 访问地址

- API Swagger: http://localhost:8001/docs
- API Health: http://localhost:8001/health
- Web 控制台: http://localhost:8502
- WebSocket: ws://localhost:8001/ws

## 代码风格

- PEP 8，类型注解签名
- 优先 dataclass / immutable 模式
- 测试全离线 mock（`tests/helpers.py` 用 `MockFeed`）
- 无 linter/formatter 配置，保持与现有代码一致
- 中文注释和日志

## 注意事项

1. **Python 3.13 必须**（`requires-python = ">=3.13"`）
2. **mootdx 不要默认安装**（与 `httpx>=0.27` 冲突），A 股实时走 akshare 回退
3. **TimescaleDB 不可用时自动降级**为 InMemoryStore
4. **data_cache/ 是行情仓库**，首次请求从真实源拉取后落盘，后续秒级返回
5. **知识库**（knowledge/）持久化因子/策略/研究日志，跨运行保留
6. **生命周期状态机**：RESEARCH → BACKTEST → PAPER → LIVE，由 `paper/promotion.py` 管理
7. **.env 和 config/ 含密钥**，已在 .gitignore 中，绝不提交
