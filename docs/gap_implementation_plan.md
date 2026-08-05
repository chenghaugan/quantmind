# QuantMind 差距补齐 · 实施方案（P1/P2）

> 依据：对 `QuantMind量化投资框架.md` 与 `quantmind/` 实际代码（173 文件）的差距审计。
> 基准：`QuantMind实现规划.md`（V3.1）收敛范围（国内多资产 + Web 统一入口）。
> 原则：只补与核心价值主张（Idea→策略→回测→模拟→晋升→实盘路由）直接相关的缺口；
> 避免在真实策略验证之前过度工程化。

> **进度更新（2026-08-05）**：✅ 方案 A（5 组件 **M1-M5 全量**）、方案 B（调度器 M1-M3）与方案 C（WS 前端 M1-M2）均已实现。
> M4（多标的组合聚合）+ M5（标的过滤）已补齐；方案 C 新增可复用 `WSClient` 并接入实时监控页。详见 §2-§4。实现清单见各节「✅ 已实现」标注。**



---

## 0. 结论速览

| 编号 | 主题 | 档位 | 优先级 |
|---|---|---|---|
| A | 5 组件模块化框架（增量） | 建议实现 | P1 |
| B | 任务调度器（APScheduler） | 建议实现 | P1 |
| C | WebSocket 前端消费落地 | 视需求 | P2 |
| — | LLM 真实接入 / 告警补强 / 因子发现增强 | 可缓 | P2 |
| — | 实盘网关（CTP/XTP/IB 真接口）| 明确搁置 | — |
| — | ML 训练管道（LightGBM/PyTorch/RL） | 不建议（现阶段） | — |
| — | MLflow/structlog/Prometheus/Grafana/DAG 编排 | 不建议（现阶段） | — |
| — | 多资产/多频率/高级订单回测 | 明确搁置 | — |
| — | 加密货币（CCXT）全栈 | 规划已排除 | — |

---

## 1. 背景：为什么只做这些

### 1.1 已实现且有据可查（不再重复投入）
- **事件引擎**：`core/engine.py` `EventEngine`（asyncio + 类型路由），已驱动回测/模拟/WebSocket。
- **生命周期 + 晋升闸门**：`paper/promotion.py` `LifecycleManager` + `PromotionGate`
  （min_sharpe / max_drawdown / min_paper_days / require_risk_review）。
- **AI 研究管道**：`ai/agent.py` `AutoResearchAgent`（想法解析→因子生成→代码生成→AST 沙箱→解释/事实表）。
- **AST 沙箱**：`ai/sandbox.py`（禁非白名单 import / exec / socket / __globals__ 等）。
- **可插拔 LLM**：`ai/provider.py`（Mock 默认 + OpenAI 兼容协议，DeepSeek/OpenAI/通义）。
- **回测/模拟/实盘路由**：`strategy/runners.py` `run_strategy(mode)` 同一代码切 3 模式。
- **回测引擎**：`backtest/engine.py`（事件驱动、成本表、保证金、平今、涨跌停剔除）。
- **walk-forward**：`backtest/walkforward.py` + `/walkforward`。
- **风控（已高度组件化）**：`risk/engine.py` `RiskEngine`（10+ 项事前检查）、
  `risk/portfolio.py` `PortfolioRiskEngine`、`risk/calendar.py`、`risk/turbulence.py`。
- **Turbulence 检测**：`risk/turbulence.py`（马氏距离 + 量化为位阈值）。
- **数据层**：`data/feed/*` 8 源 + Fallback 链；`data/store/{timescale,redis,fixture}`，
  TimescaleDB 真连接失败降级 InMemory（`api/app.py` lifespan 已验证）。
- **后端**：`api/app.py` 30+ REST + `/ws` WebSocket 广播 + 14 个 Service。
- **前端**：`web/` 16 个 Streamlit 页面，全部经 `web/utils/api_client.py` 调后端 REST。
- **因子库**：technical + wq + alpha101(~30) + alpha191(~10) + seat_futures(F1-F8) + alpha_cs(42)。

### 1.2 差距审计要点（据此定档）
| 缺口 | 证据 | 定档理由 |
|---|---|---|
| **5 组件模块化缺失** | 全库搜索 `class AlphaModel/PortfolioModel/ExecutionModel/UniverseModel` **零命中**；策略为单体 `CtaTemplate`，`on_bar()` 内联 alpha+position+risk | 框架最重要架构模式；**但 risk 已组件化、portfolio 有雏形 → 可增量补，非推翻** |
| **无调度器** | `APScheduler` 全库零命中 | 无它"模拟盘跑 7 天 / 数据同步 / 健康检查"无法自动化 |
| **WS 前端未消费** | 后端 `/ws` 广播已实现，但 16 页基本走 REST 轮询 | 若实时监控需主动推送再补；否则轮询够用 |
| **无 ML 训练管道** | pyproject 无 torch/lightgbm；`ai/` 只产出因子规格+代码 | 规则化信号已跑通；投入大，非刚需 → **搁置** |
| **无可观测全套** | MLflow/structlog/prometheus/APScheduler 均零命中 | 单机单开发者阶段过度工程 → **搁置** |
| **实盘网关为桩** | `live/*_gateway.py` 日志桩 | Phase 7 未来项，需凭证 → **搁置** |

---

## 2. 方案 A：5 组件模块化框架（增量）· P1

> ✅ **已实现（M1-M5 全量）**：`quantmind/strategy/components/`（base.py 5 Protocol + AlphaSignal；alpha.py MultiFactorAlpha/MomentumAlpha/**MultiFactorMultiSymbolAlpha**；
> portfolio.py IdentityPortfolio/**EqualWeightPortfolio**；risk.py NullRisk/RiskGateModel；execution.py TargetExecution；**universe.py AllUniverse/RuleUniverse**；composable.py ComposableStrategy）。
> 已登记 `/strategies` 与 CLI；新增 `tests/test_components.py`（10 用例）+ `tests/test_components_multi.py`（10 用例，M4/M5）→ 全量 282 项通过，
> 其中 `ComposableStrategy(alpha=MultiFactorAlpha)` 回测与 `MultiFactorStrategy` 完全一致（回归门槛达成）。
> **M4（多标的组合）**：`PortfolioModel.apply_all(signals, universe)` + `EqualWeightPortfolio` 等权聚合；`BacktestEngine.set_universe` +
> `run_strategy_multi(strategy_class, vt_symbols, data, setting, ...)` 多标的多因子组合回测入口；主标的重平衡驱动逐标的 Risk+Execution。
> **M5（标的过滤）**：`UniverseModel.select` + `RuleUniverse`（最小历史/流动性），`ComposableStrategy.on_init` 选出可交易子集。
> **默认装配退化兼容**：单标的下 ComposableStrategy 行为与之前完全一致（回归门槛通过）。

### 2.1 目标
在不破坏现有 `CtaTemplate` 模板策略的前提下，引入 Lean 风格的 **5 组件组合模型**，
让"信号生成 / 组合构建 / 风控 / 执行"可独立替换与复用，支撑未来的多策略组合与单组件测试。

### 2.2 现状基础（无需新造的部分）
- `risk/engine.py` `RiskEngine`（前置风控）→ 可作为 **RiskModel** 组件。
- `risk/portfolio.py` `PortfolioRiskEngine` → 组合级风控。
- `strategy/context.py` `StrategyContext` + `set_target` → 已实现"目标仓位"解耦。
- `backtest/engine.py` 的 `get_history / get_position / send_order` → 已具备组合所需的 context 接口。

### 2.3 新增结构（建议文件 `quantmind/strategy/components/`）

```
quantmind/strategy/components/
├── __init__.py
├── base.py        # 5 个 Protocol: UniverseModel/AlphaModel/PortfolioModel/RiskModel/ExecutionModel
├── alpha.py       # AlphaModel 实现：从因子规格/规则生成目标仓位序列
├── portfolio.py   # PortfolioModel：把多标的 alpha 信号聚合成组合目标权重
├── risk.py        # RiskModel：包装现有 RiskEngine/PortfolioRiskEngine 为组件接口
├── execution.py   # ExecutionModel：set_target → OrderRequest 的下单抽象
├── universe.py    # UniverseModel：标的过滤/纳入逻辑
└── composable.py  # ComposableStrategy：把 5 组件装配成 CtaTemplate 子类
```

### 2.4 装配方式（不破坏现有模板）
```python
# composable.py
class ComposableStrategy(CtaTemplate):
    """用 5 个可组合组件装配成的策略。"""
    def __init__(self, context, setting=None):
        super().__init__(context, setting)
        self.alpha = setting.get("alpha")        # AlphaModel
        self.portfolio = setting.get("portfolio")  # PortfolioModel
        self.risk = setting.get("risk")          # RiskModel（可空）
        self.execution = setting.get("execution")  # ExecutionModel

    def on_bar(self, bar):
        signals = self.alpha.on_bar(bar)          # 信号
        target = self.portfolio.combine(signals)  # 目标仓位
        if self.risk:
            target = self.risk.apply(bar, target) # 风控过滤
        self.execution.set_target(bar.vt_symbol, target)
```

> 现有 `MultiFactorStrategy / DualMaStrategy / PairTradingStrategy` 保持不变；
> 新组件模型与原模板可共存，`/strategies` 与 CLI 仅需登记 `ComposableStrategy` 即可。

### 2.5 里程碑
- **M1**：定义 5 个 Protocol + `ComposableStrategy` 骨架（纯接口，跑通空跑）。
- **M2**：实现 `ExecutionModel`（复用 `context.set_target`）与 `RiskModel`（包装 `RiskEngine`）。
- **M3**：把 `MultiFactorStrategy` 的信号生成重构为 `AlphaModel`（行为不变，回归测试通过）。
- **M4**：`PortfolioModel` 支持多标的聚合（激活 paper 多策略 `contexts`）。
- **M5**：`UniverseModel` 按规则过滤标的。
- **验收**：`pytest` 新增 `test_components.py`；`ComposableStrategy(alpha=MultiFactorAlpha)` 回测结果
  与现有 `MultiFactorStrategy` 一致（回归门槛）。

---

## 3. 方案 B：任务调度器（APScheduler）· P1

> ✅ **已实现（M1-M3）**：`quantmind/api/scheduler.py`（QuantMindScheduler 封装 + build_scheduler 内置任务：数据同步/风控日切/健康检查，
> required=False 降级 no-op）；`api/app.py` lifespan 挂载 start/stop；`/scheduler`、`/scheduler/start`、`/scheduler/stop` REST 已加；
> `pyproject.toml` 增 `apscheduler>=3.10`（可选依赖，未安装可降级启动）。新增 `tests/test_scheduler.py`。

### 3.1 目标
在 `api` 进程中挂载 APScheduler，把"数据同步 / 定期回测 / 健康检查 / 策略重跑"做成可注册的
周期任务，支撑"模拟盘长期运行 + 数据新鲜度监控"。

### 3.2 结构（建议文件 `quantmind/api/scheduler.py`）
```python
# scheduler.py — APScheduler 封装
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class QuantMindScheduler:
    def __init__(self): self._sched = AsyncIOScheduler()

    def register(self, name, fn, cron=None, interval=None, max_instances=1): ...

    async def start(self): self._sched.start()
    async def shutdown(self): self._sched.shutdown()

_JOB_DEFS = [
    # (name, cron, fn) — 数据同步 / 风控日切 / 健康检查
]
```

### 3.3 首批任务（复用现有模块，非新写）
| 任务 | 复用 | cron |
|---|---|---|
| 数据增量同步 | `cli._fetch` 逻辑 / `DataManager` | 交易日 15:30 |
| 风控日切 `reset_day` | `RiskEngine.reset_day` | 每日 00:00 |
| 健康检查（引擎/DB/Redis） | `/health` 逻辑 | 每 5 分钟 |
| 待定：周期策略重跑晋升检查 | `LifecycleManager` | 每日 17:00 |

### 3.4 接入点
- 在 `api/app.py` lifespan 中 `start()`/`shutdown()`。
- `pyproject.toml` 增 `apscheduler>=3.10`。

### 3.5 里程碑
- **M1**：`QuantMindScheduler` 封装 + `api/app.py` 挂载。
- **M2**：注册数据同步 + 风控日切 + 健康检查三个内置任务。
- **M3**：新增 `/scheduler` REST（列出/启停任务）。
- **验收**：`docker compose up api` 后日志出现周期任务触发；`/scheduler` 可查询。

---

## 4. 方案 C：WebSocket 前端消费（可选）· P2

> ✅ **已实现（M1-M2）**：`quantmind/web/utils/ws_client.py`（`WSClient` 后台线程 asyncio + 指数退避自动重连
> 1s→2s→…→30s + `_schedule_close`/`_schedule_cancel` 线程安全停止；`connect_ws`、`WS_URL` 入口），
> 用已安装的 `websockets` 库作客户端（不新增 `websocket-client` 依赖）。`7_实时监控.py` 页已接入事件流，
> 新增结构化仪表（账户/行情/信号/持仓/成交/风控卡片）。新增 `tests/test_ws_client.py`（3 用例：连接收 hello / 自动重连 / stop 终止线程）。
> 后端 `/ws` 广播已就绪，**零后端改动**。

### 4.1 若实施
- `web/utils/` 新增 `ws_client.py`：用 `websockets`/`st.experimental_connection` 订阅 `/ws`。
- 在 `7_实时监控` 页接入：收到 bar/signal/position/risk 事件即时刷新仪表，避免 F5 轮询。
- 后端已具备广播（`_broadcast` → `/ws`），**前端接入即可，零后端改动**。

### 4.2 里程碑
- **M1**：`ws_client` 连接 + 断线重连。✅
- **M2**：实时监控页接入事件流。✅
- **验收**：启动 backtest/paper 模拟盘，监控面板无需刷新即出现新信号/成交。✅ **已端到端验证**：
  `tests/test_ws_e2e.py`（新增）用 TestClient 走真实 lifespan，连 `/ws` 拿 hello → 触发真实回测 → 断言无需刷新即收到
  `eSignal`（vt_symbol=rb0.SHFE、target∈{-1,0,1}）/`eTrade`/`ePosition` 实时事件流。配合 `test_ws_client.py`（连接/重连/stop）+ 页面 AppTest（0 异常）
  与仪表消费逻辑真实形状断言，三层链路（后端→WS→仪表）全部闭环。⚠️ 注：独立 `WSClient` 走真实 TCP（ws://127.0.0.1:8000），
  无法直接连内存型 TestClient；端到端事件内容用 TestClient `websocket_connect` 验证，WSClient 自身的连接/重连/终止由 test_ws_client.py 覆盖。

---

## 5. 明确搁置 / 不做（避免过度工程）

| 项 | 原因 | 触发再做的条件 |
|---|---|---|
| 实盘网关真接口 | 需 simnow/券商凭证，当前离实盘远 | 模拟盘跑通且有实盘需求 |
| ML 训练管道 | 规则化信号已跑通；投入大 | 出现预测型策略需求 |
| MLflow/structlog/Prometheus/Grafana | 单机阶段过度工程 | 多策略并行 + 要 SLA |
| Prefect/DAG 编排 | 研究管道线性，无复杂依赖 | 团队协作/复杂分支实验 |
| 多资产/多频率/高级订单回测 | 实盘执行才需要 | 进入实盘阶段 |
| 加密货币全栈 | 规划明确排除 | 决策改变 |

---

## 6. 排期建议（按 P1 优先）

| 周期 | 工作 |
|---|---|
| 第 1-2 周 | **方案 B**：调度器（独立、见效快、风险低） |
| 第 2-5 周 | **方案 A**：5 组件 M1-M4（增量，回归门槛保障） |
| 第 5 周后 | **方案 C**：按需评估是否接入 WS 前端 |
| 持续 | 用已有能力跑真实策略的模拟盘，验证 alpha → 据实际短板再定后续优先级 |

**先做 B（调度器）还是先做 A（5 组件）？**
- 若你更看重"让模拟盘自动化长期跑起来" → 先 B。
- 若你更看重为"多策略/组合风控"打地基 → 先 A。
- 两者不冲突，可 B 先行（独立、半月内见效）。

---

## 7. 待确认项

1. ~~**起始优先级**：先 B（调度器）还是先 A（5 组件）？~~ ✅ 二者均已实现。
2. ~~**方案 C**：当前是否确有"实时监控主动推送"需求？（默认搁置）~~ ✅ 已实现 M1-M2（WS 前端消费落地）。
3. ~~**5 组件的范围**：只做"信号/组合/风控/执行"四件，`UniverseModel` 是否本期做？~~ ✅ M4（组合）+ M5（Universe 过滤）已补齐。
4. ~~**调度器首批任务清单**：是否认可"数据同步 / 风控日切 / 健康检查"三件套？~~ ✅ 已实现内置三件套。
