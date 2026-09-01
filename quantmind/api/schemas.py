"""API 请求/响应模型（扩展：研究 / 因子 / 回测 / 策略 / 订单 / 生命周期）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---- 历史数据 ----
class HistoryQuery(BaseModel):
    symbol: str
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None


class BarOut(BaseModel):
    symbol: str
    exchange: str
    datetime: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---- 研究（AI） ----
class ResearchRequest(BaseModel):
    idea: str
    asset_class: Optional[str] = None


class ResearchResult(BaseModel):
    idea: str
    asset_class: str
    hypothesis: str
    suggested_factors: List[str]
    risk_notes: List[str] = []
    generated_factors: List[dict] = []
    code_safe: bool = False
    code_errors: List[str] = []
    provenance: Optional[Provenance] = None


class Provenance(BaseModel):
    """研究结果溯源信息（对标 Vibe-Trading 的 evidence chain）。"""
    data_sources: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    evidence_chain: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    research_log: List[Dict[str, Any]] = []
    generated_at: Optional[str] = None


# ---- 因子 ----
class FactorRequest(BaseModel):
    symbol: str = ""              # 单标的模式的主标的（多标的模式下可忽略）
    symbols: List[str] = []        # 多标的（截面）模式：标的列表，为空则回落单标的
    exchange: str = "SHFE"
    interval: str = "1d"
    factor: str = "momentum_20"
    expression: Optional[str] = None
    window: int = 20
    forward_periods: int = 1
    n_groups: int = 5             # 多标的：截面分组数
    long_short: bool = True        # 多标的：是否多空组合
    cost_rate: float = 0.0         # 多标的：单边成本率
    start: Optional[str] = None
    end: Optional[str] = None


class FactorResult(BaseModel):
    factor_name: str
    ic_mean: Optional[float] = None
    ic_pearson: Optional[float] = None
    ir: Optional[float] = None
    ic_std: Optional[float] = None
    ic_positive_ratio: Optional[float] = None
    ic_decay: List[Optional[float]] = []
    ic_decay_half_life: Optional[float] = None
    ic_ci_low: Optional[float] = None
    ic_ci_high: Optional[float] = None
    top_quantile_return: Optional[float] = None
    long_short_return: Optional[float] = None
    monotonicity_5: Optional[float] = None
    monotonicity_10: Optional[float] = None
    turnover_annual: Optional[float] = None
    ls_portfolio_return: Optional[float] = None
    ls_portfolio_sharpe: Optional[float] = None
    ls_portfolio_mdd: Optional[float] = None
    composite_score: Optional[float] = None
    n_samples: int = 0
    note: str = ""
    error: Optional[str] = None


# ---- 回测 / 模拟 / 实盘 ----
class BacktestRequest(BaseModel):
    strategy: str = "dual_ma"      # dual_ma | multifactor
    symbol: str = "rb0"
    exchange: str = "SHFE"
    mode: str = "backtest"         # backtest | paper | live
    setting: Dict[str, Any] = {}
    gateway: str = "ctp"
    capital: float = 1_000_000.0
    commission: float = 0.0002
    sizes: Dict[str, float] = {}
    cost: bool = False            # 启用真实成本模型（按品种差异化费率/平今/印花税/保证金）
    start: Optional[str] = None   # 回测起始日期 YYYY-MM-DD（None=自动）
    end: Optional[str] = None     # 回测结束日期 YYYY-MM-DD（None=自动）


class OptimizationConfig(BaseModel):
    """网格参数优化配置：IS/OOS 切分 + 高原检验 + Deflated Sharpe 三防线。"""

    enabled: bool = False
    # 用裸 list 保住元素原始类型（int/float），避免 pydantic 强转 float 破坏策略参数语义
    param_grid: Dict[str, list] = {}   # {"window": [10,20,30]}；空 → 用策略内 PARAM_GRID
    is_ratio: float = Field(0.7, ge=0.1, le=0.9)   # 样本内占比（split_is_oos 契约）
    max_combos: int = Field(60, ge=1)         # 组合数硬上限（防组合爆炸）
    top_k: int = Field(5, ge=1)               # 进入 OOS 验证的组合数
    min_trades: int = Field(10, ge=0)         # 组合最少成交笔数（IS 段淘汰）
    plateau_ratio: float = Field(0.6, ge=0.0, le=1.0)  # 参数高原：邻域中位/最优 ≥ 阈值才可信
    use_dsr: bool = True                      # 入库判据用 Deflated Sharpe（替代原始 Sharpe）
    warmup_bars: int = Field(120, ge=0)       # OOS 回测预热根数（从 IS 尾部借）


class StrategyValidateRequest(BaseModel):
    """策略思想测试：策略思路 → LLM 预编程（或用户审定代码）→ 真实数据回测 → 门槛判定 →（可选）入有效策略库。

    策略来源：
      - ``code`` 非空：直接使用用户在对话式编程阶段审定的代码（仍过沙箱校验）。
    ``code`` 为空：把 ``idea``（策略思想，如布林带回穿规则全文）交给
        LLM 预编程为 CtaTemplate 策略代码，AST 沙箱校验后回测。

    支持多品种：``symbols`` 列表逐品种独立回测对比，每个品种各自门槛判定，
    达标品种自动写入 lifecycle（有效策略库）。
    """

    idea: str = ""                       # 策略思想/投资想法（主输入）
    code: str = ""                       # 用户审定的策略代码（非空时跳过 LLM，直接注册回测）
    symbols: List[str] = ["IC0"]         # 多品种：逐品种回测对比
    exchange: str = "CFFEX"              # 默认交易所（单品种快捷用）
    interval: str = "1d"                 # 1d/1h/30m/15m/5m/1m（旧字段，向后兼容）
    intervals: Optional[List[str]] = None  # 多周期回测：非空时逐周期回测，优先于 interval
    start: Optional[str] = None           # YYYY-MM-DD；None=全部历史
    end: Optional[str] = None
    setting: Dict[str, Any] = {}          # 预置模板参数；空 → 默认参数
    cost: bool = False                    # 是否启用真实成本模型
    # 门槛判定与自动入库（默认关闭，向后兼容）
    gate: Optional[dict] = None           # {min_sharpe, min_drawdown, ...}；None=不判定
    promote: bool = False                 # 判定 verified 后自动写入 lifecycle（有效策略库）
    optimization: Optional[OptimizationConfig] = None  # 参数优化（防过拟合三防线）


class StrategyDraftMessage(BaseModel):
    """LLM 策略编程对话的单条消息。"""

    role: str = "user"                   # "user"（思想/修改意见）| "assistant"（代码）
    content: str


class StrategyDraftRequest(BaseModel):
    """LLM 策略代码草稿（对话式编程）：无状态多轮，历史由前端每次携带。"""

    idea: str = ""                       # 首轮：策略思想；后续轮可空（修改意见在 history 末尾）
    history: List[StrategyDraftMessage] = []  # 完整对话史（截断由前端负责）


class StrategyRegisterRequest(BaseModel):
    """注册 AI 生成策略（端到端流水线产出的策略代码）。"""

    name: str
    code: str
    idea: str = ""
    run_id: str = ""            # 来源 e2e run_id（策略来源，可缺省）
    composite_fwd_ic: Optional[float] = None  # 来源 run 的 composite_fwd_ic（可缺省）


class StrategyInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}


class PaperRunRequest(BaseModel):
    """模拟盘实跑请求：把已注册策略部署到 PaperEngine 做历史回放，晋升到 PAPER。"""

    strategy: str                       # 已注册策略名（或内置策略名）
    symbol: str = "rb0"
    exchange: str = "SHFE"
    setting: Dict[str, Any] = {}
    capital: float = Field(1_000_000.0, gt=0)
    commission: float = Field(0.0002, ge=0)
    days: int = Field(400, ge=1)          # 回放窗口（自然日回溯取数）


# ---- Walk-Forward 滚动验证 ----
class WalkForwardRequest(BaseModel):
    strategy: str = "dual_ma"
    symbol: str = "rb0"
    exchange: str = "SHFE"
    train_window: int = Field(250, ge=1)
    test_window: int = Field(60, ge=1)
    step: Optional[int] = Field(None, ge=1)
    setting: Dict[str, Any] = {}
    capital: float = Field(1_000_000.0, gt=0)
    cost: bool = False


# ---- 手动下单 ----
class OrderRequestSchema(BaseModel):
    vt_symbol: str
    direction: str                # 多 / 空
    offset: str = "开"            # 开 / 平 / 平今 / 平昨
    volume: float = Field(..., gt=0)
    price: float = Field(0.0, ge=0)


# ---- 生命周期晋升 ----
class LifecycleRequest(BaseModel):
    strategy_id: str
    to: str                       # BACKTEST / PAPER / APPROVED / LIVE
    metrics: Dict[str, float] = {}
    note: str = ""


# ---- 参数优化（网格搜索 / Optuna 贝叶斯） ----
class OptimizeRequest(BaseModel):
    strategy: str = "dual_ma"
    symbol: str = "rb0"
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    #: 参数名 -> 候选值列表，例如 ``{"fast": [3,5,10], "slow": [20,40]}``（网格搜索用）
    param_space: Dict[str, List[Any]] = {}
    #: 优化方法：grid（网格）| optuna（贝叶斯）
    method: str = "grid"
    #: Optuna 专用：参数名 -> [low, high, step]，例如 ``{"fast": [3, 15, 2], "slow": [20, 60, 5]}``
    param_ranges: Dict[str, List[Any]] = {}
    n_trials: int = 30             # Optuna 试验次数
    metric: str = "sharpe"        # sharpe | total_return | calmar | max_drawdown
    capital: float = 1_000_000.0
    max_combos: int = 200         # 组合数上限，防止 Web 端跑爆


# ---- 风控 ----
class RiskCheckRequest(BaseModel):
    profile: str = "default"      # default | conservative | unlimited
    vt_symbol: str = "rb0.SHFE"
    direction: str = "多"         # 多 / 空
    offset: str = "开"            # 开 / 平 / 平今 / 平昨
    volume: float = 1.0
    price: float = 0.0
    last_price: float = 3500.0
    equity: float = 1_000_000.0
    margin_used: float = 0.0
    position_volume: float = 0.0  # 当前净持仓（正=多、负=空）
    check_session: bool = False   # Web 试算默认不校验交易时段
    #: 覆盖限额字段，例如 ``{"max_order_volume": 50}``
    overrides: Dict[str, Any] = {}


class RiskDecisionOut(BaseModel):
    passed: bool
    code: str
    reason: str = ""
    vt_symbol: str = ""


# ---- 数据质量 ----
class QualityOut(BaseModel):
    symbol: str
    total: int = 0
    gaps: int = 0
    outliers: int = 0
    rollover_jumps: int = 0
    last_ts: Optional[str] = None
    stale: bool = False
    issues: List[str] = []
    score: float = 100.0


# ---- 席位因子 ----
class SeatFactorRequest(BaseModel):
    symbol: str                   # 品种代码，如 RB / CU / AG
    exchange: str = "SHFE"        # 关联价格所用交易所（用于算 IC）
    interval: str = "1d"          # 价格数据周期
    seat_data_root: str           # 席位 CSV 根目录（含 <SYMBOL>/long/short/volume_ranking.csv）
    factor: str = "F7_net_zscore" # F1-F8
    aggregate: bool = True        # 多席位等权聚合
    forward_periods: int = 1
    n_groups: int = 5
    long_short: bool = True


class SeatFactorResult(BaseModel):
    factor: str
    n_seats: int = 0
    n_dates: int = 0
    ic_mean: Optional[float] = None
    ir: Optional[float] = None
    ic_positive_ratio: Optional[float] = None
    top_quantile_return: Optional[float] = None
    long_short_return: Optional[float] = None
    composite_score: Optional[float] = None
    error: Optional[str] = None


# ---- 表达式截面评估 / 因子迭代搜索（P0 State / P1 CoT）----
class ExprEvalRequest(BaseModel):
    """表达式截面评估请求（多标的面板，index=日期 × columns=标的）。"""

    expression: str                        # 因子表达式（函数式 mean(close,20) 或 Qlib 式 Mean($close,20)）
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    forward_periods: int = 1
    market: str = ""


class ExprEvalBatchRequest(BaseModel):
    """批量表达式评估请求。"""

    expressions: List[str] = ["Mean($close, 5)", "Rank($close, 20)"]
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    forward_periods: int = 1
    market: str = ""


class FactorSearchRequest(BaseModel):
    """因子迭代搜索请求（co / ea / tot 三种算法）。"""

    seed: str = "Mean($close, 20)"         # 初始因子表达式
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    algo: str = "co"                        # co | ea | tot（缺失默认 CoT 链式精炼）
    rounds: int = 6                          # co/ea/tot 的迭代深度（generations/depth）
    forward_periods: int = 1
    market: str = ""
    # 可选独立验证期（防泄漏）
    val_symbols: Optional[List[str]] = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None


class FactorDedupRequest(BaseModel):
    """因子相关性聚类去冗余请求。"""

    expressions: List[str] = ["Mean($close, 5)", "Rank($close, 20)"]
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    correlation_threshold: float = 0.7     # 并簇相关阈值 [0,1]
    min_abs_metric: float = 0.0            # |rank_ic| 底线（低于视为噪声丢弃）
    forward_periods: int = 1
    market: str = ""
    compute_ic: bool = True                # True 用真实 rank_ic 排序；False 用复杂度


class ExpressionBacktestRequest(BaseModel):
    """对挖掘出的 DSL 因子表达式直接做截面多空组合回测。"""

    expression: str = "delta(close, 20)"
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    forward_periods: int = 1
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0


class FactorPipelineRequest(BaseModel):
    """端到端因子挖掘流水线请求（挖掘 → 去冗余 → 逐因子OOS回测 → 复合组合）。"""

    seeds: List[str] = ["delta(close, 5)", "ts_zscore(close, 20)", "rank(close, 10)"]
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    algo: str = "co"                        # co | ea | tot
    rounds: int = 3                         # 每 seed 迭代深度
    forward_periods: int = 1
    market: str = ""
    # 去冗余
    dedup_threshold: float = 0.7            # 并簇相关阈值 [0,1]
    min_abs_ic: float = 0.03                # |rank_ic| 底线（低于 0.03 视为噪声丢弃）
    # 防泄漏切分
    train_frac: float = 0.6
    val_frac: float = 0.2
    # 复合组合
    run_composite: bool = True
    composite_scheme: str = "icir"          # equal | icir | inv_var | min_var
    # 回测
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0
    # 因子侧宽松净成本/换手闸门
    net_gate: bool = False
    max_turnover: float = 0.0
    min_net_sharpe: float = 0.0
    max_candidates: int = 8


# ---- 端到端编排（AI 证据 → 挖掘 → 复合 → 策略代码）+ 自动沉淀知识库 ----
class FactorE2ERequest(BaseModel):
    """端到端因子研究请求（复用 orchestrator.run_e2e，可选沉淀到知识库）。

    对应 :class:`quantmind.research.orchestrator.E2EConfig` 全字段，
    附加面板/标的构造参数与知识库沉淀开关（``ingest_knowledge``）。
    """

    idea: str
    asset_class: str = "期货"
    seeds: Optional[List[str]] = None       # 用户额外种子；None → 用 AI 证据阶段产出
    symbols: List[str] = ["IF0", "rb0", "hc0", "i0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    # 因子挖掘（透传 E2EConfig）
    algo: str = "co"
    rounds: int = 3
    forward_periods: int = 1
    market: str = ""
    train_frac: float = 0.6
    val_frac: float = 0.2
    dedup_threshold: float = 0.7
    min_abs_ic: float = 0.03
    run_composite: bool = True
    composite_scheme: str = "icir"
    composite_standardize: str = "zscore"
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0
    net_gate: bool = False
    max_turnover: float = 0.0
    min_net_sharpe: float = 0.0
    max_candidates: int = 8
    # AI 证据研究阶段（A 线）
    verify_threshold: float = 0.02
    run_search: bool = False
    max_rounds: int = 2
    # 领域知识增强层
    use_knowledge: bool = True            # 是否在 idea→因子 前注入领域知识
    web_fallback: bool = True             # 库内方法论命中不足时是否联网补充
    # 方法论知识层：当 e2e 返回 needs_input 时，用户补充的方法论说明（提交后先入库再重跑）
    methodology_input: Optional[str] = None
    # 是否把结果沉淀进知识库
    ingest_knowledge: bool = True
    # 门槛判定与自动入库（升级：端到端策略挖掘闭环）
    # gate: 门槛配置 {min_sharpe, min_drawdown, min_calmar, min_win_rate, min_paper_days}；
    #       None → 不启用门槛判定（保持原有行为）。
    gate: Optional[dict] = None
    # promote: 门槛判定为 verified 时，自动注册到有效策略库（lifecycle 表，6_生命周期页展示）。
    promote: bool = False


# ---- 知识库（knowledge） ----
class KnowledgeIngestRequest(BaseModel):
    """手动向知识库写入一条记录。

    ``kind``: ``factor`` | ``strategy`` | ``research_log``。
    ``payload`` 为对应 kind 的字段 dict（见 KnowledgeService.ingest）。
    """

    kind: str
    payload: Dict[str, Any]


class KnowledgeSearchRequest(BaseModel):
    """知识库轻量关键词检索请求。"""

    query: str
    top_k: int = 10
    kind: Optional[str] = None          # factor | strategy | research_log | None=全部


# ---- LLM 策略挖掘 ----
class StrategyMiningRequest(BaseModel):
    """LLM 策略挖掘请求：从因子库选择因子，LLM 设计策略规格。"""

    factors: List[Dict[str, Any]]       # 因子列表（含 name/kind/window/ic_mean/icir/sharpe 等）
    constraint: Optional[str] = None    # 用户约束（如"偏动量"、"低换手"）
    template_preference: Optional[str] = None  # 模板偏好（dual_ma/multifactor/vol_target/pair_trading）
    symbol: str = "rb0"
    exchange: str = "SHFE"


class AutoBacktestRequest(BaseModel):
    """自动回测请求：对生成的策略规格执行自动回测循环。"""

    spec: Dict[str, Any]                # StrategySpec 字典
    strategy_id: Optional[str] = None   # 生命周期策略 ID
    max_iterations: int = 3             # 最大迭代次数
    min_sharpe: float = 0.5             # 最低 Sharpe 闸门
    max_drawdown: float = -0.30         # 最大回撤下限
    max_cost_ratio: float = 0.6         # 成本/净收益上限（高换手拦截），0=关闭
    compare_zero_cost: bool = True      # 跑一次零成本对照，量化成本拖累
    cost: Optional[Union[bool, Dict[str, Any]]] = None  # 差异化成本：True=默认表/dict=自定义/False=零成本/None=按 QM_BACKTEST_COST

