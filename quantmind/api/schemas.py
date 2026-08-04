"""API 请求/响应模型（扩展：研究 / 因子 / 回测 / 策略 / 订单 / 生命周期）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---- 历史数据 ----
class HistoryQuery(BaseModel):
    symbol: str
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None


class DataDownloadRequest(BaseModel):
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


# ---- 因子 ----
class FactorRequest(BaseModel):
    symbol: str
    exchange: str = "SHFE"
    interval: str = "1d"
    factor: str = "momentum_20"
    expression: Optional[str] = None
    window: int = 20
    forward_periods: int = 1


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


class StrategyInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}


# ---- Walk-Forward 滚动验证 ----
class WalkForwardRequest(BaseModel):
    strategy: str = "dual_ma"
    symbol: str = "rb0"
    exchange: str = "SHFE"
    train_window: int = 250
    test_window: int = 60
    step: Optional[int] = None
    setting: Dict[str, Any] = {}
    capital: float = 1_000_000.0
    cost: bool = False


# ---- 手动下单 ----
class OrderRequestSchema(BaseModel):
    vt_symbol: str
    direction: str                # 多 / 空
    offset: str = "开"            # 开 / 平 / 平今 / 平昨
    volume: float
    price: float = 0.0


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


# ---- 截面（多标的面板）因子研究 ----
class CrossSectionRequest(BaseModel):
    symbols: List[str] = ["rb0", "hc0", "i0", "j0", "jm0"]
    exchange: str = "SHFE"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    factor: str = "alpha002"
    forward_periods: int = 1
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0
    backtest: bool = True         # 是否同时跑多空组合回测


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
