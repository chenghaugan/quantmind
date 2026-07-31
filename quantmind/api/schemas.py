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
    ir: Optional[float] = None
    ic_std: Optional[float] = None
    ic_positive_ratio: Optional[float] = None
    ic_decay: List[Optional[float]] = []
    top_quantile_return: Optional[float] = None
    long_short_return: Optional[float] = None
    n_samples: int = 0


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


class StrategyInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}


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
