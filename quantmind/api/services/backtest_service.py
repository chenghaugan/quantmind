"""BacktestService: 回测 / WalkForward / 策略清单"""
import asyncio
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ...core.constant import Exchange, Interval
from ...core.contracts import default_size
from ...core.engine import EventEngine
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...strategy import (
    run_strategy,
    MultiFactorStrategy,
    DualMaStrategy,
    VolTargetStrategy,
    PairTradingStrategy,
)
from ...backtest.walkforward import walk_forward
from ..schemas import BacktestRequest, WalkForwardRequest, StrategyInfo


_logger = logging.getLogger("quantmind.api")


_STRATEGY_MAP = {
    "dual_ma": DualMaStrategy,
    "multifactor": MultiFactorStrategy,
    "vol_target": VolTargetStrategy,
    "pair": PairTradingStrategy,
}


def _sanitize(o: Any) -> Any:
    """递归把非有限 float 转为 None，避免 JSON 序列化抛错"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    return o


class BacktestService:
    def __init__(self, dm: DataManager, ee: EventEngine):
        self.dm = dm
        self.ee = ee

    @staticmethod
    def list_strategies() -> List[StrategyInfo]:
        return [
            StrategyInfo(
                name="dual_ma",
                description="双均线趋势/动量策略",
                parameters={"fast": 5, "slow": 20, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="multifactor",
                description="多因子组合策略（动量+均值回复+波动率）",
                parameters={"specs": "see research", "threshold": 0.3, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="vol_target",
                description="全天候风格：波动率目标+动量过滤(单标的风险平价)",
                parameters={"lookback": 20, "target_vol": 0.20, "momentum_win": 60, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="pair",
                description="配对交易：价差合成标的 z-score 均值回复",
                parameters={"window": 30, "entry_z": 1.5, "exit_z": 0.3, "size": 1, "max_pos": 1.0},
            ),
        ]

    async def run_backtest(self, req: BacktestRequest) -> Dict[str, Any]:
        strat_class = _STRATEGY_MAP.get(req.strategy, MultiFactorStrategy)
        vt = f"{req.symbol}.{req.exchange.upper()}"
        
        try:
            bars = await self.dm.get_bar_data(
                HistoryRequest(
                    symbol=req.symbol,
                    exchange=Exchange(req.exchange.upper()),
                    interval=Interval("1d"),
                )
            )
            if not bars:
                _logger.warning(f"回测无数据: {req.symbol}.{req.exchange}")
                return {"error": "无数据"}
            sizes = dict(req.sizes) or {vt: default_size(vt)}
            result = await asyncio.to_thread(
                run_strategy,
                req.mode,
                strat_class,
                vt,
                dict(req.setting),
                bars,
                self.ee,
                sizes,
                req.gateway,
                None,
                req.cost,
            )
            _logger.info(f"回测完成: {req.strategy} on {vt}, {len(bars)} bars")
            return _sanitize(result)
        except Exception as e:
            _logger.error(f"回测失败: {req.strategy} on {vt} - {str(e)}", exc_info=True)
            return {"error": f"回测失败: {str(e)}"}

    async def run_walkforward(self, req: WalkForwardRequest) -> Dict[str, Any]:
        """Walk-Forward 滚动样本外验证"""
        strat_class = _STRATEGY_MAP.get(req.strategy, MultiFactorStrategy)
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 自动计算需要的历史数据长度
        step = req.step or req.test_window
        min_bars_needed = req.train_window + req.test_window * 2
        days_needed = max(min_bars_needed + 50, 500)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_needed)

        bars = await self.dm.get_bar_data(
            HistoryRequest(
                symbol=req.symbol,
                exchange=Exchange(req.exchange.upper()),
                interval=Interval("1d"),
                start=start_date,
                end=end_date,
            )
        )
        if not bars:
            return {"error": "无数据"}

        min_required = req.train_window + req.test_window
        if len(bars) < min_required:
            return {
                "error": f"样本不足：需要至少 {min_required} 根，当前仅 {len(bars)} 根。"
                f"请减小 train_window/test_window。"
            }

        sizes = {vt: default_size(vt)}
        try:
            result = await asyncio.to_thread(
                walk_forward,
                bars,
                strat_class,
                dict(req.setting),
                vt,
                req.train_window,
                req.test_window,
                req.step,
                sizes,
                req.capital,
                req.cost if req.cost else None,
            )
            return _sanitize(result.to_dict())
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"运行失败: {str(e)}"}
