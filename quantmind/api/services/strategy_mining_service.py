"""LLM 策略挖掘服务。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...ai.provider import LLMProvider
from ...data.manager import DataManager
from ...paper.promotion import LifecycleManager
from ...strategy_mining.architect import design_strategy
from ...strategy_mining.auto_backtest import AutoBacktestLoop
from ...strategy_mining.schema import StrategySpec, validate_spec
from ..schemas import StrategyMiningRequest, AutoBacktestRequest

_logger = logging.getLogger(__name__)


class StrategyMiningService:
    """LLM 策略挖掘服务。"""

    def __init__(
        self,
        data_manager: DataManager,
        lifecycle_manager: LifecycleManager,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.data_manager = data_manager
        self.lifecycle_manager = lifecycle_manager
        self.provider = provider

    async def architect(self, req: StrategyMiningRequest) -> Dict[str, Any]:
        """从因子设计策略规格。"""
        try:
            spec = await design_strategy(
                factors=req.factors,
                constraint=req.constraint,
                template_preference=req.template_preference,
                provider=self.provider,
                symbol=req.symbol,
                exchange=req.exchange,
            )

            # 验证规格
            is_valid, errors = validate_spec(spec)

            return {
                "ok": True,
                "spec": spec.to_dict(),
                "rationale": spec.rationale,
                "valid": is_valid,
                "errors": errors,
            }
        except Exception as e:
            _logger.error(f"策略设计失败：{e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    async def auto_backtest(self, req: AutoBacktestRequest) -> Dict[str, Any]:
        """运行自动回测循环。"""
        try:
            # 反序列化 spec
            spec = StrategySpec.from_dict(req.spec)

            # 获取历史数据
            from ...core.constant import Exchange, Interval
            from ...data.feed.base import HistoryRequest

            exchange = Exchange(req.spec.get("exchange", "SHFE"))
            symbol = req.spec.get("symbol", "rb0")

            history_req = HistoryRequest(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.DAILY,
            )

            bars = await self.data_manager.get_bar_data(history_req)

            if not bars:
                return {"ok": False, "error": f"无法获取 {symbol}.{exchange} 的历史数据"}

            # 交易成本策略：请求显式指定优先，否则按全局 QM_BACKTEST_COST（auto→True/off→False）
            from ...config import get_settings
            req_cost = getattr(req, "cost", None)
            if req_cost is not None:
                effective_cost = req_cost
            else:
                mode = get_settings().backtest_cost
                effective_cost = False if mode == "off" else True

            # 运行自动回测循环
            loop = AutoBacktestLoop(
                lifecycle_manager=self.lifecycle_manager,
                llm_provider=self.provider,
                max_iterations=req.max_iterations,
                min_sharpe=req.min_sharpe,
                max_drawdown=req.max_drawdown,
                max_cost_ratio=getattr(req, "max_cost_ratio", 0.6),
                compare_zero_cost=getattr(req, "compare_zero_cost", True),
                cost=effective_cost,
            )

            result = await loop.run(
                spec=spec,
                bars=bars,
                strategy_id=req.strategy_id,
            )

            return {
                "ok": True,
                "passed": result.passed,
                "iteration": result.iteration,
                "report": result.report,
                "adjustment_notes": result.adjustment_notes,
                "reject_reason": result.reject_reason,
                "cost_metrics": {
                    "total_cost": result.total_cost,
                    "cost_ratio": result.cost_ratio,
                    "trade_count": result.trade_count,
                },
                "zero_cost_compare": {
                    "gross_sharpe": result.gross_sharpe,
                    "gross_annual_return": result.gross_annual_return,
                    "gross_max_drawdown": result.gross_max_drawdown,
                    "cost_drag_sharpe": result.cost_drag_sharpe,
                },
                "final_spec": result.spec.to_dict(),
            }
        except Exception as e:
            _logger.error(f"自动回测失败：{e}", exc_info=True)
            return {"ok": False, "error": str(e)}
