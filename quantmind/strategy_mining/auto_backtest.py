"""自动回测循环：编译 → 回测 → 评估 → 调整（迭代）。"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ai.provider import LLMProvider
from ..core.object import BarData
from ..paper.promotion import LifecycleManager, LifecycleState
from ..strategy.runners import run_strategy
from .compiler import compile_and_validate
from .prompts import STRATEGY_ADJUSTMENT_SYSTEM, build_adjustment_prompt
from .schema import StrategySpec

_logger = logging.getLogger("quantmind.strategy_mining")


@dataclass
class BacktestResult:
    """回测结果。"""

    spec: StrategySpec
    report: Optional[Dict[str, Any]]
    passed: bool
    iteration: int
    adjustment_notes: str = ""


class AutoBacktestLoop:
    """自动回测循环。

    流程：
    1. 编译 spec → 策略实例
    2. 运行回测
    3. 评估绩效（Sharpe/MDD）
    4. 达标 → 注册生命周期
    5. 不达标 → LLM 分析失败原因，调整参数（最多 N 轮）
    """

    def __init__(
        self,
        lifecycle_manager: LifecycleManager,
        llm_provider: Optional[LLMProvider] = None,
        max_iterations: int = 3,
        min_sharpe: float = 0.5,
        max_drawdown: float = -0.30,
    ) -> None:
        self.lifecycle = lifecycle_manager
        self.provider = llm_provider
        self.max_iterations = max_iterations
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

    async def run(
        self,
        spec: StrategySpec,
        bars: List[BarData],
        strategy_id: Optional[str] = None,
    ) -> BacktestResult:
        """运行自动回测循环。

        Args:
            spec: 初始策略规格
            bars: 历史 K 线数据
            strategy_id: 生命周期策略 ID（可选）

        Returns:
            最终回测结果（最佳迭代）
        """
        current_spec = spec
        best_result: Optional[BacktestResult] = None

        for iteration in range(1, self.max_iterations + 1):
            _logger.info(f"自动回测迭代 {iteration}/{self.max_iterations}")

            # 编译
            success, error, strategy = compile_and_validate(current_spec)
            if not success:
                _logger.error(f"编译失败：{error}")
                return BacktestResult(
                    spec=current_spec,
                    report=None,
                    passed=False,
                    iteration=iteration,
                    adjustment_notes=f"编译失败：{error}",
                )

            # 回测
            vt_symbol = f"{current_spec.symbol}.{current_spec.exchange}"
            setting = strategy.__dict__
            result_dict = run_strategy(
                mode="backtest",
                strategy_class=type(strategy),
                vt_symbol=vt_symbol,
                setting=setting,
                bars=bars,
            )

            report = result_dict.get("report", {})
            sharpe = report.get("sharpe", 0.0)
            max_dd = report.get("max_drawdown", 0.0)

            # 评估
            passed = sharpe >= self.min_sharpe and max_dd >= self.max_drawdown

            result = BacktestResult(
                spec=current_spec,
                report=report,
                passed=passed,
                iteration=iteration,
            )

            # 跟踪最佳结果
            if best_result is None or (
                result.report
                and best_result.report
                and result.report.get("sharpe", 0) > best_result.report.get("sharpe", 0)
            ):
                best_result = result

            # 达标则注册并退出
            if passed:
                _logger.info(f"策略在第 {iteration} 次迭代通过闸门")
                if strategy_id:
                    self._register_to_lifecycle(strategy_id, report)
                return result

            # 不达标且非最后一次迭代，调整参数
            if iteration < self.max_iterations:
                _logger.info(f"策略未通过闸门，调整参数...")
                current_spec = await self._adjust_parameters(current_spec, report, iteration)
                result.adjustment_notes = "参数已调整，进入下一次迭代"

        # 所有迭代结束，返回最佳结果
        _logger.warning(f"策略在 {self.max_iterations} 次迭代后仍未通过闸门")
        return best_result or BacktestResult(
            spec=spec, report=None, passed=False, iteration=0, adjustment_notes="无有效结果"
        )

    async def _adjust_parameters(
        self,
        spec: StrategySpec,
        report: Dict[str, Any],
        iteration: int,
    ) -> StrategySpec:
        """使用 LLM 调整参数（基于回测失败原因）。"""
        if self.provider is None:
            return self._heuristic_adjust(spec, report)

        # 构建调整提示
        prompt = build_adjustment_prompt(
            spec_dict=spec.to_dict(),
            sharpe=report.get("sharpe", 0.0),
            max_drawdown=report.get("max_drawdown", 0.0),
            annual_return=report.get("annual_return", 0.0),
            win_rate=report.get("win_rate", 0.0),
            min_sharpe=self.min_sharpe,
            max_drawdown_threshold=self.max_drawdown,
            iteration=iteration,
        )

        # 调用 LLM
        try:
            import json

            response = await self.provider.chat(STRATEGY_ADJUSTMENT_SYSTEM, prompt)

            # 解析 JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            adjusted_dict = json.loads(json_str.strip())
            return StrategySpec.from_dict(adjusted_dict)
        except Exception as e:
            _logger.warning(f"LLM 调整失败，使用启发式：{e}")
            return self._heuristic_adjust(spec, report)

    def _heuristic_adjust(
        self, spec: StrategySpec, report: Dict[str, Any]
    ) -> StrategySpec:
        """启发式参数调整（LLM 不可用时的回退）。

        简单规则：
        - Sharpe 过低 → 降低阈值（multifactor）或放宽均线 spread（dual_ma）
        - MDD 过大 → 收紧止损或降低仓位
        """
        adjusted = copy.deepcopy(spec)

        sharpe = report.get("sharpe", 0.0)
        max_dd = report.get("max_drawdown", 0.0)

        if sharpe < self.min_sharpe:
            # 尝试提升 Sharpe
            if spec.template.value == "multifactor":
                current = spec.params.get("threshold", 0.3)
                adjusted.params["threshold"] = max(0.1, current * 0.8)
            elif spec.template.value == "dual_ma":
                fast = spec.params.get("fast", 5)
                slow = spec.params.get("slow", 20)
                adjusted.params["fast"] = max(3, fast - 1)
                adjusted.params["slow"] = slow + 5

        if max_dd < self.max_drawdown:
            # 降低风险
            adjusted.risk.max_position = max(0.3, spec.risk.max_position * 0.8)
            adjusted.risk.stop_loss = max(0.02, spec.risk.stop_loss * 0.9)

        return adjusted

    def _register_to_lifecycle(
        self,
        strategy_id: str,
        report: Dict[str, Any],
    ) -> None:
        """将达标的策略注册到生命周期。"""
        metrics = {
            "sharpe": report.get("sharpe", 0.0),
            "max_drawdown": report.get("max_drawdown", 0.0),
            "annual_return": report.get("annual_return", 0.0),
        }
        self.lifecycle.promote(
            strategy_id=strategy_id,
            to_state=LifecycleState.BACKTEST,
            metrics=metrics,
            note="自动回测通过",
        )
        _logger.info(f"策略 {strategy_id} 已注册到生命周期（BACKTEST）")
