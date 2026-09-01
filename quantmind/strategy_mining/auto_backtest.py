"""自动回测循环：编译 → 回测 → 评估 → 调整（迭代）。"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    # 成本与零成本对照（P2）
    reject_reason: str = ""
    total_cost: float = 0.0
    cost_ratio: float = 0.0
    trade_count: int = 0
    gross_sharpe: float = 0.0
    gross_annual_return: float = 0.0
    gross_max_drawdown: float = 0.0
    cost_drag_sharpe: float = 0.0


class AutoBacktestLoop:
    """自动回测循环。

    流程：
    1. 编译 spec → 策略实例
    2. 运行回测（默认启用差异化交易成本）
    3. 评估绩效（Sharpe/MDD + 成本/净收益比拦截高换手）
    4. 达标 → 注册生命周期
    5. 不达标 → LLM 分析失败原因，调整参数（最多 N 轮）

    可选跑一次零成本对照（``compare_zero_cost``），量化成本对 Sharpe 的拖累。
    """

    def __init__(
        self,
        lifecycle_manager: LifecycleManager,
        llm_provider: Optional[LLMProvider] = None,
        max_iterations: int = 3,
        min_sharpe: float = 0.5,
        max_drawdown: float = -0.30,
        cost: Union[bool, Dict[str, Any]] = True,
        max_cost_ratio: float = 0.6,
        compare_zero_cost: bool = True,
    ) -> None:
        self.lifecycle = lifecycle_manager
        self.provider = llm_provider
        self.max_iterations = max_iterations
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        # 挖掘回测默认启用差异化交易成本（按品种计费：平今/印花税/最低手续费/滑点）。
        # 传 False 可退回旧式单一费率用于零成本对照；传 dict 用自定义成本表。
        self.cost: Union[bool, Dict[str, Any]] = cost
        # 成本/净收益上限：启用成本时若超出则判定为高换手策略并拒绝（0=关闭）
        self.max_cost_ratio = max_cost_ratio
        # 是否额外跑一次零成本对照，量化成本拖累
        self.compare_zero_cost = compare_zero_cost

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

            # 回测（净：启用差异化成本）
            vt_symbol = f"{current_spec.symbol}.{current_spec.exchange}"
            setting = strategy.__dict__
            result_dict = run_strategy(
                mode="backtest",
                strategy_class=type(strategy),
                vt_symbol=vt_symbol,
                setting=setting,
                bars=bars,
                cost=self.cost,
            )

            report = result_dict.get("report", {})
            sharpe = report.get("sharpe", 0.0)
            max_dd = report.get("max_drawdown", 0.0)
            total_cost = report.get("total_cost", 0.0) or 0.0
            cost_ratio = report.get("cost_ratio", 0.0) or 0.0
            trade_count = report.get("trade_count", 0)

            # 评估：净 Sharpe/净回撤 + 成本占比拦截高换手
            reject_reason = ""
            passed = sharpe >= self.min_sharpe and max_dd >= self.max_drawdown
            if passed and self.cost and self.max_cost_ratio > 0 and cost_ratio > self.max_cost_ratio:
                # 成本吃掉过多净收益 → 高换手策略，即便零成本 Sharpe 再高也不入库
                passed = False
                reject_reason = (
                    f"成本/净收益 {cost_ratio:.1%} 超过上限 {self.max_cost_ratio:.0%}，"
                    f"总成本 {total_cost:.0f}，判定为高换手策略"
                )

            # 零成本对照：量化成本对 Sharpe 的拖累（仅当启用成本时才有意义）
            gross_sharpe = sharpe
            gross_annual_return = report.get("annual_return", 0.0)
            gross_max_drawdown = max_dd
            if self.compare_zero_cost and self.cost:
                gross_result = run_strategy(
                    mode="backtest",
                    strategy_class=type(strategy),
                    vt_symbol=vt_symbol,
                    setting=setting,
                    bars=bars,
                    cost=False,
                    commission=0.0,
                    slippage=0.0,
                )
                g_report = gross_result.get("report", {})
                gross_sharpe = g_report.get("sharpe", 0.0)
                gross_annual_return = g_report.get("annual_return", 0.0)
                gross_max_drawdown = g_report.get("max_drawdown", 0.0)

            result = BacktestResult(
                spec=current_spec,
                report=report,
                passed=passed,
                iteration=iteration,
                reject_reason=reject_reason,
                total_cost=total_cost,
                cost_ratio=cost_ratio,
                trade_count=trade_count,
                gross_sharpe=gross_sharpe,
                gross_annual_return=gross_annual_return,
                gross_max_drawdown=gross_max_drawdown,
                cost_drag_sharpe=gross_sharpe - sharpe,
            )

            # 跟踪最佳结果（净 Sharpe 最高）
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
                    self._register_to_lifecycle(strategy_id, report, result)
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
            total_cost=report.get("total_cost", 0.0),
            cost_ratio=report.get("cost_ratio", 0.0),
            trade_count=report.get("trade_count", 0),
            max_cost_ratio=self.max_cost_ratio,
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

        # 成本占比过高（高换手）→ 降低仓位/收紧以压缩交易次数
        cost_ratio = report.get("cost_ratio", 0.0)
        if self.max_cost_ratio > 0 and cost_ratio > self.max_cost_ratio:
            adjusted.risk.max_position = max(0.3, adjusted.risk.max_position * 0.8)

        return adjusted

    def _register_to_lifecycle(
        self,
        strategy_id: str,
        report: Dict[str, Any],
        result: Optional[BacktestResult] = None,
    ) -> None:
        """将达标的策略注册到生命周期。"""
        metrics = {
            "sharpe": report.get("sharpe", 0.0),
            "max_drawdown": report.get("max_drawdown", 0.0),
            "annual_return": report.get("annual_return", 0.0),
            "trade_count": report.get("trade_count", 0),
        }
        if result is not None:
            metrics.update({
                "total_cost": round(result.total_cost, 2),
                "cost_ratio": round(result.cost_ratio, 4),
                "gross_sharpe": round(result.gross_sharpe, 3),
                "cost_drag_sharpe": round(result.cost_drag_sharpe, 3),
            })
        self.lifecycle.promote(
            strategy_id=strategy_id,
            to_state=LifecycleState.BACKTEST,
            metrics=metrics,
            note="自动回测通过（含交易成本）",
        )
        _logger.info(f"策略 {strategy_id} 已注册到生命周期（BACKTEST）")
