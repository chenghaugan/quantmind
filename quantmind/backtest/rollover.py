"""期货主力换月（rollover）成本估算。

为何需要
--------
策略挖掘/回测通常跑在「主力连续」序列（如 rb0）上，价格已把换月跳空揉进序列里，
但**换月当天的机械操作成本**（平旧约 + 开新约 + 价差滑点）没有被引擎逐笔计费。
这是期货回测最大的隐性成本来源之一，尤其对持仓周期接近主力周期的中低频策略影响显著。

本模块提供（引擎无侵入、纯估算）：
  - ``round_trip_cost_bps(cost)``：单次完整往返（开+平）的成本，换算成 bps（万分之）。
  - ``estimate_rollover_drag(cost, notional_per_position, rollovers_per_year=4)``：
    主力每年换月若干次导致的年度成本拖累（bps 与金额）。

用法（研究参考）
  1. 用 ``round_trip_cost_bps`` 看某品种一次换月的直接成本；
  2. 用 ``estimate_rollover_drag`` 折算成年度拖累，将此拖累叠加到策略的年化收益/Sharpe
     预期上做机会成本判断，或在手动「复跑含换月成本」时按需把费率上调。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .cost import CostModel, lookup_cost


def round_trip_cost_bps(cost: CostModel, volume: float = 1.0,
                        price: float = 1.0, size: float = 1.0) -> float:
    """单次完整往返（开仓 + 平仓）的手续费 + 滑点成本，折算为 bps（万分之 N）。

    近似：开仓一次 + 平昨一次（忽略平今与最低手续费对整数的影响，便于跨品种对比）；
    滑点按 tick 或比例各计一次。返回相对成交额的 bps：``cost / notional * 1e4``。
    """
    notional = price * volume * size
    if notional <= 0:
        return 0.0

    # 手续费（开 + 平昨）
    fee_open = cost.commission_for(volume, price, size, False, False)
    fee_close = cost.commission_for(volume, price, size, False, True)
    # 最低手续费：开/平各计一次
    if cost.min_commission > 0:
        fee_open = max(fee_open, cost.min_commission)
        fee_close = max(fee_close, cost.min_commission)

    # 滑点（开 + 平各一次，买高/卖低都是成本）
    slip_money = 0.0
    if cost.slippage_ticks:
        slip_money += 2 * cost.slippage_ticks * cost.tick_size * volume * size
    if cost.slippage_rate:
        slip_money += 2 * notional * cost.slippage_rate

    total = fee_open + fee_close + slip_money
    return total / notional * 1e4


def estimate_rollover_drag(
    cost: CostModel,
    notional_per_position: float,
    rollovers_per_year: int = 4,
) -> Tuple[float, float]:
    """主力换月的年度成本拖累估算。

    Args:
        cost: 品种成本模型（``lookup_cost`` 解析）。
        notional_per_position: 单次持仓的名义金额（用于把 bps 折算成金额）。
        rollovers_per_year: 每年换月次数（商品期货主力一般每 1-3 个月换一次，取 4-12）。

    Returns:
        (年度拖累 bps, 年度拖累金额)。
    """
    bps = round_trip_cost_bps(cost) * rollovers_per_year
    amount = bps / 1e4 * notional_per_position
    return bps, amount


def summarize_rollovers(vt_symbol: str, notional_per_position: float,
                        rollovers_per_year: int = 4,
                        table: Optional[Dict[str, CostModel]] = None) -> dict:
    """便捷汇总：解析品种成本 → 换月拖累 bps/金额（用于报告/CLI 展示）。"""
    cost = lookup_cost(vt_symbol, table)
    bps, amount = estimate_rollover_drag(cost, notional_per_position, rollovers_per_year)
    per_roll_bps = round_trip_cost_bps(cost)
    return {
        "vt_symbol": vt_symbol,
        "asset_class": cost.asset_class,
        "round_trip_cost_bps": round(per_roll_bps, 4),
        "rollovers_per_year": rollovers_per_year,
        "annual_rollover_drag_bps": round(bps, 4),
        "annual_rollover_drag_amount": round(amount, 2),
        "notional_per_position": notional_per_position,
        "note": cost.note,
    }
