"""回测绩效分析（年化收益、夏普、索提诺、最大回撤、胜率、盈利因子等）。

输入：权益曲线（list of {date, equity}）与成交记录。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from ..core.object import TradeData


@dataclass
class PerformanceReport:
    """绩效报告。"""

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    profit_factor: float = 0.0
    final_equity: float = 0.0
    equity_curve: List[dict] = field(default_factory=list)
    # 交易成本统计（由 BacktestEngine 附加）
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0
    total_impact: float = 0.0
    total_slippage: float = 0.0
    total_cost: float = 0.0
    margin_used: float = 0.0
    cost_ratio: float = 0.0          # 总成本 / |净收益|

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "annual_return": round(self.annual_return, 4),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar": round(self.calmar, 3),
            "win_rate": round(self.win_rate, 4),
            "trade_count": self.trade_count,
            "profit_factor": round(self.profit_factor, 3),
            "final_equity": round(self.final_equity, 2),
            "total_commission": round(self.total_commission, 2),
            "total_stamp_tax": round(self.total_stamp_tax, 2),
            "total_impact": round(self.total_impact, 2),
            "total_slippage": round(self.total_slippage, 2),
            "total_cost": round(self.total_cost, 2),
            "margin_used": round(self.margin_used, 2),
            "cost_ratio": round(self.cost_ratio, 4),
        }


class PerformanceAnalyzer:
    """绩效分析器。"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        self.tdpy = trading_days_per_year

    def analyze(self, equity_curve: List[dict], trades: List[TradeData]) -> PerformanceReport:
        if not equity_curve:
            return PerformanceReport()
        df = pd.DataFrame(equity_curve).sort_values("date").reset_index(drop=True)
        df["equity"] = df["equity"].astype(float)
        init = df["equity"].iloc[0]
        final = df["equity"].iloc[-1]
        total_return = final / init - 1.0 if init else 0.0

        df["ret"] = df["equity"].pct_change().fillna(0.0)
        n = len(df)
        mean_ret = df["ret"].mean()
        std_ret = df["ret"].std()
        # 年化因子：按「交易日数 × 每日 bar 数」折算每年 bar 数。
        # 日线（每交易日一根）退化为 tdpy；内日数据（M5/H1）不再被当成交易日。
        try:
            _dts = pd.to_datetime(df["date"])
            n_days = max(_dts.dt.date.nunique(), 1)
        except Exception:  # noqa: BLE001
            n_days = n
        bars_per_day = max(n / n_days, 1.0)
        ppy = self.tdpy * bars_per_day  # 每年 bar 数
        annual_return = (1 + total_return) ** (self.tdpy / max(n_days, 1)) - 1 if total_return > -1 else -1.0
        sharpe = (mean_ret / std_ret * (ppy ** 0.5)) if std_ret and std_ret > 0 else 0.0
        downside = df["ret"][df["ret"] < 0].std()
        sortino = (mean_ret / downside * (ppy ** 0.5)) if downside and downside > 0 else 0.0

        # 最大回撤
        df["cummax"] = df["equity"].cummax()
        df["dd"] = df["equity"] / df["cummax"] - 1.0
        max_dd = df["dd"].min()

        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

        # 成交相关
        if trades:
            # 以相邻同标的反向成交近似配对盈亏（简化）
            pnl = self._pair_pnl(trades)
            wins = [p for p in pnl if p > 0]
            losses = [p for p in pnl if p < 0]
            win_rate = len(wins) / len(pnl) if pnl else 0.0
            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return PerformanceReport(
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            win_rate=win_rate,
            trade_count=len(trades),
            profit_factor=profit_factor,
            final_equity=final,
            equity_curve=df.to_dict("records"),
        )

    @staticmethod
    def _pair_pnl(trades: List[TradeData]) -> List[float]:
        """把同一合约的相邻反向成交配对，估算每笔平仓盈亏（用于胜率）。"""
        import dataclasses

        positions: Dict[str, List[TradeData]] = {}
        pnl: List[float] = []
        for t in trades:
            vt = t.vt_symbol
            positions.setdefault(vt, [])
            stack = positions[vt]
            if not stack:
                stack.append(t)
                continue
            prev = stack[-1]
            if prev.direction == t.direction:
                # 同向相邻成交（加仓）：不是平仓回合，继续入栈
                stack.append(t)
                continue
            stack.pop()
            direction = 1 if prev.direction.value == "多" else -1
            matched = min(prev.volume, t.volume)
            pnl.append((t.price - prev.price) * matched * direction)
            # 部分平仓/反手时保留残量，后续配对方向才不会错
            if t.volume > prev.volume:
                residual = dataclasses.replace(t, volume=t.volume - prev.volume)
                stack.append(residual)
            elif prev.volume > t.volume:
                # 开仓量 > 平仓量：开仓记录保留残量回填，剩余平仓才不会错配成新开仓
                stack[-1] = dataclasses.replace(prev, volume=prev.volume - t.volume)
        return pnl
