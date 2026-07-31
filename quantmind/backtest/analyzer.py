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
        annual_return = (1 + total_return) ** (self.tdpy / max(n, 1)) - 1 if total_return > -1 else -1.0
        sharpe = (mean_ret / std_ret * (self.tdpy ** 0.5)) if std_ret and std_ret > 0 else 0.0
        downside = df["ret"][df["ret"] < 0].std()
        sortino = (mean_ret / downside * (self.tdpy ** 0.5)) if downside and downside > 0 else 0.0

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
        """把同一合约的相邻开平配对，估算每笔平仓盈亏（用于胜率）。"""
        positions: Dict[str, List[TradeData]] = {}
        pnl: List[float] = []
        for t in trades:
            vt = t.vt_symbol
            positions.setdefault(vt, [])
            stack = positions[vt]
            if not stack:
                stack.append(t)
            else:
                # 简化：用后一成交价减前一成交价，按方向估算
                prev = stack.pop()
                direction = 1 if prev.direction.value == "多" else -1
                p = (t.price - prev.price) * t.volume * direction
                pnl.append(p)
        return pnl
