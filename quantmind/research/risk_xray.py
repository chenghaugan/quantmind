"""风险 X 光分析：回测/模拟盘结果的风险诊断报告。

对标 Vibe-Trading 的 risk_xray 产物，为每次回测/模拟盘运行生成结构化风险指标，
包括：集中度、波动率、最大回撤、尾部风险、风险调整后收益等。

输出格式：
- risk_xray.json: 机器可读的风险指标字典
- risk_xray.md: 人类可读的风险诊断报告（Markdown 格式）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger("quantmind.research.risk_xray")


@dataclass
class RiskXrayMetrics:
    """风险 X 光指标集合。"""

    # 收益指标
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float

    # 风险指标
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # 最大回撤持续天数
    calmar_ratio: float

    # 尾部风险
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    skewness: float
    kurtosis: float

    # 集中度
    top_5_concentration: float  # 前 5 大持仓占比
    herfindahl_index: float  # Herfindahl 指数

    # 交易统计
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_days: float

    # 元数据
    generated_at: str
    backtest_days: int
    risk_free_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（JSON 友好）。"""
        return {
            "return": {
                "total": self.total_return,
                "annualized": self.annualized_return,
                "sharpe": self.sharpe_ratio,
                "sortino": self.sortino_ratio,
            },
            "risk": {
                "volatility": self.volatility,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_duration": self.max_drawdown_duration,
                "calmar": self.calmar_ratio,
            },
            "tail_risk": {
                "var_95": self.var_95,
                "cvar_95": self.cvar_95,
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
            },
            "concentration": {
                "top_5": self.top_5_concentration,
                "herfindahl": self.herfindahl_index,
            },
            "trading": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "avg_holding_days": self.avg_holding_days,
            },
            "metadata": {
                "generated_at": self.generated_at,
                "backtest_days": self.backtest_days,
                "risk_free_rate": self.risk_free_rate,
            },
        }

    def to_markdown(self) -> str:
        """生成 Markdown 格式的风险诊断报告。"""
        lines = [
            "# 风险 X 光诊断报告",
            "",
            f"**生成时间**: {self.generated_at}",
            f"**回测天数**: {self.backtest_days} 天",
            "",
            "## 收益指标",
            f"- 总收益率: {self.total_return:.2%}",
            f"- 年化收益率: {self.annualized_return:.2%}",
            f"- 夏普比率: {self.sharpe_ratio:.2f}",
            f"- 索提诺比率: {self.sortino_ratio:.2f}",
            "",
            "## 风险指标",
            f"- 年化波动率: {self.volatility:.2%}",
            f"- 最大回撤: {self.max_drawdown:.2%}",
            f"- 最大回撤持续: {self.max_drawdown_duration} 天",
            f"- 卡尔玛比率: {self.calmar_ratio:.2f}",
            "",
            "## 尾部风险",
            f"- 95% VaR: {self.var_95:.2%}",
            f"- 95% CVaR: {self.cvar_95:.2%}",
            f"- 偏度: {self.skewness:.2f}",
            f"- 峰度: {self.kurtosis:.2f}",
            "",
            "## 集中度",
            f"- 前 5 大持仓占比: {self.top_5_concentration:.2%}",
            f"- Herfindahl 指数: {self.herfindahl_index:.4f}",
            "",
            "## 交易统计",
            f"- 总交易次数: {self.total_trades}",
            f"- 胜率: {self.win_rate:.2%}",
            f"- 盈亏比: {self.profit_factor:.2f}",
            f"- 平均持仓天数: {self.avg_holding_days:.1f}",
            "",
            "## 风险诊断",
            self._generate_diagnosis(),
        ]
        return "\n".join(lines)

    def _generate_diagnosis(self) -> str:
        """生成风险诊断建议。"""
        warnings = []

        if self.max_drawdown < -0.30:
            warnings.append("⚠️ 最大回撤超过 30%，风险较高")
        if self.sharpe_ratio < 0.5:
            warnings.append("⚠️ 夏普比率低于 0.5，风险调整后收益不佳")
        if self.var_95 < -0.05:
            warnings.append("⚠️ 95% VaR 超过 5%，尾部风险较大")
        if self.top_5_concentration > 0.80:
            warnings.append("⚠️ 持仓集中度过高（前 5 大占比 > 80%）")
        if self.win_rate < 0.40:
            warnings.append("⚠️ 胜率低于 40%，需检查策略逻辑")

        if not warnings:
            return "✅ 风险指标正常，无明显异常"
        return "\n".join(warnings)


def compute_risk_xray(
    equity_curve: pd.Series,
    trades: Optional[List[Dict[str, Any]]] = None,
    positions: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.0,
) -> RiskXrayMetrics:
    """从权益曲线和交易记录计算风险 X 光指标。

    :param equity_curve: 权益曲线（index=日期, value=权益）。
    :param trades: 交易记录列表（可选）。
    :param positions: 当前持仓（可选，用于计算集中度）。
    :param risk_free_rate: 无风险利率（年化）。
    :return: RiskXrayMetrics 实例。
    """
    # 基础收益指标
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_days = len(equity_curve)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    volatility = returns.std() * np.sqrt(252)

    # 风险调整后收益
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0.0

    # 最大回撤
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = drawdown.min()

    # 最大回撤持续天数
    underwater = drawdown < 0
    if underwater.any():
        underwater_groups = (~underwater).cumsum()
        max_drawdown_duration = underwater.groupby(underwater_groups).sum().max()
    else:
        max_drawdown_duration = 0

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 尾部风险
    var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # 集中度（如果有持仓数据）
    if positions and len(positions) > 0:
        pos_values = np.array(list(positions.values()))
        total_pos = pos_values.sum()
        if total_pos > 0:
            pos_weights = pos_values / total_pos
            top_5_concentration = np.sort(pos_weights)[-5:].sum() if len(pos_weights) >= 5 else pos_weights.sum()
            herfindahl_index = (pos_weights ** 2).sum()
        else:
            top_5_concentration = 0.0
            herfindahl_index = 0.0
    else:
        top_5_concentration = 0.0
        herfindahl_index = 0.0

    # 交易统计
    if trades and len(trades) > 0:
        total_trades = len(trades)
        profits = [t.get("profit", 0) for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 1.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0

        # 平均持仓天数（简化计算）
        avg_holding_days = 5.0  # 默认值，实际应从交易记录计算
    else:
        total_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_holding_days = 0.0

    return RiskXrayMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        volatility=volatility,
        max_drawdown=max_drawdown,
        max_drawdown_duration=int(max_drawdown_duration),
        calmar_ratio=calmar_ratio,
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skewness,
        kurtosis=kurtosis,
        top_5_concentration=top_5_concentration,
        herfindahl_index=herfindahl_index,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_holding_days=avg_holding_days,
        generated_at=datetime.now().isoformat(),
        backtest_days=n_days,
        risk_free_rate=risk_free_rate,
    )


def save_risk_xray(
    metrics: RiskXrayMetrics,
    output_dir: str | Path,
    prefix: str = "risk_xray",
) -> Dict[str, Path]:
    """保存风险 X 光报告到文件。

    :param metrics: RiskXrayMetrics 实例。
    :param output_dir: 输出目录。
    :param prefix: 文件名前缀。
    :return: {"json": json_path, "md": md_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    md_path = output_dir / f"{prefix}_{timestamp}.md"

    # 保存 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)

    # 保存 Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(metrics.to_markdown())

    _logger.info("风险 X 光报告已保存: %s, %s", json_path, md_path)
    return {"json": json_path, "md": md_path}


__all__ = [
    "RiskXrayMetrics",
    "compute_risk_xray",
    "save_risk_xray",
]
