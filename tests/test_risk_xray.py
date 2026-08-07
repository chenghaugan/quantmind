"""风险 X 光分析测试。"""
import pytest
import pandas as pd
import numpy as np

from quantmind.research.risk_xray import (
    RiskXrayMetrics,
    compute_risk_xray,
    save_risk_xray,
)


class TestRiskXrayMetrics:
    """风险 X 光指标测试。"""

    def test_compute_basic_metrics(self):
        """基础指标计算。"""
        # 构造简单权益曲线
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        metrics = compute_risk_xray(equity)

        assert isinstance(metrics, RiskXrayMetrics)
        assert metrics.backtest_days == 100
        assert -1 <= metrics.total_return
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown <= 0

    def test_compute_with_trades(self):
        """带交易记录的指标计算。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        trades = [
            {"profit": 10, "holding_days": 5},
            {"profit": -5, "holding_days": 3},
            {"profit": 15, "holding_days": 7},
        ]

        metrics = compute_risk_xray(equity, trades=trades)

        assert metrics.total_trades == 3
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0

    def test_compute_with_positions(self):
        """带持仓数据的集中度计算。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        positions = {
            "stock1": 10000,
            "stock2": 20000,
            "stock3": 30000,
            "stock4": 15000,
            "stock5": 25000,
        }

        metrics = compute_risk_xray(equity, positions=positions)

        assert 0 <= metrics.top_5_concentration <= 1
        assert metrics.herfindahl_index > 0

    def test_to_dict(self):
        """字典序列化。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        metrics = compute_risk_xray(equity)
        d = metrics.to_dict()

        assert "return" in d
        assert "risk" in d
        assert "tail_risk" in d
        assert "concentration" in d
        assert "trading" in d
        assert "metadata" in d

    def test_to_markdown(self):
        """Markdown 报告生成。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        metrics = compute_risk_xray(equity)
        md = metrics.to_markdown()

        assert "# 风险 X 光诊断报告" in md
        assert "## 收益指标" in md
        assert "## 风险指标" in md
        assert "## 尾部风险" in md

    def test_tail_risk_metrics(self):
        """尾部风险指标。"""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        # 构造有厚尾的分布
        returns = np.random.randn(1000) * 0.02
        returns[::10] *= 3  # 每 10 天放大波动
        equity = pd.Series(100 * (1 + returns.cumsum()), index=dates)

        metrics = compute_risk_xray(equity)

        assert metrics.var_95 < 0
        assert metrics.cvar_95 <= metrics.var_95
        assert metrics.skewness is not None
        assert metrics.kurtosis is not None


class TestSaveRiskXray:
    """保存风险 X 光报告测试。"""

    def test_save_to_files(self, tmp_path):
        """保存到 JSON 和 Markdown 文件。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        metrics = compute_risk_xray(equity)
        paths = save_risk_xray(metrics, tmp_path)

        assert "json" in paths
        assert "md" in paths
        assert paths["json"].exists()
        assert paths["md"].exists()

        # 验证 JSON 内容
        import json
        with open(paths["json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "return" in data
        assert "risk" in data
