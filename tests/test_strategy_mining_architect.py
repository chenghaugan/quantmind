"""策略架构师测试。"""
import pytest

from quantmind.strategy_mining.architect import _mock_strategy_architect


class TestMockStrategyArchitect:
    """Mock 策略架构师测试。"""

    def test_no_factors(self):
        """测试无因子输入。"""
        result = _mock_strategy_architect([], None, "rb0", "SHFE")
        assert result["template"] == "dual_ma"
        assert result["symbol"] == "rb0"

    def test_single_momentum_factor(self):
        """测试单动量因子。"""
        factors = [{"name": "mom_20", "kind": "momentum", "window": 20}]
        result = _mock_strategy_architect(factors, None, "rb0", "SHFE")
        assert result["template"] == "dual_ma"
        assert "fast" in result["params"]
        assert "slow" in result["params"]

    def test_multiple_good_factors(self):
        """测试多个高质量因子（ICIR > 0.3）。"""
        factors = [
            {"name": "mom_20", "kind": "momentum", "window": 20, "icir": 0.5},
            {"name": "rev_60", "kind": "mean_reversion", "window": 60, "icir": 0.4},
        ]
        result = _mock_strategy_architect(factors, None, "rb0", "SHFE")
        assert result["template"] == "multifactor"
        assert len(result["factors"]) == 2
        assert result["params"]["threshold"] == 0.3

    def test_vol_and_momentum_factors(self):
        """测试波动率 + 动量因子。"""
        factors = [
            {"name": "vol_20", "kind": "volatility", "window": 20},
            {"name": "mom_60", "kind": "momentum", "window": 60},
        ]
        result = _mock_strategy_architect(factors, None, "rb0", "SHFE")
        assert result["template"] == "vol_target"
        assert "lookback" in result["params"]
        assert "target_vol" in result["params"]

    def test_default_multifactor(self):
        """测试默认回退到 multifactor。"""
        factors = [
            {"name": "f1", "kind": "other", "window": 10},
            {"name": "f2", "kind": "other", "window": 20},
        ]
        result = _mock_strategy_architect(factors, None, "rb0", "SHFE")
        assert result["template"] == "multifactor"
