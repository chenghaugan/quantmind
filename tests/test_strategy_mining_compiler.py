"""确定性编译测试。"""
import pytest

from quantmind.strategy_mining.compiler import compile_and_validate, compile_strategy
from quantmind.strategy_mining.schema import (
    FactorInput,
    RiskParams,
    StrategySpec,
    StrategyTemplateType,
)


class TestCompileStrategy:
    """策略编译测试。"""

    def test_compile_dual_ma(self):
        """测试编译 dual_ma 策略。"""
        spec = StrategySpec(
            name="test_dual_ma",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 5, "slow": 20},
        )
        strategy = compile_strategy(spec)
        assert strategy is not None
        assert strategy.fast == 5
        assert strategy.slow == 20

    def test_compile_multifactor(self):
        """测试编译 multifactor 策略。"""
        spec = StrategySpec(
            name="test_multifactor",
            template=StrategyTemplateType.MULTIFACTOR,
            factors=[
                FactorInput(name="mom_20", kind="momentum", window=20, weight=1.0),
            ],
            params={"threshold": 0.3},
        )
        strategy = compile_strategy(spec)
        assert strategy is not None
        assert len(strategy.specs) == 1
        assert strategy.threshold == 0.3

    def test_compile_vol_target(self):
        """测试编译 vol_target 策略。"""
        spec = StrategySpec(
            name="test_vol_target",
            template=StrategyTemplateType.VOL_TARGET,
            params={"lookback": 20, "target_vol": 0.20, "momentum_win": 60},
        )
        strategy = compile_strategy(spec)
        assert strategy is not None
        assert strategy.lookback == 20
        assert strategy.target_vol == 0.20

    def test_compile_pair_trading(self):
        """测试编译 pair_trading 策略。"""
        spec = StrategySpec(
            name="test_pair",
            template=StrategyTemplateType.PAIR_TRADING,
            params={"window": 30, "entry_z": 1.5, "exit_z": 0.3},
        )
        strategy = compile_strategy(spec)
        assert strategy is not None
        assert strategy.window == 30
        assert strategy.entry_z == 1.5

    def test_compile_invalid_spec(self):
        """测试编译无效规格。"""
        spec = StrategySpec(
            name="invalid",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 20, "slow": 5},  # fast > slow
        )
        with pytest.raises(ValueError, match="验证失败"):
            compile_strategy(spec)

    def test_compile_and_validate_success(self):
        """测试 compile_and_validate 成功情况。"""
        spec = StrategySpec(
            name="valid",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 5, "slow": 20},
        )
        success, error, strategy = compile_and_validate(spec)
        assert success
        assert error == ""
        assert strategy is not None

    def test_compile_and_validate_failure(self):
        """测试 compile_and_validate 失败情况。"""
        spec = StrategySpec(
            name="invalid",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 20, "slow": 5},
        )
        success, error, strategy = compile_and_validate(spec)
        assert not success
        assert error != ""
        assert strategy is None
