"""StrategySpec 数据模型测试。"""
import pytest

from quantmind.strategy_mining.schema import (
    FactorInput,
    RiskParams,
    StrategySpec,
    StrategyTemplateType,
    validate_spec,
)


class TestStrategySpec:
    """StrategySpec 序列化/反序列化测试。"""

    def test_to_dict(self):
        """测试序列化为字典。"""
        spec = StrategySpec(
            name="test_strategy",
            template=StrategyTemplateType.DUAL_MA,
            description="测试策略",
            params={"fast": 5, "slow": 20},
            risk=RiskParams(stop_loss=0.05, take_profit=0.15, max_position=1.0),
            symbol="rb0",
            exchange="SHFE",
            capital=1_000_000.0,
            rationale="测试理由",
        )

        d = spec.to_dict()

        assert d["name"] == "test_strategy"
        assert d["template"] == "dual_ma"
        assert d["params"]["fast"] == 5
        assert d["risk"]["stop_loss"] == 0.05
        assert d["symbol"] == "rb0"

    def test_from_dict(self):
        """测试从字典反序列化。"""
        data = {
            "name": "test_multifactor",
            "template": "multifactor",
            "description": "多因子策略",
            "factors": [
                {
                    "name": "momentum_20",
                    "kind": "momentum",
                    "window": 20,
                    "weight": 1.0,
                    "icir": 0.5,
                }
            ],
            "params": {"threshold": 0.3},
            "risk": {"stop_loss": 0.05, "take_profit": 0.15, "max_position": 1.0},
            "symbol": "rb0",
            "exchange": "SHFE",
            "capital": 1_000_000.0,
            "rationale": "测试",
        }

        spec = StrategySpec.from_dict(data)

        assert spec.name == "test_multifactor"
        assert spec.template == StrategyTemplateType.MULTIFACTOR
        assert len(spec.factors) == 1
        assert spec.factors[0].name == "momentum_20"
        assert spec.params["threshold"] == 0.3

    def test_roundtrip(self):
        """测试序列化/反序列化往返。"""
        original = StrategySpec(
            name="roundtrip_test",
            template=StrategyTemplateType.VOL_TARGET,
            factors=[
                FactorInput(name="vol_20", kind="volatility", window=20, weight=1.0)
            ],
            params={"lookback": 20, "target_vol": 0.20, "momentum_win": 60},
            risk=RiskParams(stop_loss=0.05, take_profit=0.15, max_position=0.8),
        )

        d = original.to_dict()
        restored = StrategySpec.from_dict(d)

        assert restored.name == original.name
        assert restored.template == original.template
        assert restored.params == original.params
        assert restored.risk.stop_loss == original.risk.stop_loss


class TestValidateSpec:
    """StrategySpec 验证测试。"""

    def test_valid_dual_ma(self):
        """测试有效的 dual_ma 规格。"""
        spec = StrategySpec(
            name="valid_dual_ma",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 5, "slow": 20},
        )
        is_valid, errors = validate_spec(spec)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_dual_ma_missing_params(self):
        """测试缺少参数的 dual_ma。"""
        spec = StrategySpec(
            name="invalid_dual_ma",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 5},  # 缺少 slow
        )
        is_valid, errors = validate_spec(spec)
        assert not is_valid
        assert any("fast" in e and "slow" in e for e in errors)

    def test_invalid_dual_ma_fast_ge_slow(self):
        """测试 fast >= slow 的 dual_ma。"""
        spec = StrategySpec(
            name="invalid_dual_ma",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 20, "slow": 10},
        )
        is_valid, errors = validate_spec(spec)
        assert not is_valid
        assert any("快线" in e for e in errors)

    def test_valid_multifactor(self):
        """测试有效的 multifactor 规格。"""
        spec = StrategySpec(
            name="valid_multifactor",
            template=StrategyTemplateType.MULTIFACTOR,
            factors=[FactorInput(name="mom_20", kind="momentum", window=20)],
            params={"threshold": 0.3},
        )
        is_valid, errors = validate_spec(spec)
        assert is_valid

    def test_invalid_multifactor_no_factors(self):
        """测试无因子的 multifactor。"""
        spec = StrategySpec(
            name="invalid_multifactor",
            template=StrategyTemplateType.MULTIFACTOR,
            factors=[],
            params={"threshold": 0.3},
        )
        is_valid, errors = validate_spec(spec)
        assert not is_valid
        assert any("至少一个因子" in e for e in errors)

    def test_invalid_risk_params(self):
        """测试无效的风险参数。"""
        spec = StrategySpec(
            name="invalid_risk",
            template=StrategyTemplateType.DUAL_MA,
            params={"fast": 5, "slow": 20},
            risk=RiskParams(stop_loss=-0.05, max_position=1.5),
        )
        is_valid, errors = validate_spec(spec)
        assert not is_valid
        assert any("止损" in e for e in errors)
        assert any("仓位" in e for e in errors)
