"""因子衰减监控测试。"""
import numpy as np
import pandas as pd
import pytest

from quantmind.research.decay import (
    FactorState,
    DecayConfig,
    DecayMetrics,
    FactorDecayScanner,
)


class TestFactorDecayScanner:
    """衰减扫描器测试。"""

    def test_compute_metrics_insufficient_data(self):
        """数据不足时返回 notes 提示。"""
        scanner = FactorDecayScanner()
        ic = pd.Series([0.03, 0.02, 0.01], index=pd.date_range("2020-01-01", periods=3))
        metrics = scanner.compute_metrics("f1", ic)
        assert metrics.state == FactorState.ACTIVE
        assert "数据不足" in metrics.notes[0]

    def test_compute_metrics_no_decay(self):
        """IC 稳定时不触发衰减。"""
        scanner = FactorDecayScanner()
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        ic = pd.Series(np.random.normal(0.05, 0.02, 300), index=dates)
        metrics = scanner.compute_metrics("f1", ic)
        assert metrics.state == FactorState.ACTIVE
        assert metrics.ic_mean_recent is not None
        assert metrics.ic_decay_ratio is not None
        assert metrics.ic_decay_ratio > 0.5  # 未衰减

    def test_compute_metrics_with_decay(self):
        """IC 衰减时触发状态转移。"""
        scanner = FactorDecayScanner()
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        # 前 240 天正常 IC=0.05，后 60 天衰减到 IC=0.01（更明显的衰减）
        ic_values = np.random.normal(0.05, 0.01, 240).tolist()
        ic_values.extend(np.random.normal(0.01, 0.01, 60).tolist())
        ic = pd.Series(ic_values, index=dates)

        metrics = scanner.compute_metrics("f1", ic, current_state=FactorState.ACTIVE)
        assert metrics.ic_decay_ratio is not None
        assert metrics.ic_decay_ratio < 0.5  # 衰减超过阈值

        # 触发状态转移
        new_state = scanner.transition_if_needed(metrics)
        assert new_state == FactorState.MONITORING

    def test_transition_monitoring_to_decayed(self):
        """MONITORING 持续衰减 → DECAYED。"""
        scanner = FactorDecayScanner()
        metrics = DecayMetrics(
            factor_id="f1",
            state=FactorState.MONITORING,
            ic_decay_ratio=0.3,  # 低于阈值
        )
        new_state = scanner.transition_if_needed(metrics)
        assert new_state == FactorState.DECAYED

    def test_transition_decayed_to_disabled(self):
        """DECAYED 持续衰减 → DISABLED。"""
        scanner = FactorDecayScanner()
        metrics = DecayMetrics(
            factor_id="f1",
            state=FactorState.DECAYED,
            ic_decay_ratio=0.2,
        )
        new_state = scanner.transition_if_needed(metrics)
        assert new_state == FactorState.DISABLED

    def test_scan_all(self):
        """批量扫描多因子。"""
        scanner = FactorDecayScanner()
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        factor_ic_map = {
            "f1": pd.Series(np.random.normal(0.05, 0.02, 300), index=dates),
            "f2": pd.Series(np.random.normal(0.02, 0.02, 300), index=dates),
        }
        results = scanner.scan_all(factor_ic_map)
        assert len(results) == 2
        assert all(isinstance(m, DecayMetrics) for m in results)

    def test_metrics_to_dict(self):
        """DecayMetrics 序列化。"""
        metrics = DecayMetrics(
            factor_id="f1",
            state=FactorState.ACTIVE,
            ic_mean_recent=0.05,
            ic_mean_history=0.06,
            ic_decay_ratio=0.83,
        )
        d = metrics.to_dict()
        assert d["factor_id"] == "f1"
        assert d["state"] == "active"
        assert d["ic_decay_ratio"] == pytest.approx(0.83, rel=1e-2)


class TestDecayConfig:
    """衰减配置测试。"""

    def test_default_config(self):
        """默认配置值。"""
        cfg = DecayConfig()
        assert cfg.ic_decay_ratio == 0.5
        assert cfg.ic_window_days == 60
        assert cfg.history_window_days == 252
