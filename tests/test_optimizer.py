"""optimizer.py 单元测试：网格枚举 / IS-OOS 切分 / DSR / 参数高原。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quantmind.core.object import BarData
from quantmind.strategy.optimizer import (
    combo_key,
    daily_returns_from_equity,
    deflated_sharpe,
    enumerate_grid,
    neighbor_combos,
    plateau_check,
    split_is_oos,
)

UTC = timezone.utc


def _bar(day_offset: int) -> BarData:
    """构造最小可用的 BarData（datetime 递增即可，切分不感知价格）。"""
    return BarData(
        symbol="t0",
        datetime=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day_offset),
        close_price=100.0,
    )


# ---------------------------------------------------------------------------
# enumerate_grid
# ---------------------------------------------------------------------------

class TestEnumerateGrid:
    def test_cartesian_product(self):
        combos = enumerate_grid({"w": [10, 20], "k": [1.5, 2.0, 2.5]})
        assert len(combos) == 6
        assert {"w": 10, "k": 1.5} in combos
        assert {"w": 20, "k": 2.5} in combos

    def test_single_value_dim_is_fixed(self):
        combos = enumerate_grid({"w": [20], "k": [1.5, 2.0]})
        assert len(combos) == 2
        assert all(c["w"] == 20 for c in combos)

    def test_over_limit_raises(self):
        with pytest.raises(ValueError, match="上限"):
            enumerate_grid({"a": [1, 2], "b": [1, 2, 3], "c": [1, 2, 3, 4],
                            "d": [1, 2, 3, 4, 5]}, max_combos=100)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            enumerate_grid({})

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="非数值"):
            enumerate_grid({"w": ["10", 20]})

    def test_too_many_candidates_raises(self):
        with pytest.raises(ValueError, match="候选值过多"):
            enumerate_grid({"w": list(range(7))})


# ---------------------------------------------------------------------------
# neighbor_combos
# ---------------------------------------------------------------------------

class TestNeighborCombos:
    def test_adjacent_per_dimension(self):
        grid = {"w": [10, 20, 30], "k": [1.5, 2.0]}
        nbrs = neighbor_combos(grid, {"w": 20, "k": 2.0})
        # w=20 的邻居是 10/30（两维），k=2.0 在边界只有 1.5 一个邻居 → 共 3 个
        keys = {tuple(sorted(n.items())) for n in nbrs}
        assert keys == {
            (("k", 2.0), ("w", 10)),
            (("k", 2.0), ("w", 30)),
            (("k", 1.5), ("w", 20)),
        }

    def test_boundary_combo(self):
        grid = {"w": [10, 20, 30]}
        nbrs = neighbor_combos(grid, {"w": 10})
        assert nbrs == [{"w": 20}]  # 边界只有内侧邻居

    def test_missing_param_tolerated(self):
        grid = {"w": [10, 20, 30]}
        assert neighbor_combos(grid, {"other": 1}) == []


# ---------------------------------------------------------------------------
# split_is_oos
# ---------------------------------------------------------------------------

class TestSplitIsOos:
    def test_chronological_split_no_overlap(self):
        bars = [_bar(i) for i in range(1000)]
        is_bars, oos_bars, info = split_is_oos(bars, is_ratio=0.7,
                                               warmup_bars=0)
        assert len(is_bars) == 700
        assert len(oos_bars) == 300
        assert info == {"total": 1000, "is_bars": 700, "oos_bars": 300,
                        "warmup_bars": 0, "degraded": False}
        # 时序无重叠：IS 末根 < OOS 首根
        assert is_bars[-1].datetime < oos_bars[0].datetime

    def test_warmup_borrowed_from_is_tail(self):
        bars = [_bar(i) for i in range(1000)]
        is_bars, oos_bars, info = split_is_oos(bars, is_ratio=0.7,
                                               warmup_bars=120)
        # OOS = 300 根 + 120 根预热（借自 IS 尾部）
        assert info["warmup_bars"] == 120
        assert len(oos_bars) == 300 + 120
        assert oos_bars[0].datetime == is_bars[-120].datetime   # 预热首根来自 IS 尾部
        assert oos_bars[119].datetime == is_bars[-1].datetime   # 预热末根 = IS 末根
        assert oos_bars[120].datetime > is_bars[-1].datetime    # 真正的 OOS 从这里开始

    def test_degraded_when_oos_too_short(self):
        bars = [_bar(i) for i in range(100)]
        is_bars, oos_bars, info = split_is_oos(bars, is_ratio=0.7,
                                               min_oos_bars=60)
        assert oos_bars == []
        assert info["degraded"] is True
        # IS 仍返回全量，供降级单次回测
        assert len(is_bars) == 100

    def test_warmup_capped_to_is_size(self):
        bars = [_bar(i) for i in range(1000)]  # IS=700, OOS=300
        is_bars, oos_bars, info = split_is_oos(bars, is_ratio=0.7,
                                               warmup_bars=500)
        # 预热受限于 warmup_bars=500（IS 至少保留 50 根 → 上限 650，此处 500 生效）
        assert info["warmup_bars"] == 500
        assert len(oos_bars) == 300 + 500

    def test_unsorted_input_gets_sorted(self):
        bars = [_bar(i) for i in [5, 3, 1, 4, 2, 6, 8, 7, 9, 0] * 20]
        is_bars, oos_bars, _ = split_is_oos(bars, is_ratio=0.7, warmup_bars=0)
        all_bars = sorted(bars, key=lambda b: b.datetime)
        assert is_bars[-1].datetime < oos_bars[0].datetime
        assert len(is_bars) + len(oos_bars) == len(bars)
        assert is_bars == sorted(is_bars, key=lambda b: b.datetime)


# ---------------------------------------------------------------------------
# daily_returns_from_equity / deflated_sharpe
# ---------------------------------------------------------------------------

def _equity_from_returns(rets, init=1_000_000.0):
    eq, cur = [], init
    for r in rets:
        cur *= (1 + r)
        eq.append({"equity": cur})
    return eq


class TestDailyReturns:
    def test_basic(self):
        eq = [{"equity": 100.0}, {"equity": 110.0}, {"equity": 99.0}]
        rets = daily_returns_from_equity(eq)
        assert rets.shape == (2,)
        assert rets[0] == pytest.approx(0.10)
        assert rets[1] == pytest.approx(-0.10)


class TestDeflatedSharpe:
    def _norm_sharpe_annual(self, rets, tdpy=252):
        r = np.asarray(rets)
        return float(r.mean() / r.std() * tdpy ** 0.5)

    def test_hand_computed_psr(self):
        """零偏度收益、单次试验：DSR=PSR=Φ(SR_daily·√(T-1)/√(1+((kurt-1)/4)SR²))。"""
        # 确定性三点分布：均值 0、std 0.01、偏度 0、峰度 3（精确成立）
        # 取值 {0(2/3), ±a(1/6)}：m2=(1/3)a², m4/m2²=1/(1-2/3)=3
        a = 0.01 * np.sqrt(3)
        rets = np.array([0.0, 0.0, a, 0.0, 0.0, -a] * 33)  # 198 点，完全对称
        sr_annual = 0.1 * 252 ** 0.5           # 观测日 Sharpe = 0.1
        dsr = deflated_sharpe(sr_annual, rets, n_trials=1)
        from statistics import NormalDist
        z = 0.1 * np.sqrt(len(rets) - 1) / np.sqrt(1 + 0.005)
        assert dsr == pytest.approx(NormalDist().cdf(z), abs=1e-9)

    def test_decreases_with_trials(self):
        rng = np.random.default_rng(42)
        rets = rng.standard_normal(500) * 0.01 + 0.0008
        sr = self._norm_sharpe_annual(rets)
        prev = None
        for n in (1, 10, 100, 1000):
            d = deflated_sharpe(sr, rets, n_trials=n)
            assert 0.0 <= d <= 1.0
            if prev is not None:
                assert d <= prev + 1e-9       # 试验数越多，校正越狠
            prev = d

    def test_high_trial_count_kills_marginal_sharpe(self):
        rng = np.random.default_rng(1)
        raw = rng.standard_normal(300)
        rets = (raw - raw.mean()) / raw.std() * 0.01 + 0.001  # SR_daily=0.1（年化≈1.59）
        sr = self._norm_sharpe_annual(rets)
        # 试了 1000 组参数后，SR₀≈0.20（日频），观测 0.1 远低于基准 → 不可信
        assert deflated_sharpe(sr, rets, n_trials=1000) < 0.5
        # 只试了 1 次，同样的 Sharpe 大概率可信
        assert deflated_sharpe(sr, rets, n_trials=1) > 0.9

    def test_flat_returns_zero(self):
        assert deflated_sharpe(0.0, np.zeros(100), n_trials=5) == 0.0

    def test_short_series_zero(self):
        assert deflated_sharpe(1.5, np.array([0.01]), n_trials=5) == 0.0


# ---------------------------------------------------------------------------
# plateau_check
# ---------------------------------------------------------------------------

class TestPlateauCheck:
    def test_plateau_ok(self):
        grid = {"w": [10, 20, 30]}
        is_sharpes = {combo_key({"w": 10}): 1.0, combo_key({"w": 20}): 1.2, combo_key({"w": 30}): 0.9}
        res = plateau_check(grid, {"w": 20}, is_sharpes)
        assert res["ok"] is True
        assert res["median_ratio"] == pytest.approx(0.95 / 1.2, abs=0.01)

    def test_spike_rejected(self):
        grid = {"w": [10, 20, 30]}
        is_sharpes = {combo_key({"w": 10}): 0.2, combo_key({"w": 20}): 1.5, combo_key({"w": 30}): 0.1}
        res = plateau_check(grid, {"w": 20}, is_sharpes)
        assert res["ok"] is False
        assert res["median_ratio"] < 0.6

    def test_degenerate_grid_passes(self):
        res = plateau_check({"w": [20]}, {"w": 20}, {combo_key({"w": 20}): 1.5})
        assert res["ok"] is True
        assert res["neighbors"] == 0

    def test_nonpositive_best_rejected(self):
        grid = {"w": [10, 20, 30]}
        is_sharpes = {combo_key({"w": 10}): 0.1, combo_key({"w": 20}): -0.5, combo_key({"w": 30}): 0.1}
        res = plateau_check(grid, {"w": 20}, is_sharpes)
        assert res["ok"] is False


# ---------------------------------------------------------------- auto_param_grid
class _ManyParamStrategy:
    """多参数策略桩：混合 int/float/bool/str/非字面量/执行类参数。"""

    parameters = ["fast", "slow", "alpha", "size", "max_pos", "label", "flag", "dynamic"]

    def __init__(self, context, setting=None):
        self.fast = 10          # int → [5, 10, 20]
        self.slow = 20          # int → [10, 20, 40]
        self.alpha = 0.2        # float → [0.1, 0.2, 0.3]
        self.size = 1           # 执行参数 → 跳过
        self.max_pos = 1.0      # 执行参数 → 跳过
        self.label = "x"        # str → 跳过
        self.flag = True        # bool → 跳过
        self.dynamic = compute()  # 非字面量 → 跳过（仅出现在源码中，不执行）
        super().__init__()


def compute():  # noqa: N802 — 供上面源码引用
    return 1.0


def test_auto_grid_generic():
    from quantmind.strategy.optimizer import auto_param_grid
    grid = auto_param_grid(_ManyParamStrategy)
    assert grid == {"fast": [5, 10, 20], "slow": [10, 20, 40],
                    "alpha": [0.1, 0.2, 0.3]}


def test_auto_grid_real_momentum():
    """真实模板：window/threshold 自动邻域，执行参数不进网格。"""
    from quantmind.strategy.optimizer import auto_param_grid
    from quantmind.strategy.validation import MomentumCtaStrategy
    grid = auto_param_grid(MomentumCtaStrategy)
    assert grid == {"window": [10, 20, 40], "threshold": [0.015, 0.03, 0.045]}


def test_auto_grid_no_parameters():
    from quantmind.strategy.optimizer import auto_param_grid

    class _Empty:
        pass

    assert auto_param_grid(_Empty) == {}


def test_fit_grid_trims_middle():
    """4 维×3=81 > 60：削减到 ≤60 且保留各维端点。"""
    from quantmind.strategy.optimizer import fit_grid
    grid = {f"p{i}": [1, 2, 3] for i in range(4)}
    fitted = fit_grid(grid, max_combos=60)
    p = 1
    for name, vals in fitted.items():
        if len(vals) < 3:
            assert vals[0] == 1 and vals[-1] == 3  # 被削的维保留端点
        p *= len(v) if (v := vals) else 1
    assert p <= 60


def test_fit_grid_drops_last_dim_when_all_binary():
    from quantmind.strategy.optimizer import fit_grid
    grid = {"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1, 2], "e": [1, 2]}
    fitted = fit_grid(grid, max_combos=8)
    p = 1
    for v in fitted.values():
        p *= len(v)
    assert p <= 8 and len(fitted) < 5


def test_fit_grid_single_dim_truncates():
    from quantmind.strategy.optimizer import fit_grid
    fitted = fit_grid({"w": list(range(1, 11))}, max_combos=5)
    assert fitted == {"w": [1, 2, 3, 4, 5]}


def test_fit_grid_passthrough_small():
    from quantmind.strategy.optimizer import fit_grid
    grid = {"w": [10, 20, 40], "t": [0.015, 0.03, 0.045]}
    assert fit_grid(grid, max_combos=60) == grid


def test_auto_grid_from_source_string():
    """exec 字符串注册的 LLM 策略：显式传源码也能推导。"""
    from quantmind.strategy.optimizer import auto_param_grid
    src = '''
from quantmind.strategy.base import CtaTemplate

class TpStrategy(CtaTemplate):
    parameters = ["trend_window", "pullback", "stop_loss", "trail",
                  "max_pos", "size", "regime", "label"]
    def __init__(self, context, setting=None):
        self.trend_window = 50
        self.pullback = 0.1
        self.stop_loss = 0.02
        self.trail = 0.05
        self.max_pos = 1.0
        self.size = 1
        self.regime = None
        self.label = "tp"
        super().__init__(context, setting)
    def on_bar(self, bar):
        pass
'''
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns, ns)
    # 不传源码：exec 字符串类 getsource 不到 → 空（调用方回退其它层级）
    assert auto_param_grid(ns["TpStrategy"]) == {}
    # 传入源码：前 4 个信号参数进网格，执行/非数值参数跳过
    grid = auto_param_grid(ns["TpStrategy"], source=src)
    assert grid == {"trend_window": [25, 50, 100], "pullback": [0.05, 0.1, 0.15],
                    "stop_loss": [0.01, 0.02, 0.03], "trail": [0.025, 0.05, 0.075]}


def test_auto_grid_skips_fixed_size():
    """fixed_size 也是执行参数，不进网格。"""
    from quantmind.strategy.optimizer import auto_param_grid
    from quantmind.api.services.backtest_service import _STRATEGY_MAP
    grid = auto_param_grid(_STRATEGY_MAP["dual_ma"])
    assert "fixed_size" not in grid and "size" not in grid
    assert set(grid) == {"fast", "slow"}
