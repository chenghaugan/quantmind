"""面板级因子表达式 DSL 测试（panel_expr.py）。

覆盖：Qlib式/函数式等价、算子数值正确性、形状一致、AST 安全。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.panel_expr import (
    panel_eval_expression, ExpressionError, list_panel_operators,
)
from quantmind.research.factors.wq import _rank_cs, _corr, _delay, _delta, _ts_min, _ts_max


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 4, n_dates: int = 60, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_qlib_and_functional_equivalent():
    """Qlib 式(Mean/$close) 与 函数式(mean/close) 求值结果一致。"""
    panel = _make_panel()
    a = panel_eval_expression("Mean($close, 5)", panel)
    b = panel_eval_expression("mean(close, 5)", panel)
    assert a.shape == panel.close.shape
    assert a.columns.tolist() == list(panel.close.columns)
    pd.testing.assert_frame_equal(a, b)


def test_rank_is_cross_sectional():
    """rank(close) 应 = 截面百分位排名 _rank_cs。"""
    panel = _make_panel()
    out = panel_eval_expression("rank(close)", panel)
    expected = _rank_cs(panel.close)
    assert np.allclose(out.values, expected.values, atol=1e-9)


def test_corr_matches_wq():
    """corr(a, b, 10) 应等于 wq._corr。"""
    panel = _make_panel()
    out = panel_eval_expression("corr(close, volume, 10)", panel)
    expected = _corr(panel.close, panel.volume, 10).fillna(0.0)
    assert np.allclose(out.values, expected.values, atol=1e-9)


def test_delay_and_delta():
    """delay(a,2) 与 delta(a,2) 数值正确（与 wq 原语一致）。"""
    panel = _make_panel()
    d = panel_eval_expression("delay(close, 2)", panel)
    assert np.allclose(d.values, _delay(panel.close, 2).fillna(0.0).values, atol=1e-9)
    dl = panel_eval_expression("delta(close, 2)", panel)
    assert np.allclose(dl.values, _delta(panel.close, 2).fillna(0.0).values, atol=1e-9)


def test_arithmetic_composition():
    """支持算术组合：mean(close,5)/rank(close) - 1。"""
    panel = _make_panel()
    out = panel_eval_expression("mean(close,5) / rank(close) - 1.0", panel)
    assert out.shape == panel.close.shape
    assert out.notna().all().all()
    assert np.isfinite(out.values).all()


def test_all_operators_return_panel_shaped_grid():
    """全部算子返回与面板对齐的 DataFrame（index/columns 一致）。"""
    panel = _make_panel(n_symbols=5, n_dates=80)
    unary_ts = ["mean", "std", "sum", "ts_zscore", "ts_min", "ts_max",
                "ts_arg_max", "ts_arg_min", "ts_product", "ts_median", "slope",
                "decay_linear", "ts_rank"]
    for op in unary_ts:
        out = panel_eval_expression(f"{op}(close, 10)", panel)
        assert isinstance(out, pd.DataFrame), op
        assert out.shape == panel.close.shape, op
        assert np.isfinite(out.values).all(), op
    for op in ["rank", "cs_zscore"]:
        out = panel_eval_expression(f"{op}(close)", panel)
        assert out.shape == panel.close.shape, op
        assert np.isfinite(out.values).all(), op
    # 标量弹数
    for op, args in [("sign", "close"), ("abs", "close"), ("log", "close"),
                     ("power", "close, 2")]:
        out = panel_eval_expression(f"{op}({args})", panel)
        assert out.shape == panel.close.shape, op


def test_security_rejects_unknown_and_attribute_access():
    """未知算子 / 未知变量 / 属性访问必须抛 ExpressionError。"""
    panel = _make_panel()
    with pytest.raises(ExpressionError):
        panel_eval_expression("evil(close)", panel)
    with pytest.raises(ExpressionError):
        panel_eval_expression("mean(close.foo, 5)", panel)
    with pytest.raises(ExpressionError):
        panel_eval_expression("mean(unknown_var, 5)", panel)
    with pytest.raises(ExpressionError):
        panel_eval_expression("__import__('os').system('ls')", panel)


def test_list_operators_nonempty():
    names = list_panel_operators()
    assert "rank" in names
    assert "corr" in names
    assert "Mean" in names  # Qlib 别名
    assert "delay" in names


def test_empty_panel_raises():
    empty = Panel(close=pd.DataFrame(), open=pd.DataFrame(), high=pd.DataFrame(),
                  low=pd.DataFrame(), volume=pd.DataFrame())
    with pytest.raises(ExpressionError):
        panel_eval_expression("rank(close)", empty)
