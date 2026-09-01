"""因子表达式统一评估入口 + 持久缓存测试（eval.py）。

覆盖：evaluate_expression 闭环、batch_evaluate_expressions、FactorEvalCache 命中/写入。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.eval import (
    evaluate_expression, batch_evaluate_expressions, FactorEvalCache,
)


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 8, n_dates: int = 150, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    open_ = close * 0.99
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=open_, high=high, low=low, volume=volume)


def test_evaluate_expression_returns_report():
    """统一入口返回含有限 IC 的 FactorReport，且样本数充足。"""
    panel = _make_panel()
    rep = evaluate_expression("Rank($close, 20)", panel, use_cache=False)
    assert rep.n_samples > 50
    assert pd.notna(rep.ic_mean)
    assert abs(rep.ic_mean) < 1.0
    assert rep.factor_name == "Rank($close, 20)"


def test_evaluate_expression_accepts_bars_dict():
    """传 {symbol: List[BarData]} 自动构造面板（各标的独立走势，保证截面差异）。"""
    from quantmind.core.constant import Exchange, Interval
    from quantmind.core.object import BarData
    bars_by_symbol = {}
    for si, s in enumerate(["S0", "S1", "S2"]):
        bars = []
        rng = np.random.default_rng(si)
        ret = 0.01 * np.sin(np.linspace(0, 20, 120)) + rng.normal(0, 0.005, 120)
        c = 100.0
        for i in range(120):
            c = c * (1.0 + ret[i])
            bars.append(BarData(
                symbol=s, exchange=Exchange.SHFE,
                datetime=_utc(i), interval=Interval.DAILY,
                open_price=c, high_price=c * 1.01, low_price=c * 0.99,
                close_price=c, volume=1000.0 + 100.0 * si, open_interest=0.0,
            ))
        bars_by_symbol[s] = bars
    rep = evaluate_expression("Rank($close, 20)", bars_by_symbol, use_cache=False)
    assert pd.notna(rep.ic_mean)
    assert rep.n_samples > 50


def test_batch_evaluate_expressions_order():
    """批量评估返回与输入同序的列表长度。"""
    panel = _make_panel()
    exprs = ["Rank($close, 20)", "Mean($volume, 5)", "Corr($close, $volume, 10)"]
    reps = batch_evaluate_expressions(exprs, panel, use_cache=False)
    assert len(reps) == 3
    for rep in reps:
        assert pd.notna(rep.ic_mean)


def test_cache_hit_returns_same_report(tmp_path):
    """缓存命中时返回与首次一致的报告（用临时 sqlite）。"""
    db = str(tmp_path / "test_cache.sqlite")
    panel = _make_panel()
    cache = FactorEvalCache(db_path=db)

    r1 = evaluate_expression("Rank($close, 20)", panel, use_cache=True, cache=cache,
                             market="csi300")
    # 强制重新求值但走缓存：命中应返回相同结果（缓存放的是经 to_dict 圆整的报告）
    r2 = evaluate_expression("Rank($close, 20)", panel, use_cache=True, cache=cache,
                             market="csi300")
    assert r1.n_samples == r2.n_samples
    # to_dict() 对数值做 r4/r6 圆整，故容差取 1e-3
    assert abs(r1.ic_mean - r2.ic_mean) < 1e-3
    # 验证缓存确实写入（evaluate_expression 写入时带数据指纹，读取需一致）
    from quantmind.research.eval import panel_fingerprint
    fp = panel_fingerprint(panel)
    hit = cache.get("Rank($close, 20)", market="csi300", forward_periods=1,
                    data_fingerprint=fp)
    assert hit is not None
    assert abs(hit.ic_mean - r1.ic_mean) < 1e-3


def test_cache_miss_on_different_key(tmp_path):
    """不同表达式是不同缓存键。"""
    cache = FactorEvalCache(db_path=str(tmp_path / "k.sqlite"))
    assert cache.get("Mean($close, 5)", forward_periods=1) is None
    cache.set(evaluate_expression("Mean($close, 5)", _make_panel(), use_cache=False),
              "Mean($close, 5)", forward_periods=1)
    assert cache.get("Mean($close, 5)", forward_periods=1) is not None
    # 另一个表达式未命中
    assert cache.get("Rank($close, 20)", forward_periods=1) is None
