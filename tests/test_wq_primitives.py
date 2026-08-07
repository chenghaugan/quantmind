"""wq.py 时序原语测试（含 pandas 3.0 下多列 rolling 缺陷的回归保护）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.wq import _corr, _cov, _reg_beta, _reg_resi, _slope


@pytest.fixture
def reg_data() -> tuple:
    rng = np.random.default_rng(0)
    n = 200
    y = pd.Series(rng.standard_normal(n) + 0.5 * np.arange(n) / n)
    x = pd.Series(rng.standard_normal(n) * 0.7 + 0.3 * np.arange(n) / n)
    return y, x


def test_reg_beta_finite_and_no_error(reg_data):
    y, x = reg_data
    # pandas 3.0 下 DataFrame.rolling.apply 会逐列应用并传 1 维窗口，
    # 旧实现会 IndexError。此处应正常产出（回归保护）。
    b = _reg_beta(y, x, 60)
    assert b.shape == y.shape
    finite = b.notna().sum()
    # min_periods=30，warmup 约 29 个 NaN
    assert finite == len(y) - 29
    assert not np.isinf(b.dropna()).any()


def test_reg_resi_finite_and_no_error(reg_data):
    y, x = reg_data
    r = _reg_resi(y, x, 60)
    assert r.shape == y.shape
    finite = r.notna().sum()
    assert finite == len(y) - 29
    assert not np.isinf(r.dropna()).any()


def test_reg_beta_equivalent_to_roll_beta(reg_data):
    """与 academic 模块已验证的滚动协方差实现等价（偏差 0）。"""
    from quantmind.research.factors.academic import _roll_beta
    y, x = reg_data
    b = _reg_beta(y, x, 60)
    ab = _roll_beta(y, x, 60)
    valid = b.notna() & ab.notna()
    assert valid.sum() > 0
    assert float((b - ab).abs()[valid].max()) < 1e-9


def test_reg_resi_explains_variance(reg_data):
    """残差波动应不高于被解释序列本身波动（OLS 有解释力下限）。"""
    y, x = reg_data
    r = _reg_resi(y, x, 60).replace([np.inf, -np.inf], np.nan)
    r_std = float(np.nanstd(r))
    y_std = float(np.nanstd(y))
    assert r_std <= y_std * 1.05


def test_cov_corr_simple():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    c = _cov(a, b, 5)
    # 完全负相关 => 协方差为负
    assert c.dropna().iloc[-1] < 0
    cc = _corr(a, b, 5)
    assert abs(cc.dropna().iloc[-1] + 1.0) < 1e-6
