"""P4：回测成本策略全局配置。"""
from __future__ import annotations

from quantmind.config import Settings


def test_backtest_cost_setting_default_auto():
    """QM_BACKTEST_COST 默认 auto（差异化成本表）。"""
    assert Settings().backtest_cost == "auto"


def test_backtest_cost_setting_override():
    """支持 off / custom 覆盖。"""
    assert Settings(_env_file=None, backtest_cost="off").backtest_cost == "off"
    assert Settings(_env_file=None, backtest_cost="custom").backtest_cost == "custom"
