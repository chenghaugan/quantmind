"""时点切分 + regime + 种子池 + 配对持久化测试。

覆盖：
  - time_split / PanelSplitter 按时间切分 train/val/test 且不引入前视；
  - regime_labels 给出 bull/bear/sideways；
  - seed_pool 全是合法 DSL 表达式；
  - FactorPairStore 增删查评估配对。
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.panel_expr import panel_eval_expression
from quantmind.research.split import time_split, regime_labels, PanelSplitter
from quantmind.research.factors.seed_pool import (
    list_seed_pool,
    FactorPairStore,
    DEFAULT_SEED_POOL,
)


def _utc(i: int):
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


def _make_panel(n_symbols: int = 6, n_dates: int = 100, seed: int = 11) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [_utc(i) for i in range(n_dates)]
    cols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (n_dates, n_symbols))), index=dates, columns=cols)
    return Panel(close=close, open=close * 0.99, high=close * 1.02,
                 low=close * 0.98,
                 volume=pd.DataFrame(np.abs(rng.normal(1000, 100, (n_dates, n_symbols))),
                                     index=dates, columns=cols))


def test_time_split_sequential():
    """time_split 应保持时间顺序且三段不相交。"""
    panel = _make_panel(n_dates=100)
    tr, va, te = time_split(panel, 0.6, 0.2)
    # 3 段按时间顺序，总数一致
    assert len(tr.dates) + len(va.dates) + len(te.dates) == 100
    assert len(tr.dates) == 60 and len(va.dates) == 20 and len(te.dates) == 20
    # 不重叠
    assert not set(tr.dates) & set(va.dates)
    assert not set(te.dates) & set(va.dates)


def test_regime_labels_all_types_ok():
    """regime_labels 只产出 bull/bear/sideways 且无 NaN 标签。"""
    close = pd.Series(np.cumsum(np.random.default_rng(1).normal(0.01, 1, 80)),
                      index=[_utc(i) for i in range(80)])
    lab = regime_labels(close)
    assert set(lab.unique()) <= {"bull", "bear", "sideways"}
    assert lab.notna().all()


def test_panel_splitter_regime():
    """PanelSplitter 产出三段面板与 regime 分布。"""
    panel = _make_panel(n_dates=90)
    sp = PanelSplitter(train_frac=0.6, val_frac=0.2, regime_window=10)
    res = sp.split(panel, benchmark_series=panel.close.mean(axis=1))
    assert len(res.train.dates) + len(res.val.dates) + len(res.test.dates) == 90
    assert set(res.regime_train.dropna().unique()) <= {"bull", "bear", "sideways"}


def test_seed_pool_all_legal_expressions():
    """种子池里所有表达式都必须是合法可求值的 DSL。"""
    panel = _make_panel()
    assert len(DEFAULT_SEED_POOL) >= 15
    for name, expr in list_seed_pool():
        assert panel_eval_expression(expr, panel).shape == panel.close.shape, f"{name}: {expr}"


def test_factor_pair_store_roundtrip():
    """FactorPairStore 能写入并读回评估配对。"""
    with tempfile.TemporaryDirectory() as td:
        store = FactorPairStore(db_path=str(Path(td) / "pairs.sqlite"))
        n = store.add_pairs(
            [("delta(close,20)", 0.03, 0.025), ("std(close,10)", 0.01, 0.008, 0.1, 0.5, 120)],
            market="csi300", forward_periods=1,
        )
        assert n == 2
        assert store.count() == 2
        rows = store.load_pairs(market="csi300")
        assert len(rows) == 2
        exprs = {r["expression"] for r in rows}
        assert "delta(close,20)" in exprs and "std(close,10)" in exprs
        # 覆盖写入：同一表达式更新而非追加
        store.add_pairs([("delta(close,20)", 0.05, 0.04)], market="csi300")
        assert store.count() == 2
        updated = [r for r in store.load_pairs(market="csi300")
                   if r["expression"] == "delta(close,20)"][0]
        assert updated["rank_ic"] == 0.04
