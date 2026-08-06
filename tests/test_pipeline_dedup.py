"""新增功能测试：DSL 新算子 / 表达式回测 / 因子去冗 / 端到端流水线。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.factors.panel_expr import (
    panel_eval_expression,
    list_panel_operators,
)
from quantmind.research.cross_sectional_backtest import (
    cross_sectional_backtest,
    factor_expression_backtest,
)
from quantmind.research.dedup import (
    factor_correlation_matrix,
    greedy_cluster_dedup,
    dedup_expressions,
    dedup_factor_panels,
)
from quantmind.research.pipeline import PipelineConfig, run_pipeline
from quantmind.research import (
    factor_expression_backtest as exp_bt,
    dedup_expressions as dedup_exprs,
    run_pipeline as rp,
)


def _make_panel(n_dates: int = 90, n_sym: int = 8, seed: int = 0) -> Panel:
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    cols = [f"S{i}" for i in range(n_sym)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(rng.standard_normal((n_dates, n_sym)).cumsum(0) + 50,
                         index=idx, columns=cols)
    vol = pd.DataFrame(np.abs(rng.standard_normal((n_dates, n_sym))) * 1000 + 1e4,
                       index=idx, columns=cols)
    return Panel(close=close, open=close * 0.99, high=close * 1.01,
                 low=close * 0.99, volume=vol, amount=vol * close)


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


# ---- #4 新算子 -------------------------------------------------------------
class TestNewOperators:
    def test_ts_percentile(self, panel):
        out = panel_eval_expression("ts_percentile(close, 20, 0.9)", panel)
        assert out.shape == panel.close.shape
        # p90 应 ≥ 中位数（同一窗口）
        med = panel_eval_expression("ts_median(close, 20)", panel)
        assert float(out.iloc[-1].mean()) >= float(med.iloc[-1].mean()) - 1e-9

    def test_ts_skew_kurt(self, panel):
        sk = panel_eval_expression("ts_skew(close, 20)", panel)
        ku = panel_eval_expression("ts_kurt(close, 20)", panel)
        assert sk.shape == panel.close.shape
        assert ku.shape == panel.close.shape

    def test_winsorize(self, panel):
        w = panel_eval_expression("winsorize(close, 3.0)", panel)
        assert float(w.max().max()) <= 3.0 + 1e-9
        assert float(w.min().min()) >= -3.0 - 1e-9

    def test_winsorize_q(self, panel):
        q = panel_eval_expression("winsorize_q(close, 0.05, 0.95)", panel)
        assert q.shape == panel.close.shape

    def test_new_ops_registered(self):
        ops = list_panel_operators()
        for o in ["ts_percentile", "ts_skew", "ts_kurt", "winsorize", "winsorize_q"]:
            assert o in ops

    def test_qlib_alias(self, panel):
        out = panel_eval_expression("TsSkew(close, 20)", panel)
        assert out.shape == panel.close.shape


# ---- #2 表达式回测 -------------------------------------------------------
class TestExpressionBacktest:
    def test_cs_backtest_expression(self, panel):
        r = cross_sectional_backtest(panel, "alpha002", expression="delta(close, 20)")
        assert r["factor"] == "delta(close, 20)"
        assert r["n_dates"] > 0
        assert r["portfolio"]["daily_returns"]

    def test_cs_backtest_alpha_name(self, panel):
        r = cross_sectional_backtest(panel, "alpha002")
        assert r["factor"] == "alpha002"
        assert r["n_dates"] > 0

    def test_factor_expression_backtest(self, panel):
        r = factor_expression_backtest("ts_zscore(close, 30)", panel)
        assert r["expression"] == "ts_zscore(close, 30)"
        assert "ic_report" in r and r["ic_report"] is not None

    def test_export(self, panel):
        assert exp_bt == factor_expression_backtest
        r = exp_bt("rank(close, 20)", panel)
        assert r["n_dates"] > 0


# ---- #3 去冗余 ------------------------------------------------------------
class TestDedup:
    def test_correlation_matrix(self, panel):
        dfs = {
            "a": panel_eval_expression("delta(close,20)", panel),
            "b": panel_eval_expression("delta(close,21)", panel),
            "c": panel_eval_expression("ts_zscore(close,30)", panel),
        }
        mat = factor_correlation_matrix(dfs)
        assert list(mat.index) == ["a", "b", "c"]
        assert float(mat.loc["a", "a"]) == 1.0
        # delta20 与 delta21 应高度相关
        assert float(mat.loc["a", "b"]) > 0.8

    def test_greedy_dedup_keeps_high_metric_rep(self):
        import pandas as pd
        mat = pd.DataFrame(
            [[1.0, 0.9, 0.2], [0.9, 1.0, 0.1], [0.2, 0.1, 1.0]],
            index=["a", "b", "c"], columns=["a", "b", "c"])
        metric = {"a": 0.5, "b": 0.9, "c": 0.1}
        clusters = greedy_cluster_dedup(["a", "b", "c"], mat, metric, 0.7)
        reps = [c["name"] for c in clusters]
        assert "b" in reps   # b 是 a/b 簇里 metric 最高者
        assert "c" in reps   # c 独立成簇
        assert len(reps) == 2

    def test_dedup_expressions(self, panel):
        exprs = ["delta(close,20)", "delta(close,21)", "delta(close,22)",
                 "ts_zscore(close,30)", "rank(close,20)", "corr(close,volume,10)"]
        out = dedup_expressions(exprs, panel, correlation_threshold=0.85)
        # 高度相关的一组 delta 应被合并
        assert len(out) < len(exprs)

    def test_dedup_export(self, panel):
        assert dedup_exprs == dedup_expressions
        out = dedup_exprs(["std(close,20)", "std(close,21)"], panel, 0.95)
        assert len(out) >= 1


# ---- #1 端到端流水线 ------------------------------------------------------
class TestPipeline:
    def test_run_pipeline(self, panel):
        cfg = PipelineConfig(seeds=["delta(close,5)", "ts_zscore(close,30)"],
                             algo="co", rounds=2, max_candidates=4,
                             persist_pairs=False)
        rep = run_pipeline(panel, config=cfg)
        assert "summary" in rep and "steps" in rep
        assert rep["summary"]["backtested_count"] >= 1
        for s in rep["steps"]:
            assert s["expression"]

    def test_run_pipeline_export(self, panel):
        assert rp == run_pipeline
        rep = rp(_make_panel(n_dates=100), config=PipelineConfig(
            seeds=["delta(close,5)"], algo="tot", rounds=1, max_candidates=3,
            persist_pairs=False))
        assert rep["summary"]["algo"] == "tot"

    def test_pipeline_filters_weak_factors(self, panel):
        # min_abs_ic 足够高时，metric 全部被当成噪声丢弃 → 无代表回测
        cfg = PipelineConfig(seeds=["delta(close,5)"], algo="co", rounds=1,
                             min_abs_ic=10.0, max_candidates=3, persist_pairs=False)
        rep = run_pipeline(panel, config=cfg)
        assert rep["summary"]["backtested_count"] == 0

    def test_step_reports_have_visual_series(self, panel):
        """每个代表因子 step 应带 daily_returns（逐因子净值/回撤）与 ic_series（IC 时序）。"""
        cfg = PipelineConfig(seeds=["delta(close,5)", "ts_zscore(close,30)"],
                             algo="co", rounds=2, max_candidates=4,
                             run_composite=True, persist_pairs=False)
        rep = run_pipeline(panel, config=cfg)
        assert rep["summary"]["backtested_count"] >= 1
        for s in rep["steps"]:
            assert "daily_returns" in s
            assert "ic_series" in s
            # 至少有一个代表有可画序列
            if s.get("test_sharpe") is not None or s.get("test_ic") is not None:
                assert len(s.get("daily_returns") or []) > 0
                assert len(s.get("ic_series") or []) > 0
