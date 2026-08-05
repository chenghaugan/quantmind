"""多因子组合构建与权重优化测试（combine.py）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.combine import (
    cs_rank_panel,
    cs_zscore_panel,
    standardize_panel,
    equal_weights,
    icir_weights,
    inverse_variance_weights,
    min_variance_weights,
    combine_factor_panels,
    optimize_weights,
    composite_backtest,
)
from quantmind.research import (
    combine_icir_weights,
    combine_factor_panels as c2,
    optimize_weights as ow,
    composite_backtest as cb,
)
from quantmind.research.factors.panel_expr import panel_eval_expression
from quantmind.research.pipeline import PipelineConfig, run_pipeline


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


@pytest.fixture(scope="module")
def factor_dfs(panel):
    return {
        "delta20": panel_eval_expression("delta(close,20)", panel),
        "z30": panel_eval_expression("ts_zscore(close,30)", panel),
        "mom5": panel_eval_expression("rank(close,5)", panel),
    }


# ---- 截面标准化 ------------------------------------------------------------
class TestStandardize:
    def test_cs_rank_range(self, panel):
        r = cs_rank_panel(panel.close - 50)
        v = r.stack().dropna()
        assert v.min() >= 0.0 and v.max() <= 1.0

    def test_cs_zscore_mean_std(self, panel):
        z = cs_zscore_panel(panel.close)
        per_date = z.T
        for d in z.index:
            row = z.loc[d].dropna()
            if len(row) >= 2:
                assert abs(row.mean()) < 1e-9
                assert abs(row.std() - 1.0) < 1e-6

    def test_cs_zscore_clip(self, panel):
        z = cs_zscore_panel(panel.close, clip=1.0)
        assert float(z.max().max()) <= 1.0 + 1e-9
        assert float(z.min().min()) >= -1.0 - 1e-9

    def test_standardize_rank_dispatch(self, panel):
        r = standardize_panel(panel.close, "rank")
        assert r.min().min() >= 0.0

    def test_standardize_invalid_method(self, panel):
        with pytest.raises(ValueError):
            standardize_panel(panel.close, "bogus")


# ---- 权重方案 --------------------------------------------------------------
class TestWeights:
    def test_equal_weights(self):
        w = equal_weights(4)
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < 1e-12
        assert np.allclose(w, 0.25)

    def test_equal_weights_zero(self):
        assert equal_weights(0).size == 0

    def test_icir_nonneg_normalized(self):
        w = icir_weights([0.08, 0.02, -0.05], [0.1, 0.1, 0.1])
        assert w.size == 3
        assert abs(w.sum() - 1.0) < 1e-12
        assert np.all(w >= 0)          # long_only 钳负
        assert w[0] > w[1]             # ICIR 高者权重大

    def test_icir_all_invalid_falls_back_equal(self):
        w = icir_weights([0.0, 0.0], [0.0, 0.0])
        assert np.allclose(w, 0.5)

    def test_inverse_variance(self):
        w = inverse_variance_weights([0.25, 1.0, 4.0])  # 方差1,4,16
        assert abs(w.sum() - 1.0) < 1e-12
        assert w[0] > w[1] > w[2]      # 波动小者权重大

    def test_inverse_variance_invalid(self):
        w = inverse_variance_weights([0.0, 2.0, np.nan])
        assert abs(w.sum() - 1.0) < 1e-12

    def test_min_variance_identity(self):
        corr = pd.DataFrame(np.eye(3), index=["a", "b", "c"],
                            columns=["a", "b", "c"])
        w = min_variance_weights(corr)
        assert abs(w.sum() - 1.0) < 1e-12
        # 等相关时最小方差权重相等
        assert np.allclose(w, 1.0 / 3, atol=1e-6)

    def test_min_variance_deweights_correlated(self):
        # b/c 与 a 高度相关 → a 权重应明显高（分散化奖励低相关）
        corr = pd.DataFrame(
            [[1.0, 0.2, 0.2], [0.2, 1.0, 0.99], [0.2, 0.99, 1.0]],
            index=["a", "b", "c"], columns=["a", "b", "c"])
        w = min_variance_weights(corr)
        assert abs(w.sum() - 1.0) < 1e-12
        assert w[0] > w[1] and w[0] > w[2]


# ---- 组合合成 --------------------------------------------------------------
class TestCombine:
    def test_combine_equal(self, factor_dfs):
        comp = combine_factor_panels(factor_dfs)
        assert comp.shape == factor_dfs["delta20"].shape
        assert comp.notna().any().any()

    def test_combine_weights_sum(self, factor_dfs):
        comp = combine_factor_panels(factor_dfs, weights=[0.5, 0.3, 0.2])
        assert comp.shape == factor_dfs["delta20"].shape

    def test_combine_weight_mismatch(self, factor_dfs):
        with pytest.raises(ValueError):
            combine_factor_panels(factor_dfs, weights=[0.5])

    def test_combine_rank_method(self, factor_dfs):
        comp = combine_factor_panels(factor_dfs, standardize="rank")
        assert comp.shape == factor_dfs["delta20"].shape

    def test_combine_orthogonal(self, factor_dfs):
        comp = combine_factor_panels(factor_dfs, standardize="zscore",
                                     orthogonalize=True)
        assert comp.shape == factor_dfs["delta20"].shape


# ---- 权重优化 --------------------------------------------------------------
class TestOptimize:
    def test_optimize_equal(self, factor_dfs):
        w = optimize_weights(factor_dfs, scheme="equal")
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w.keys()) == set(factor_dfs.keys())

    def test_optimize_icir(self, factor_dfs):
        ic = {k: {"ic_mean": 0.06, "ic_std": 0.1} for k in factor_dfs}
        w = optimize_weights(factor_dfs, scheme="icir", ic_reports=ic)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_optimize_inv_var(self, factor_dfs):
        w = optimize_weights(factor_dfs, scheme="inv_var")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_optimize_min_var(self, factor_dfs):
        w = optimize_weights(factor_dfs, scheme="min_var")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_optimize_invalid_scheme(self, factor_dfs):
        with pytest.raises(ValueError):
            optimize_weights(factor_dfs, scheme="bogus")

    def test_optimize_empty(self):
        assert optimize_weights({}) == {}


# ---- 复合回测 --------------------------------------------------------------
class TestCompositeBacktest:
    def test_composite_backtest_basic(self, panel):
        r = composite_backtest(
            ["delta(close,10)", "ts_zscore(close,20)", "rank(close,5)"],
            panel, scheme="icir", forward_periods=1)
        assert "weights" in r and "portfolio" in r and "ic_report" in r
        assert r["n_dates"] > 0
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-6

    def test_composite_correlation_matrix(self, panel):
        """复合回测返回因子相关矩阵（供前端热力图）。"""
        r = composite_backtest(
            ["delta(close,10)", "ts_zscore(close,20)", "rank(close,5)"],
            panel, scheme="equal")
        corr = r["correlation"]
        assert set(corr["columns"]) == {"delta(close,10)", "ts_zscore(close,20)",
                                        "rank(close,5)"}
        assert len(corr["values"]) == len(corr["columns"])
        # 对角线为 1，对称
        for i, c in enumerate(corr["columns"]):
            assert corr["values"][i][i] == 1 or corr["values"][i][i] is None
        v01 = corr["values"][0][1]
        v10 = corr["values"][1][0]
        if v01 is not None and v10 is not None:
            assert abs(v01 - v10) < 1e-6

    def test_composite_backtest_oos_split(self, panel):
        # train 拟合权重, test 期 OOS 回测（防泄漏）
        train = _make_panel(n_dates=60, seed=1)
        test = _make_panel(n_dates=40, seed=2)
        r = composite_backtest(
            ["delta(close,10)", "ts_zscore(close,20)", "rank(close,5)"],
            test, training_panel=train, scheme="icir")
        assert r["n_dates"] > 0
        assert r["portfolio"]["daily_returns"]

    def test_composite_backtest_min_var(self, panel):
        r = composite_backtest(
            ["delta(close,10)", "ts_zscore(close,20)", "rank(close,5)"],
            panel, scheme="min_var")
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-6

    def test_composite_backtest_no_exprs(self, panel):
        with pytest.raises(ValueError):
            composite_backtest([], panel)

    def test_composite_export(self, panel):
        assert cb == composite_backtest
        r = cb(["delta(close,10)", "rank(close,5)"], panel, scheme="equal")
        assert "portfolio" in r


# ---- 导出 ------------------------------------------------------------------
class TestExport:
    def test_combine_icir_export(self):
        assert combine_icir_weights == icir_weights
        w = combine_icir_weights([0.1, 0.05], [0.1, 0.1])
        assert abs(w.sum() - 1.0) < 1e-12

    def test_combine_export_identity(self, factor_dfs):
        assert c2 == combine_factor_panels
        assert ow == optimize_weights


# ---- 流水线复合步骤 --------------------------------------------------------
class TestPipelineComposite:
    def test_pipeline_composite_off_by_default(self, panel):
        rep = run_pipeline(panel, config=PipelineConfig(
            seeds=["delta(close,5)", "ts_zscore(close,30)"], algo="co", rounds=1,
            max_candidates=4, persist_pairs=False))
        assert rep["composite"] is None    # 默认不跑复合

    def test_pipeline_composite_on(self, panel):
        rep = run_pipeline(panel, config=PipelineConfig(
            seeds=["delta(close,5)", "ts_zscore(close,30)", "rank(close,5)"],
            algo="co", rounds=1, max_candidates=5, persist_pairs=False,
            run_composite=True, composite_scheme="icir",
            train_frac=0.5, val_frac=0.2))
        comp = rep["composite"]
        assert comp is not None            # 复合组合已回测
        assert "weights" in comp and "portfolio" in comp
        assert abs(sum(comp["weights"].values()) - 1.0) < 1e-6
        assert comp["portfolio"]["daily_returns"]

    def test_pipeline_composite_min_var(self, panel):
        rep = run_pipeline(panel, config=PipelineConfig(
            seeds=["delta(close,5)", "ts_zscore(close,30)", "rank(close,5)"],
            algo="co", rounds=1, max_candidates=5, persist_pairs=False,
            run_composite=True, composite_scheme="min_var",
            train_frac=0.5, val_frac=0.2))
        assert rep["composite"]["scheme"] == "min_var"
