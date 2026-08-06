"""端到端编排器测试：``run_e2e`` 统一契约的结构/键/类型断言。

合成随机面板的 IC 不稳定，因此只断言返回值的结构、键存在与类型，
不断言任何具体数值大小。provider 传 None → Mock，离线可跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.research import E2EConfig, run_e2e
from quantmind.research.factors.alpha_cs import Panel


def _make_panel(n_dates: int = 60, n_sym: int = 6, seed: int = 0) -> Panel:
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    cols = [f"S{i}" for i in range(n_sym)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(rng.standard_normal((n_dates, n_sym)).cumsum(0) + 50,
                         index=idx, columns=cols)
    vol = pd.DataFrame(np.abs(rng.standard_normal((n_dates, n_sym))) * 1000 + 1e4,
                       index=idx, columns=cols)
    return Panel(close=close, open=close * 0.99, high=close * 1.01,
                 low=close * 0.99, volume=vol, amount=vol * close)


class TestRunE2E:
    def test_run_e2e_full_contract(self):
        """run_e2e 返回完整契约：evidence / pipeline / strategy 三块齐全。"""
        out = run_e2e(
            _make_panel(),
            config=E2EConfig(idea="螺纹钢动量", asset_class="期货",
                             rounds=2, max_candidates=4, run_composite=True),
        )
        assert isinstance(out, dict)
        assert out["client_ready"] is True

        assert "evidence" in out
        assert "hypotheses" in out["evidence"]
        assert "verified_exprs" in out["evidence"]
        assert "factors" in out["evidence"]
        assert "fact_sheet" in out["evidence"]

        assert "pipeline" in out
        assert "summary" in out["pipeline"]
        assert "steps" in out["pipeline"]

        assert "strategy" in out
        assert "code" in out["strategy"]
        assert "code_safe" in out["strategy"]

    def test_e2e_seeds_inject_evidence(self):
        """用户显式 seed 与 AI 证据回灌种子合并，pipeline.config.seeds 至少含用户 seed。"""
        user_seeds = ["delta(close,5)"]
        out = run_e2e(
            _make_panel(),
            config=E2EConfig(idea="螺纹钢动量", asset_class="期货",
                             seeds=user_seeds, rounds=1, max_candidates=3),
        )
        summary = out["pipeline"]["summary"]
        assert summary.get("seed_count", 0) >= 1
        # 用户显式提供的 seed 应出现在 pipeline 配置的 seeds 中
        pipe_seeds = out["pipeline"]["config"].get("seeds", [])
        assert isinstance(pipe_seeds, list)
        assert user_seeds[0] in pipe_seeds

    def test_e2e_evidence_verified_as_seed(self):
        """断链闭合：verified_exprs 是 str 列表；若存在则回灌进 pipeline 种子。"""
        out = run_e2e(
            _make_panel(),
            config=E2EConfig(idea="螺纹钢动量", asset_class="期货",
                             rounds=1, max_candidates=3),
        )
        factors = out["evidence"]["factors"]
        verified = out["evidence"]["verified_exprs"]
        assert isinstance(factors, list)
        assert isinstance(verified, list)
        assert all(isinstance(e, str) for e in verified)

        pipe_seeds = out["pipeline"]["config"].get("seeds", [])
        assert isinstance(pipe_seeds, list)
        for expr in verified:
            assert expr in pipe_seeds

    def test_e2e_strategy_code_sandbox_no_crash(self):
        """策略代码沙箱校验不抛异常，且 code_errors 是 list。"""
        out = run_e2e(
            _make_panel(),
            config=E2EConfig(idea="螺纹钢动量", asset_class="期货",
                             rounds=1, max_candidates=3),
        )
        strat = out["strategy"]
        assert isinstance(strat["code_errors"], list)
        assert isinstance(strat["code"], str)
        assert isinstance(strat["code_safe"], bool)
        assert isinstance(strat["lookahead"], list)

    def test_e2e_min_2_symbols_ok(self):
        """最少 2 个标的也能跑通不抛异常。"""
        out = run_e2e(
            _make_panel(n_sym=2),
            config=E2EConfig(idea="螺纹钢动量", asset_class="期货",
                             rounds=1, max_candidates=3),
        )
        assert isinstance(out, dict)
        assert out["client_ready"] is True
