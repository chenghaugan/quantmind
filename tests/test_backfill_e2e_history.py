"""backfill_e2e_history 回填脚本的轻量测试。

使用 ``tmp_path`` 指向的独立 SQLite 临时库（绝不触碰默认 db/knowledge.db），
验证：
  - 能把老表 factors/strategies 回填成一条 e2e run（含 trials 与 brief）。
  - 幂等：重复 backfill 不产生重复 run / 重复 trials。
  - 未破坏老表（factors/strategies 行数不变）。

离线无网络、零 LLM。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from quantmind.knowledge import KnowledgeStore

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_e2e_history.py"
_spec = importlib.util.spec_from_file_location("backfill_e2e_history", _SCRIPT)
_backfill_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_backfill_mod)

backfill = _backfill_mod.backfill
BACKFILL_RUN_ID = _backfill_mod.BACKFILL_RUN_ID


def _seed_history(store: KnowledgeStore) -> None:
    """在临时库中构造几条老表因子/策略（模拟历史端到端跑批结果）。"""
    store.ingest_factor(name="动量20", expression="delta(close,20)",
                        idea="螺纹钢动量", ic=0.06, status="verified")
    store.ingest_factor(name="低波60", expression="std(close,60)",
                        idea="螺纹钢动量", ic=0.02, status="active")
    store.ingest_factor(name="期限结构", expression="term_structure()",
                        idea="螺纹钢期限结构", ic=0.04, status="n/a")
    store.ingest_strategy(code="Momentum20", code_safe=True,
                          idea="螺纹钢动量", composite_scheme="icir",
                          composite_sharpe=1.1)
    store.ingest_strategy(code="TermStr", code_safe=True,
                          idea="螺纹钢期限结构", composite_scheme="icir",
                          composite_sharpe=0.9)


def _n_rows(store: KnowledgeStore, kind: str) -> int:
    return len(store.list_items(kind=kind, limit=500))


def test_backfill_creates_run_with_trials_and_brief(tmp_path):
    """回填后生成一条 run，trials 数=因子数，含 brief 与统计。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_history(store)

    res = backfill(store)
    assert res["skipped"] is False
    assert res["n_factors"] == 3
    assert res["n_verified"] == 1      # 只有 verified 归为已验证
    assert res["n_strategies"] == 2

    run = store.get_run(BACKFILL_RUN_ID)
    assert run is not None
    meta = run["metadata"]
    assert meta["status"] == "done"
    assert meta["n_representative"] == 1
    assert "3 个因子" in (meta["brief"] or "")
    assert "1 个" in (meta["brief"] or "")

    trials = store.trials_for_run(BACKFILL_RUN_ID)
    assert len(trials) == 3
    statuses = {t["metadata"]["status"] for t in trials}
    assert statuses == {"verified", "active"}


def test_backfill_is_idempotent(tmp_path):
    """重复 backfill：第二次 skipped，run/trials 不重复。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_history(store)

    first = backfill(store)
    assert first["skipped"] is False

    second = backfill(store)
    assert second["skipped"] is True

    runs = store.list_runs(limit=50)
    assert [r["run_id"] for r in runs].count(BACKFILL_RUN_ID) == 1
    assert len(store.trials_for_run(BACKFILL_RUN_ID)) == 3


def test_backfill_preserves_old_tables(tmp_path):
    """回填不改动老表 factors/strategies（行数不变）。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_history(store)

    factors_before = _n_rows(store, "factor")
    strategies_before = _n_rows(store, "strategy")

    backfill(store)

    assert _n_rows(store, "factor") == factors_before
    assert _n_rows(store, "strategy") == strategies_before


def test_backfill_on_empty_history(tmp_path):
    """历史库无因子时仍应生成一条 run 闭环记录（n=0），且可幂等。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))

    res = backfill(store)
    assert res["skipped"] is False
    assert res["n_factors"] == 0
    assert res["n_verified"] == 0

    run = store.get_run(BACKFILL_RUN_ID)
    assert run is not None
    assert run["metadata"]["status"] == "done"
    assert store.trials_for_run(BACKFILL_RUN_ID) == []

    # 幂等
    assert backfill(store)["skipped"] is True
