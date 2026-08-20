"""端到端挖掘沉底（e2e_runs / factor_trials）KnowledgeStore 落库测试。

覆盖 start/finish run、ingest trial、successful/failed_factors、
trials_for_run、list_runs、search(kind='trial'/'run')。
所有用例使用 ``tmp_path`` 指向的独立 SQLite 文件，离线无网络。
"""
from __future__ import annotations

import pytest

from quantmind.knowledge import KnowledgeStore


def _seed_run(store: KnowledgeStore, run_id: str = "run-1",
              idea: str = "螺纹钢动量") -> str:
    """开启一次运行并落库若干成功/失败试验，返回 run_id。"""
    store.start_e2e_run(
        run_id=run_id, idea=idea, asset_class="期货", market="SHFE",
        symbols=["rb0.SHFE"], exchange="SHFE", interval="daily", algo="co",
        rounds=3, forward_periods=5,
    )
    store.ingest_factor_trial(
        run_id=run_id, expression="delta(close,20)", algo="co",
        train_ic=0.05, val_ic=0.04, test_ic=0.06, test_sharpe=1.2,
        is_representative=True, status="verified",
        reason="动量在样本外保持正向 IC，趋势延续",
    )
    store.ingest_factor_trial(
        run_id=run_id, expression="std(close,60)", algo="ea",
        train_ic=0.02, test_ic=0.01, status="passed",
        reason="低波有效但增益有限",
    )
    store.ingest_factor_trial(
        run_id=run_id, expression="wma(close,10)", algo="tot",
        train_ic=-0.01, test_ic=-0.03, status="rejected",
        reason="样本外 IC 转负，噪声因子避坑",
    )
    store.ingest_factor_trial(
        run_id=run_id, expression="rank(close)", algo="ea",
        train_ic=0.03, test_ic=0.02, status="redundant",
        reason="与 delta(close,20) 高度相关被去重",
        removed_redundant=["delta(close,10)"],
    )
    store.finish_e2e_run(
        run_id=run_id, n_representative=1, n_verified_hypotheses=2,
        composite_scheme="icw", composite_fwd_ic=0.055, composite_sharpe=1.1,
        brief="动量+低波复合，样本外稳健", status="done",
    )
    return run_id


def test_start_and_finish_run(tmp_path):
    """start/finish e2e run：字段可回填，持久化到库。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    run_id = _seed_run(store, idea="螺纹钢动量")

    run = store.get_run(run_id)
    assert run is not None
    assert run["kind"] == "run"
    meta = run["metadata"]
    assert meta["run_id"] == run_id
    assert meta["idea"] == "螺纹钢动量"
    assert meta["status"] == "done"
    assert meta["n_representative"] == 1
    assert meta["composite_fwd_ic"] == 0.055
    assert "复合" in (meta["brief"] or "")


def test_finish_unknown_run_ignored(tmp_path):
    """finish 不存在的 run 需静默忽略，不报错。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    store.finish_e2e_run("missing-run", n_representative=1)
    assert store.get_run("missing-run") is None


def test_ingest_factor_trial_returns_id(tmp_path):
    """ingest_factor_trial 返回非空 trial_id，且含成败状态。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store)
    tid = store.ingest_factor_trial(
        run_id="run-1", expression="ts_sum(close,5)", status="active")
    assert isinstance(tid, str) and tid


def test_trials_for_run_ordering_and_fields(tmp_path):
    """trials_for_run 全部返回、created_at 升序、含成功与失败。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    run_id = _seed_run(store)

    trials = store.trials_for_run(run_id)
    assert len(trials) == 4
    statuses = {t["metadata"]["status"] for t in trials}
    assert {"verified", "passed", "rejected", "redundant"} <= statuses
    ts = [t["created_at"] for t in trials]
    assert ts == sorted(ts)                      # 升序

    # 其它 run 下的试验互不串扰
    assert store.trials_for_run("run-other") == []


def test_successful_factors(tmp_path):
    """successful_factors 只返回 verified/passed，按 IC 降序。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    run_id = _seed_run(store)

    hits = store.successful_factors()
    assert len(hits) >= 2
    statuses = {h["status"] for h in hits}
    assert statuses <= {"verified", "passed"}
    ics = [h["test_ic"] for h in hits]
    assert ics == sorted(ics, reverse=True)      # 降序
    # 所有命中都来自本 run
    assert all(h["run_id"] == run_id for h in hits)
    for h in hits:
        assert "expression" in h
        assert "status" in h
        assert h["metadata"]["idea"] == "螺纹钢动量"


def test_successful_factors_idea_filter(tmp_path):
    """idea 宽松关键词可过滤 expression。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store)

    hits = store.successful_factors(idea="delta")
    assert len(hits) >= 1
    assert all("delta" in h["expression"] or "delta" in (h["metadata"]["idea"] or "")
               for h in hits)


def test_failed_factors(tmp_path):
    """failed_factors 只返回 rejected/redundant，供避坑参考。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store)

    hits = store.failed_factors()
    assert len(hits) >= 2
    statuses = {h["status"] for h in hits}
    assert statuses <= {"rejected", "redundant"}
    assert all(h["reason"] for h in hits)        # 失败均带 AI 判读原因


def test_list_runs_newest_first(tmp_path):
    """list_runs 返回全部、最新在前，含 metadata 摘要。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store, run_id="run-1", idea="螺纹钢动量")
    _seed_run(store, run_id="run-2", idea="铁矿石基差")

    runs = store.list_runs()
    assert len(runs) == 2
    # 第二笔更晚创建 → 在前
    assert runs[0]["run_id"] == "run-2"
    assert {r["kind"] for r in runs} == {"run"}
    assert all("metadata" in r for r in runs)


def test_list_runs_limit(tmp_path):
    """list_runs 支持 limit。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    for i in range(3):
        _seed_run(store, run_id=f"run-{i}", idea=f"idea-{i}")
    assert len(store.list_runs(limit=2)) == 2


def test_search_kind_run(tmp_path):
    """search(kind='run') 只返回 run 条目。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store, idea="螺纹钢动量")
    hits = store.search("螺纹钢", kind="run")
    assert len(hits) > 0
    assert all(h["kind"] == "run" for h in hits)
    meta = hits[0]["metadata"]
    assert meta["run_id"] == "run-1"


def test_search_kind_trial(tmp_path):
    """search(kind='trial') 只返回 factor_trials 条目。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store)
    hits = store.search("动量", kind="trial")
    assert len(hits) > 0
    assert all(h["kind"] == "trial" for h in hits)
    assert "expression" in hits[0]["metadata"]


def test_list_items_does_not_mix_run_trial(tmp_path):
    """list_items() 缺省不混入 run/trial（保持原有四类全部列表）。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed_run(store)
    store.ingest_factor(name="动量20", expression="delta(close,20)",
                        idea="螺纹钢动量")
    items = store.list_items()
    kinds = {i["kind"] for i in items}
    assert "run" not in kinds
    assert "trial" not in kinds
    # 显式传 kind 可取得
    assert all(i["kind"] == "run" for i in store.list_items(kind="run"))
    assert all(i["kind"] == "trial" for i in store.list_items(kind="trial"))


def test_schema_all_schemas_count():
    """ALL_SCHEMAS 现含 7 张表（既有 4 + e2e_runs + factor_trials + lifecycle）。"""
    from quantmind.knowledge.schema import ALL_SCHEMAS
    assert len(ALL_SCHEMAS) == 7
