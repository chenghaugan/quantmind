"""知识库（KnowledgeStore）测试：落库 / 检索 / 列表 / 空库。

所有用例使用 ``tmp_path`` 指向的独立 SQLite 文件，避免污染默认库。
"""
from __future__ import annotations

import pytest

from quantmind.knowledge import KnowledgeStore


def _seed(store: KnowledgeStore) -> None:
    store.ingest_factor(
        name="动量20", expression="delta(close,20)", idea="螺纹钢动量",
        ic=0.05, ir=0.3, status="verified",
        symbols=["S0", "S1"], asset_class="期货", market="SHFE",
    )
    store.ingest_factor(
        name="波动率", expression="std(close,20)", idea="低波动",
        ic=0.02, ir=0.1, status="pending",
        symbols=["S0"], asset_class="股票", market="CN",
    )
    store.ingest_strategy(
        code="def algo():\n    return []", code_safe=True,
        idea="螺纹钢动量", composite_scheme="icir", composite_sharpe=0.8,
        symbols=["S0", "S1"],
    )
    store.ingest_research_log(
        idea="螺纹钢动量研究",
        hypotheses=[{"id": "h1", "statement": "动量有效", "status": "VERIFIED"}],
        evidence={"verified_exprs": ["delta(close,20)"]},
    )


def test_ingest_all_kinds(tmp_path):
    """三类对象（factor/strategy/research_log）各自返回非空 kb_id。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    fid = store.ingest_factor(
        name="动量", expression="delta(close,5)", idea="测试", symbols=["S0"])
    sid = store.ingest_strategy(
        code="def algo():\n    return []", code_safe=True, idea="测试")
    rid = store.ingest_research_log(idea="测试")

    assert isinstance(fid, str) and fid
    assert isinstance(sid, str) and sid
    assert isinstance(rid, str) and rid
    assert len({fid, sid, rid}) == 3


def test_search_returns_hits(tmp_path):
    """search 返回 list，每条含 kb_id/kind/text/score，score 为数值。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    hits = store.search("动量", top_k=5)
    assert isinstance(hits, list)
    assert len(hits) > 0
    for hit in hits:
        assert "kb_id" in hit
        assert "kind" in hit
        assert "text" in hit
        assert "score" in hit
        assert isinstance(hit["score"], (int, float))


def test_search_kind_filter(tmp_path):
    """kind 过滤后只返回该种类的结果。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    hits = store.search("动量", kind="factor")
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert all(h["kind"] == "factor" for h in hits)


def test_list_items(tmp_path):
    """list_items 返回非空 list，每条含 kb_id/kind/text。"""
    store = KnowledgeStore(db_path=str(tmp_path / "kb.db"))
    _seed(store)
    items = store.list_items()
    assert isinstance(items, list)
    assert len(items) > 0
    for item in items:
        assert "kb_id" in item
        assert "kind" in item
        assert "text" in item


def test_empty_db(tmp_path):
    """全新空库：search 与 list_items 均返回空 list。"""
    store = KnowledgeStore(db_path=str(tmp_path / "empty.db"))
    assert store.search("xx") == []
    assert store.list_items() == []
