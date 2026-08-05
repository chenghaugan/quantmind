"""SearchService / 因子搜索 REST 端点测试（FastAPI TestClient，离线）。

验证 P2 接入：表达式截面评估、批量评估、CoT 迭代搜索三个端点可用，
返回结构与 schema 兼容（app 的 mock 数据源各标的独立 seed，保证截面差异）。
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
from quantmind.api.app import app

SYMBOLS = ["rb0", "hc0", "bu0", "i0"]


@pytest.fixture(scope="module")
def client():
    """复用单个 TestClient，避免每个测试重复初始化 app lifespan（开销大）。"""
    with TestClient(app) as c:
        yield c


def test_expr_eval_ep(client):
    """POST /factor/expr-eval 返回截面 IC 报告。"""
    r = client.post("/factor/expr-eval", json={
        "expression": "Rank($close, 20)",
        "symbols": SYMBOLS,
        "exchange": "SHFE",
        "forward_periods": 1,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_symbols"] >= 2
    assert body["n_dates"] > 0
    # 随机 mock 数据下 IC 应有限
    assert body["ic_mean"] is None or abs(body["ic_mean"]) <= 1.0
    assert "ic_decay" in body


def test_expr_eval_invalid_expression(client):
    """非法表达式 → 400（表达式求值错误属客户端输入，ExpressionError 是 ValueError）。"""
    r = client.post("/factor/expr-eval", json={
        "expression": "evil(close, 5)",
        "symbols": SYMBOLS,
    })
    assert r.status_code == 400
    assert "error" in r.json() or "detail" in r.json()


def test_expr_batch_ep(client):
    """POST /factor/expr-batch 返回等长报告列表。"""
    r = client.post("/factor/expr-batch", json={
        "expressions": ["Mean($close, 5)", "Rank($close, 20)", "Corr($close,$volume,10)"],
        "symbols": SYMBOLS,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 3
    for rep in body:
        assert "ic_mean" in rep


def test_factor_search_ep(client):
    """POST /factor/search 返回 SearchResult（seed/best/trajectory）。"""
    r = client.post("/factor/search", json={
        "seed": "Mean($close, 20)",
        "symbols": SYMBOLS,
        "rounds": 2,
        "forward_periods": 1,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed"] == "Mean($close, 20)"
    assert body["best_expression"]
    assert isinstance(body["history"], list)
    # 每步候选都是字符串（历史轨迹完整）
    assert len(body["history"]) >= 1


def test_factor_search_lt2_symbols(client):
    """少于 2 个标的 → 400。"""
    r = client.post("/factor/search", json={
        "seed": "Mean($close, 20)",
        "symbols": ["rb0"],
    })
    assert r.status_code == 400
    assert "error" in r.json() or "detail" in r.json()
