"""API 端到端测试（FastAPI TestClient，离线）。"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
from quantmind.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_data_offline(client):
    r = client.get("/data", params={"symbol": "rb0", "exchange": "SHFE", "interval": "1d"})
    assert r.status_code == 200
    body = r.json()
    # 新分页结构: {"data": [...], "pagination": {...}}
    assert "data" in body
    assert "pagination" in body
    bars = body["data"]
    assert len(bars) > 0
    assert "close" in bars[0]
    # 验证分页字段
    assert body["pagination"]["page"] == 1
    assert body["pagination"]["total"] > 0


def test_research(client):
    r = client.post("/research", json={"idea": "螺纹钢期货动量策略", "asset_class": "期货"})
    assert r.status_code == 200
    body = r.json()
    assert body["asset_class"]
    assert len(body["generated_factors"]) >= 1


def test_factor(client):
    r = client.post("/factor", json={"symbol": "rb0", "exchange": "SHFE",
                                     "factor": "momentum_20", "forward_periods": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_samples"] > 0


def test_backtest(client):
    r = client.post("/backtest", json={"strategy": "multifactor", "symbol": "rb0",
                                       "exchange": "SHFE", "mode": "backtest"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "backtest"
    assert body["trades"] > 0


def test_lifecycle_blocked(client):
    r = client.post("/lifecycle", json={"strategy_id": "t1", "to": "LIVE",
                                       "metrics": {"sharpe": 0.1, "max_drawdown": -0.1}})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_strategies_list(client):
    r = client.get("/strategies")
    assert r.status_code == 200
    assert len(r.json()) >= 2
