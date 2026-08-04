"""新增功能端到端测试：席位因子 / Optuna 优化 / 订单持仓 / 数据管理 / 告警配置。"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
from quantmind.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ----------------------------------------------------------------- 席位因子
def test_seat_factor_list(client):
    r = client.get("/seat-factors")
    assert r.status_code == 200
    factors = r.json().get("factors", {})
    # F1-F8 全注册
    for name in ("F1_net_position", "F3_net_ratio", "F7_net_zscore", "F8_seat_sentiment"):
        assert name in factors


def test_seat_factor_compute_missing_root(client):
    """根目录不存在时应返回 error 而非抛 500。"""
    r = client.post("/seat-factor", json={
        "symbol": "RB", "exchange": "SHFE", "interval": "1d",
        "seat_data_root": "/nonexistent/root", "factor": "F7_net_zscore",
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" in body or "ic_mean" in body


# ----------------------------------------------------------------- Optuna
def test_optimize_optuna(client):
    r = client.post("/optimize/optuna", json={
        "strategy": "dual_ma", "symbol": "rb0", "exchange": "SHFE",
        "method": "optuna",
        "param_ranges": {"fast": [3, 15, 2], "slow": [20, 60, 5]},
        "n_trials": 4, "metric": "sharpe",
    })
    assert r.status_code == 200
    body = r.json()
    assert body.get("method") == "optuna"
    assert "best_setting" in body
    assert "fast" in body.get("best_setting", {})
    assert body.get("combos", 0) >= 1


def test_optimize_optuna_requires_ranges(client):
    r = client.post("/optimize/optuna", json={
        "strategy": "dual_ma", "symbol": "rb0", "exchange": "SHFE",
        "method": "optuna", "param_ranges": {}, "n_trials": 4,
    })
    assert r.status_code == 400


# ----------------------------------------------------------------- 订单 / 持仓 / 撤单
def test_order_positions_cancel_flow(client):
    # 下单
    r = client.post("/order", json={
        "vt_symbol": "rb0.SHFE", "direction": "多", "offset": "开",
        "volume": 3, "price": 0.0,
    })
    assert r.status_code == 200
    oid = r.json().get("order_id")
    assert oid

    # 订单历史
    r = client.get("/orders")
    assert r.status_code == 200
    assert any(o["order_id"] == oid for o in r.json()["orders"])

    # 持仓推导
    r = client.get("/positions")
    assert r.status_code == 200
    pos = [p for p in r.json()["positions"] if p["vt_symbol"] == "rb0.SHFE"]
    assert pos and pos[0]["net_volume"] == 3

    # 撤单
    r = client.delete(f"/order/{oid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r = client.delete(f"/order/{oid}")
    assert r.json()["ok"] is False  # 重复撤单返回失败


# ----------------------------------------------------------------- 本地数据 / 告警
def test_settings_data_get(client):
    """GET /settings/data 应返回 5 个本地数据路径字段。"""
    r = client.get("/settings/data")
    assert r.status_code == 200
    body = r.json()
    for k in ("local_data_root", "local_stock_root", "local_hk_root",
              "local_option_root", "seat_data_root"):
        assert k in body


def test_settings_data_service_roundtrip(tmp_path, monkeypatch):
    """DataSettingsService：JSON 持久化 + .env 双写，路径隔离在 tmp_path（不污染项目配置）。"""
    monkeypatch.chdir(tmp_path)
    from quantmind.api.services.data_settings_service import DataSettingsService, _ENV_MAP
    svc = DataSettingsService()
    svc.path = tmp_path / "data_settings.json"
    out = svc.save({"local_stock_root": "/tmp/astock", "local_hk_root": "/tmp/hk"})
    assert out["local_stock_root"] == "/tmp/astock"
    assert out["synced_env"] is True
    # .env 双写
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "QM_LOCAL_STOCK_ROOT" in env_text


def test_data_files(client):
    r = client.get("/data/files")
    assert r.status_code == 200
    body = r.json()
    assert "groups" in body and "total_files" in body


def test_settings_alert_get(client):
    """GET /settings/alert 应返回告警配置默认字段。"""
    r = client.get("/settings/alert")
    assert r.status_code == 200
    body = r.json()
    assert "webhook_url" in body and "enabled" in body


def test_settings_alert_service_roundtrip(tmp_path, monkeypatch):
    """AlertSettingsService：配置持久化，路径隔离在 tmp_path。"""
    monkeypatch.chdir(tmp_path)
    from quantmind.api.services.alert_settings_service import AlertSettingsService
    svc = AlertSettingsService()
    svc.path = tmp_path / "alert_settings.json"
    out = svc.save({
        "enabled": True, "channel": "telegram",
        "webhook_url": "https://api.telegram.org/botX/sendMessage", "chat_id": "123",
    })
    assert out["webhook_url"] == "https://api.telegram.org/botX/sendMessage"
    assert out["enabled"] is True
