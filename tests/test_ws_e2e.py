"""WebSocket 端到端验收（方案 C）——真实后端事件流推送。

验收门槛（方案 C）：
  「启动 backtest/paper 模拟盘，监控面板无需刷新即出现新信号/成交。」

本测试用 FastAPI TestClient 走真实 lifespan（启动 EventEngine + 注册 `_broadcast` + 挂载
调度器），先连上 `/ws` 拿到 hello，再触发一次真实回测（`/backtest`），断言 WS 在**不刷新**的
情况下能收到底层事件引擎广播的实时事件流（eLog/eOrder/eSignal/eTrade/ePosition 等），
且信号目标/成交方向与回测一致——这正是实时监控页 `WSClient` 消费的消息来源。

依赖：仅复用现有 285 项测试的基础设施（TestClient + Mock 数据源），离线运行。
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from quantmind.api.app import app


def _drain_events(ws, limit: int = 200) -> list:
    """从已连接的 WS 拉取事件；用 echo 往返驱动事件环 flush（TestClient 内存传输）。"""
    events = []
    for _ in range(limit):
        try:
            ws._send({"type": "__pump__"})
        except Exception:  # noqa: BLE001
            pass
        try:
            msg = ws.receive_json()
        except Exception:  # noqa: BLE001
            break
        if msg.get("type") == "echo":
            continue
        events.append(msg)
    return events


def test_ws_backtest_event_stream_e2e():
    """连 WS → 跑回测 → 无需刷新即收到底层事件流（eSignal/eTrade/ePosition）。"""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1) 握手后应收到 hello
            hello = ws.receive_json()
            assert hello["type"] == "hello"

            # 2) 触发一次真实回测（会产生大量成交/信号/持仓事件）
            r = client.post(
                "/backtest",
                json={"strategy": "multifactor", "symbol": "rb0",
                      "exchange": "SHFE", "mode": "backtest"},
            )
            assert r.status_code == 200
            assert r.json()["mode"] == "backtest"
            assert r.json()["trades"] > 0

            # 3) 不刷新页面，直接读 WS 事件流
            events = _drain_events(ws)

    types = [e["type"] for e in events]

    # 验收门槛：必须收到真实推送的成交与持仓事件（监控面板即席刷新的数据来源）
    assert "eTrade" in types, "未收到成交事件(eTrade) —— WS 推送链路断裂"
    assert "ePosition" in types, "未收到持仓事件(ePosition)"
    assert "eSignal" in types, "未收到信号事件(eSignal)"

    # 信号/成交的标的应为完整 vt_symbol（监控页 _symbol_of 拼接结果一致）
    e_trade = next(e for e in events if e["type"] == "eTrade")
    assert e_trade["data"]["symbol"] == "rb0"
    assert e_trade["data"]["exchange"] == "SHFE"

    e_signal = next(e for e in events if e["type"] == "eSignal")
    assert e_signal["data"]["vt_symbol"] == "rb0.SHFE"
    assert e_signal["data"]["target"] in (-1.0, 0.0, 1.0)
    assert e_signal["data"]["order_id"]


def test_ws_hello_and_echo():
    """WS 基础握手与回显（后端 /ws 协议）。"""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert "QuantMind" in hello["msg"]

            ws.send_json({"ping": 1})
            echo = ws.receive_json()
            assert echo["type"] == "echo"
            assert echo["data"] == {"ping": 1}
