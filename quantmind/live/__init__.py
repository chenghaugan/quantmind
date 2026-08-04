"""实盘模块：CTP / XTP / IB 网关桩 + SIM 模拟券商 + 实盘路由引擎 + 订单状态机 + 对账。"""
from .ctp_gateway import CtpGateway
from .xtp_gateway import XtpGateway
from .ib_gateway import IbGateway
from .sim import SimGateway, simulate_one_trade
from .order_manager import ManagedOrder, OrderManager, ACTIVE_STATUSES, FINAL_STATUSES
from .reconcile import PositionDiff, ReconcileReport, reconcile, reconcile_positions
from .runner import LiveEngine

# 网关工厂（按名称构造，接入凭证即可启用；sim=在进程内撮合的模拟券商，离线可测）
GATEWAY_REGISTRY = {
    "ctp": CtpGateway,
    "xtp": XtpGateway,
    "ib": IbGateway,
    "sim": SimGateway,
}


def build_gateway(name: str, event_engine, settings: dict | None = None):
    cls = GATEWAY_REGISTRY.get(name.lower())
    if cls is None:
        raise KeyError(f"未知网关: {name}")
    gw = cls(event_engine)
    gw.connect(settings or {})
    return gw


__all__ = [
    "CtpGateway",
    "XtpGateway",
    "IbGateway",
    "SimGateway",
    "simulate_one_trade",
    "LiveEngine",
    "OrderManager",
    "ManagedOrder",
    "ACTIVE_STATUSES",
    "FINAL_STATUSES",
    "ReconcileReport",
    "PositionDiff",
    "reconcile",
    "reconcile_positions",
    "GATEWAY_REGISTRY",
    "build_gateway",
]
