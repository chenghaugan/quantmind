"""LifecycleService: 策略生命周期晋升管理 + 订单/持仓台账"""
from typing import Dict, Any, List
from datetime import datetime
import uuid

from ...paper.promotion import LifecycleManager, LifecycleState
from ...core.engine import EventEngine
from ...core.constant import Exchange, Direction, Offset
from ...core.object import OrderData, Status
from ...core.event import Event, EventType
from ..schemas import LifecycleRequest, OrderRequestSchema


class LifecycleService:
    def __init__(self, lifecycle: LifecycleManager, ee: EventEngine):
        self.lifecycle = lifecycle
        self.ee = ee
        # 内存台账：订单历史 + 由此推导的净持仓（无需依赖实盘网关即可驱动监控页）
        self._orders: List[Dict[str, Any]] = []
        self._seq = 0

    async def promote(self, req: LifecycleRequest) -> Dict[str, Any]:
        try:
            to = LifecycleState(req.to)
        except ValueError:
            return {"ok": False, "msg": f"非法状态: {req.to}"}
        ok, reasons = self.lifecycle.promote(req.strategy_id, to, req.metrics, req.note)
        return {
            "ok": ok,
            "state": self.lifecycle.get_or_create(req.strategy_id).state.value,
            "reasons": reasons,
        }

    # ----------------------------------------------------------------- 台账
    @staticmethod
    def _apply_position(position, vt_symbol: str, direction, offset, volume):
        """按开平与方向更新净持仓。"""
        delta = volume
        if direction == Direction.SHORT:
            delta = -delta
        # 平仓方向与开仓相反
        if offset in (Offset.CLOSE, Offset.CLOSE_TODAY, Offset.CLOSE_YESTERDAY):
            delta = -delta
        position["vt_symbol"] = vt_symbol
        position["net_volume"] = round(position.get("net_volume", 0.0) + delta, 4)
        position["updated"] = datetime.now().isoformat()
        return position

    async def place_order(self, req: OrderRequestSchema) -> Dict[str, Any]:
        direction = Direction(req.direction)
        offset = Offset(req.offset)
        sym, exch = req.vt_symbol.rsplit(".", 1)
        order_id = f"WEB-{uuid.uuid4().hex[:8].upper()}"
        self._seq += 1
        od = OrderData(
            symbol=sym,
            exchange=Exchange(exch),
            order_id=order_id,
            direction=direction,
            offset=offset,
            price=req.price,
            volume=req.volume,
            status=Status.SUBMITTED,
        )
        self.ee.put_event(EventType.EVENT_ORDER, od)
        from ...core.object import LogData
        self.ee.put_event(
            EventType.EVENT_LOG,
            LogData(msg=f"手动下单: {req.vt_symbol} {req.direction} x{req.volume}"),
        )
        self._orders.append({
            "order_id": order_id,
            "seq": self._seq,
            "vt_symbol": req.vt_symbol,
            "direction": req.direction,
            "offset": req.offset,
            "volume": req.volume,
            "price": req.price,
            "status": "已报",
            "datetime": datetime.now().isoformat(),
        })
        return {"ok": True, "order": req.vt_symbol, "order_id": order_id}

    def list_orders(self) -> Dict[str, Any]:
        orders = [o for o in reversed(self._orders)]
        return {"orders": orders, "count": len(orders)}

    def list_positions(self) -> Dict[str, Any]:
        positions: Dict[str, Dict[str, Any]] = {}
        for o in self._orders:
            if o["status"] == "已撤":
                continue
            direction = Direction(o["direction"])
            offset = Offset(o["offset"])
            pos = positions.setdefault(
                o["vt_symbol"], {"vt_symbol": o["vt_symbol"], "net_volume": 0.0,
                                 "avg_price": 0.0, "updated": ""}
            )
            self._apply_position(pos, o["vt_symbol"], direction, offset, o["volume"])
        rows = [p for p in positions.values() if abs(p["net_volume"]) > 1e-6]
        return {"positions": rows, "count": len(rows)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        for o in self._orders:
            if o["order_id"] == order_id:
                if o["status"] == "已撤":
                    return {"ok": False, "msg": "订单已撤销"}
                o["status"] = "已撤"
                o["canceled"] = datetime.now().isoformat()
                return {"ok": True, "order_id": order_id, "status": "已撤"}
        return {"ok": False, "msg": f"订单不存在: {order_id}"}
