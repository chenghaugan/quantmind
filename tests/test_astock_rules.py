"""A股交易规则专项测试。

测试三个核心规则：
1. T+1限制：当日买入的股票当日不能卖出
2. 整手交易：买入数量必须是100的整数倍
3. 涨跌停价格限制：委托价格不能超出涨跌停价
"""
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import pytest

from quantmind.backtest.engine import BacktestEngine
from quantmind.backtest.diagnostics import limit_price_range
from quantmind.core.object import BarData
from quantmind.core.constant import Direction, Exchange, Offset, Interval
from quantmind.core.gateway import OrderRequest


def _bar(symbol: str, exchange: Exchange, dt: datetime,
         o: float, h: float, l: float, c: float, v: float = 1000.0) -> BarData:
    return BarData(
        symbol=symbol, exchange=exchange, datetime=dt,
        interval=Interval.DAILY,
        open_price=o, high_price=h, low_price=l, close_price=c, volume=v,
    )


def _astock_bars(code: str, dates, prices):
    """快速构造A股日K数据。dates: list[datetime], prices: list[(o,h,l,c)]"""
    exch = Exchange.SSE if code.startswith("6") else Exchange.SZSE
    return [_bar(code, exch, d, *p) for d, p in zip(dates, prices)]


# ============================================================
# 1. 整手交易规则
# ============================================================

class TestLotSize:
    """A股买入必须是100股整数倍。"""

    def _engine(self, code="600000"):
        # 两根 bar：send_order 要求存在下一交易日可撮合（末根 bar 拒单）
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        bars = _astock_bars(code, [d0, d1], [(10.0, 10.5, 9.5, 10.0),
                                             (10.0, 10.5, 9.5, 10.0)])
        eng = BacktestEngine({f"{code}.{Exchange.SSE.value}": bars})
        eng._current_date = d0
        return eng

    def test_buy_round_down_to_lot(self):
        """买入150股 → 自动调整为100股。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=150, price=10.0)
        oid = eng.send_order(req)
        assert oid, "整手调整后应成功下单"
        # 检查 pending 里的实际数量
        assert eng.pending[-1]["req"].volume == 100

    def test_buy_250_becomes_200(self):
        """买入250股 → 调整为200股。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=250, price=10.0)
        oid = eng.send_order(req)
        assert oid
        assert eng.pending[-1]["req"].volume == 200

    def test_buy_less_than_one_lot_rejected(self):
        """买入50股（不足1手）→ 拒单返回空字符串。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=50, price=10.0)
        oid = eng.send_order(req)
        assert oid == ""

    def test_sell_not_rounded(self):
        """卖出不要求整手（可卖零股）。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.SHORT, offset=Offset.CLOSE,
                           volume=73, price=10.0)
        oid = eng.send_order(req)
        assert oid, "卖出不要求整手"
        assert eng.pending[-1]["req"].volume == 73

    def test_exact_lot_no_change(self):
        """整手数量不变。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=300, price=10.0)
        oid = eng.send_order(req)
        assert oid
        assert eng.pending[-1]["req"].volume == 300

    def test_non_astock_not_rounded(self):
        """非A股（期货）不做整手调整。"""
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        bars = [_bar("rb2505", Exchange.SHFE, d0, 3500, 3550, 3450, 3500),
                _bar("rb2505", Exchange.SHFE, d1, 3500, 3550, 3450, 3500)]
        eng = BacktestEngine({"rb2505.SHFE": bars})
        eng._current_date = d0
        req = OrderRequest(symbol="rb2505", exchange=Exchange.SHFE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=3, price=3500)
        oid = eng.send_order(req)
        assert oid
        assert eng.pending[-1]["req"].volume == 3


# ============================================================
# 2. 涨跌停价格限制
# ============================================================

class TestLimitPrice:
    """涨跌停价格限制：委托价超出涨跌停范围则拒单。"""

    def _engine(self):
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        # Day0 收盘10元 → Day1 涨停11元 / 跌停9元
        bars = _astock_bars("600000", [d0, d1], [
            (10.0, 10.5, 9.5, 10.0),
            (10.0, 10.5, 9.5, 10.2),
        ])
        eng = BacktestEngine(
            {"600000.SSE": bars},
            exclude_limit=True,
            limit_pct=0.10,
        )
        # 将 _current_date 设为 d0（存在下一交易日可撮合）
        eng._current_date = d0
        return eng

    def test_price_above_limit_up_rejected(self):
        """委托价超涨停价 → 拒单。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=100, price=11.50)
        oid = eng.send_order(req)
        assert oid == ""

    def test_price_below_limit_down_rejected(self):
        """委托价低于跌停价 → 拒单。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.SHORT, offset=Offset.CLOSE,
                           volume=100, price=8.50)
        oid = eng.send_order(req)
        assert oid == ""

    def test_price_within_range_accepted(self):
        """委托价在涨跌停范围内 → 正常下单。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=100, price=10.50)
        oid = eng.send_order(req)
        assert oid != ""

    def test_price_at_limit_accepted(self):
        """委托价恰好等于涨停价 → 接受。"""
        eng = self._engine()
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=100, price=11.00)
        oid = eng.send_order(req)
        assert oid != ""

    def test_no_exclude_limit_no_check(self):
        """未启用 exclude_limit 时不做价格限制。"""
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        bars = _astock_bars("600000", [d0, d1], [
            (10.0, 10.5, 9.5, 10.0),
            (10.0, 10.5, 9.5, 10.2),
        ])
        eng = BacktestEngine({"600000.SSE": bars}, exclude_limit=False)
        eng._current_date = d0
        req = OrderRequest(symbol="600000", exchange=Exchange.SSE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=100, price=15.00)
        oid = eng.send_order(req)
        assert oid != ""

    def test_non_astock_no_limit_check(self):
        """非A股不做涨跌停价格限制。"""
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        bars = [_bar("rb2505", Exchange.SHFE, d0, 3500, 3550, 3450, 3500),
                _bar("rb2505", Exchange.SHFE, d1, 3500, 3550, 3450, 3500)]
        eng = BacktestEngine({"rb2505.SHFE": bars}, exclude_limit=True, limit_pct=0.10)
        eng._current_date = d0
        req = OrderRequest(symbol="rb2505", exchange=Exchange.SHFE,
                           direction=Direction.LONG, offset=Offset.OPEN,
                           volume=1, price=5000)
        oid = eng.send_order(req)
        assert oid != ""


# ============================================================
# 3. T+1 交易限制
# ============================================================

class TestT1Rule:
    """A股T+1：当日买入的股票当日不能卖出。"""

    def _setup_engine_with_position(self):
        """构造一个引擎，在 Day0 买入，然后测试 Day0/Day1 卖出。"""
        d0 = datetime(2025, 1, 2)
        d1 = datetime(2025, 1, 3)
        d2 = datetime(2025, 1, 6)
        bars = _astock_bars("600000", [d0, d1, d2], [
            (10.0, 10.5, 9.5, 10.0),
            (10.0, 10.5, 9.5, 10.2),
            (10.2, 10.8, 10.0, 10.5),
        ])
        eng = BacktestEngine({"600000.SSE": bars})
        return eng, d0, d1, d2

    def test_check_t1_sell_blocks_same_day(self):
        """_check_t1_sell：当日买入后当日卖出应被阻止。"""
        eng, d0, d1, d2 = self._setup_engine_with_position()
        # 模拟 Day0 买入 100 股
        eng._apply_fill("600000.SSE", Direction.LONG, 100, 10.0, Offset.OPEN, d0)
        # 当日卖出 → 应被阻止
        assert not eng._check_t1_sell("600000.SSE", 100, d0)

    def test_check_t1_sell_allows_next_day(self):
        """_check_t1_sell：次日卖出应允许。"""
        eng, d0, d1, d2 = self._setup_engine_with_position()
        eng._apply_fill("600000.SSE", Direction.LONG, 100, 10.0, Offset.OPEN, d0)
        # 次日卖出 → 应允许
        assert eng._check_t1_sell("600000.SSE", 100, d1)

    def test_check_t1_partial_sell(self):
        """_check_t1_sell：今日买入100+昨日持仓200，今日可卖200。"""
        eng, d0, d1, d2 = self._setup_engine_with_position()
        # Day0 买入 200 股（昨日仓）
        eng._apply_fill("600000.SSE", Direction.LONG, 200, 10.0, Offset.OPEN, d0)
        # Day1 又买入 100 股（今日仓）
        eng._apply_fill("600000.SSE", Direction.LONG, 100, 10.2, Offset.OPEN, d1)
        # Day1 卖 200 股 → 应允许（只卖昨日仓）
        assert eng._check_t1_sell("600000.SSE", 200, d1)
        # Day1 卖 300 股 → 应阻止（超出可卖量）
        assert not eng._check_t1_sell("600000.SSE", 300, d1)

    def test_non_astock_not_affected(self):
        """非A股不受T+1限制。"""
        d0 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        d1 = datetime(2025, 1, 3, tzinfo=timezone.utc)
        bars = [_bar("rb2505", Exchange.SHFE, d0, 3500, 3550, 3450, 3500),
                _bar("rb2505", Exchange.SHFE, d1, 3500, 3550, 3450, 3500)]
        eng = BacktestEngine({"rb2505.SHFE": bars})
        eng._current_date = d0
        eng._apply_fill("rb2505.SHFE", Direction.LONG, 1, 3500, Offset.OPEN, d0)
        # 期货当日卖出 → 应允许
        assert eng._check_t1_sell("rb2505.SHFE", 1, d0)


# ============================================================
# 4. limit_price_range 函数测试
# ============================================================

class TestLimitPriceRange:
    """diagnostics.limit_price_range 辅助函数测试。"""

    def test_basic(self):
        d0 = datetime(2025, 1, 2)
        d1 = datetime(2025, 1, 3)
        bars = [_bar("600000", Exchange.SSE, d0, 10, 10.5, 9.5, 10.0),
                _bar("600000", Exchange.SSE, d1, 10, 10.5, 9.5, 10.2)]
        ld, lu = limit_price_range(bars, 1, 0.10)
        assert ld == 9.0
        assert lu == 11.0

    def test_first_bar_returns_none(self):
        d0 = datetime(2025, 1, 2)
        bars = [_bar("600000", Exchange.SSE, d0, 10, 10.5, 9.5, 10.0)]
        ld, lu = limit_price_range(bars, 0, 0.10)
        assert ld is None
        assert lu is None

    def test_zero_prev_close(self):
        d0 = datetime(2025, 1, 2)
        d1 = datetime(2025, 1, 3)
        bars = [_bar("600000", Exchange.SSE, d0, 0, 0, 0, 0),
                _bar("600000", Exchange.SSE, d1, 10, 10.5, 9.5, 10.2)]
        ld, lu = limit_price_range(bars, 1, 0.10)
        assert ld is None
        assert lu is None

    def test_st_stock_5pct(self):
        """ST股涨跌停幅度5%。"""
        d0 = datetime(2025, 1, 2)
        d1 = datetime(2025, 1, 3)
        bars = [_bar("600000", Exchange.SSE, d0, 10, 10.5, 9.5, 10.0),
                _bar("600000", Exchange.SSE, d1, 10, 10.5, 9.5, 10.2)]
        ld, lu = limit_price_range(bars, 1, 0.05)
        assert ld == 9.50
        assert lu == 10.50
