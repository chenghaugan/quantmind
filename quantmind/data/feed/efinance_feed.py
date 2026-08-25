"""国内期货数据源：efinance（东方财富）。

efinance 基于东方财富 API，支持期货日线/分钟线数据。
相比 akshare（新浪），efinance 的分钟数据没有 1023 根硬限制，可能获取更长历史。

行情 ID 映射：
  - 股指期货主力：IF0 → 8.IF0, IC0 → 8.IC0, IH0 → 8.IH0, IM0 → 8.IM0
  - 商品期货主力：rb0 → 115.rb0, cu0 → 115.cu0 等（需动态查询）

注意：efinance 为同步库，用 asyncio.to_thread 包裹避免阻塞事件循环；
导入延迟化，未安装 efinance 时本模块仍可导入（仅运行时报缺依赖）。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.efinance_feed")


# 股指期货主力行情 ID 映射
STOCK_INDEX_FUTURES_QUOTE_IDS = {
    "IF0": "8.IF0",  # 沪深300主力
    "IC0": "8.IC0",  # 中证500主力
    "IH0": "8.IH0",  # 上证50主力
    "IM0": "8.IM0",  # 中证1000主力
}

# 周期映射：Interval → efinance klt 参数
INTERVAL_TO_KLT = {
    Interval.MINUTE: 1,      # 1分钟
    Interval.MINUTE_5: 5,    # 5分钟
    Interval.MINUTE_15: 15,  # 15分钟
    Interval.MINUTE_30: 30,  # 30分钟
    Interval.HOUR: 60,       # 60分钟
    Interval.DAILY: 101,     # 日线
    Interval.WEEKLY: 102,    # 周线
}


def _parse_dt(value) -> datetime:
    """解析 efinance 返回的日期/时间字段。"""
    if isinstance(value, datetime):
        return value
    s = str(value)
    # efinance 返回格式：2024-01-15 或 2024-01-15 09:30:00
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(s)


def _find_column(df, candidates: List[str]) -> Optional[str]:
    """在 DataFrame 列名中查找候选列名（支持中文和英文）。"""
    for col in df.columns:
        col_lower = str(col).lower()
        for candidate in candidates:
            if candidate in col_lower:
                return col
    return None


class EfinanceFeed(BaseDataFeed):
    """efinance 数据源（东方财富）。"""
    
    name = "efinance"
    
    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        """获取期货历史数据。"""
        import efinance as ef
        
        symbol = req.symbol
        quote_id = self._get_quote_id(symbol)
        
        if not quote_id:
            _logger.warning(f"无法映射 {symbol} 到 efinance 行情 ID，跳过")
            return []
        
        klt = INTERVAL_TO_KLT.get(req.interval)
        if not klt:
            _logger.warning(f"不支持的周期 {req.interval}，跳过")
            return []
        
        # 调用 efinance API
        df = await asyncio.to_thread(
            ef.futures.get_quote_history,
            quote_id,
            klt=klt,
        )
        
        if df is None or len(df) == 0:
            _logger.warning(f"efinance 返回空数据: {symbol} ({quote_id})")
            return []
        
        return self._df_to_bars(df, symbol, req.exchange, req.interval)
    
    @staticmethod
    def _get_quote_id(symbol: str) -> Optional[str]:
        """将品种代码映射到 efinance 行情 ID。"""
        # 先查股指期货映射表
        if symbol in STOCK_INDEX_FUTURES_QUOTE_IDS:
            return STOCK_INDEX_FUTURES_QUOTE_IDS[symbol]
        
        # 商品期货需要动态查询（暂时返回 None，后续可扩展）
        # TODO: 实现商品期货行情 ID 动态查询
        _logger.debug(f"商品期货 {symbol} 的行情 ID 映射暂未实现")
        return None
    
    @staticmethod
    def _df_to_bars(df, symbol: str, exchange: Exchange, interval: Interval) -> List[BarData]:
        """将 efinance DataFrame 转换为 BarData 列表。"""
        bars: List[BarData] = []
        
        # efinance 返回的列名是中文：日期、开盘、收盘、最高、最低、成交量、成交额等
        date_col = _find_column(df, ["日期", "date", "时间", "datetime"])
        open_col = _find_column(df, ["开盘", "open"])
        high_col = _find_column(df, ["最高", "high"])
        low_col = _find_column(df, ["最低", "low"])
        close_col = _find_column(df, ["收盘", "close"])
        volume_col = _find_column(df, ["成交量", "volume"])
        amount_col = _find_column(df, ["成交额", "amount", "turnover"])
        
        if not all([date_col, open_col, high_col, low_col, close_col]):
            _logger.error(f"efinance 数据缺少关键列，实际列名={list(df.columns)}")
            return bars
        
        for _, row in df.iterrows():
            try:
                bars.append(
                    BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=_parse_dt(row[date_col]),
                        interval=interval,
                        open_price=float(row[open_col]),
                        high_price=float(row[high_col]),
                        low_price=float(row[low_col]),
                        close_price=float(row[close_col]),
                        volume=float(row[volume_col]) if volume_col and row[volume_col] else 0.0,
                        open_interest=0.0,  # efinance 期货数据暂无持仓量
                        turnover=float(row[amount_col]) if amount_col and row[amount_col] else 0.0,
                    )
                )
            except (ValueError, TypeError) as e:
                _logger.debug(f"解析行失败: {e}")
                continue
        
        return bars
    
    @staticmethod
    async def get_quote_ids() -> dict:
        """获取所有期货行情 ID 列表（用于调试和动态映射）。"""
        import efinance as ef
        
        try:
            df = await asyncio.to_thread(ef.futures.get_realtime_quotes)
            if df is None or len(df) == 0:
                return {}
            
            # 返回 {期货代码: 行情ID} 映射
            result = {}
            for _, row in df.iterrows():
                code = row.get("期货代码")
                quote_id = row.get("行情ID")
                if code and quote_id:
                    result[code] = quote_id
            return result
        except Exception as e:
            _logger.error(f"获取行情 ID 列表失败: {e}")
            return {}
