"""国内期货数据源：TqSdk（天勤量化）。

TqSdk 免费版提供 8000 根 K 线（无论周期），远超新浪的 1023 根限制：
  - 1m: 8000 根 ≈ 47 天
  - 5m: 8000 根 ≈ 251 天
  - 60m: 8000 根 ≈ 6.6 年
  - 日线: 2586 根 ≈ 10.6 年

合约代码映射：
  - 股指期货主力连续：IF0 → KQ.m@CFFEX.IF（沪深300主连）
  - 具体合约：IF2609 → CFFEX.IF2609

注意：TqSdk 需要快期账户认证（免费注册 https://www.shinnytech.com/register）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from .base import BaseDataFeed, HistoryRequest
from ...core.constant import Exchange, Interval
from ...core.object import BarData

_logger = logging.getLogger("quantmind.data.tqsdk_feed")


# 股指期货主力连续合约映射（TqSdk 格式）
STOCK_INDEX_FUTURES_TQSDK = {
    "IF0": "KQ.m@CFFEX.IF",  # 沪深300 主力连续
    "IC0": "KQ.m@CFFEX.IC",  # 中证500 主力连续
    "IH0": "KQ.m@CFFEX.IH",  # 上证50 主力连续
    "IM0": "KQ.m@CFFEX.IM",  # 中证1000 主力连续
}

# 周期映射：Interval → TqSdk duration_seconds
INTERVAL_TO_DURATION = {
    Interval.MINUTE: 60,       # 1分钟
    Interval.MINUTE_3: 180,    # 3分钟
    Interval.MINUTE_5: 300,    # 5分钟
    Interval.MINUTE_15: 900,   # 15分钟
    Interval.MINUTE_30: 1800,  # 30分钟
    Interval.HOUR: 3600,       # 60分钟
    Interval.HOUR_2: 7200,     # 2小时
    Interval.HOUR_4: 14400,    # 4小时
    Interval.DAILY: 86400,     # 日线
}


def _get_tqsdk_symbol(symbol: str, exchange: Exchange) -> Optional[str]:
    """将内部合约代码映射到 TqSdk 格式。"""
    # 先查主力连续映射表
    if symbol in STOCK_INDEX_FUTURES_TQSDK:
        return STOCK_INDEX_FUTURES_TQSDK[symbol]
    
    # 具体合约：IF2609 → CFFEX.IF2609
    if exchange == Exchange.CFFEX and symbol.startswith(("IF", "IC", "IH", "IM")):
        return f"CFFEX.{symbol}"
    
    # 商品期货：rb2609 → SHFE.rb2609（需要知道交易所）
    # TODO: 实现商品期货的交易所自动推断
    _logger.debug(f"合约 {symbol} 的 TqSdk 映射暂未实现")
    return None


class TqSdkFeed(BaseDataFeed):
    """TqSdk 数据源（天勤量化）。"""
    
    name = "tqsdk"
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """初始化 TqSdk 数据源。
        
        Args:
            username: 快期账户（邮箱或手机号），默认从环境变量 QM_TQSDK_USER 读取
            password: 快期密码，默认从环境变量 QM_TQSDK_PASS 读取
        """
        self._username = username or os.environ.get("QM_TQSDK_USER")
        self._password = password or os.environ.get("QM_TQSDK_PASS")
        self._api = None
        self._lock = asyncio.Lock()
        self._fetch_lock = asyncio.Lock()  # TqApi 非线程安全，fetch 串行化
        self._thread = None
        self._thread_loop = None
    
    def _ensure_thread_loop(self):
        """确保有一个专用线程和事件循环供 TqSdk 使用。"""
        if self._thread is not None and self._thread.is_alive():
            return
        
        import threading
        
        def thread_main():
            self._thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._thread_loop)
            self._thread_loop.run_forever()
        
        self._thread = threading.Thread(target=thread_main, daemon=True)
        self._thread.start()
        # 等待循环启动
        import time
        for _ in range(50):
            if self._thread_loop is not None:
                break
            time.sleep(0.1)
    
    async def _ensure_api(self):
        """确保 TqApi 已连接（懒加载 + 单例）。"""
        if self._api is not None:
            return self._api
        
        async with self._lock:
            # 双重检查
            if self._api is not None:
                return self._api
            
            if not self._username or not self._password:
                raise ValueError(
                    "TqSdk 需要快期账户认证。请设置环境变量 QM_TQSDK_USER 和 QM_TQSDK_PASS，"
                    "或在初始化时传入 username 和 password。"
                    "免费注册: https://www.shinnytech.com/register"
                )
            
            def _connect():
                from tqsdk import TqApi, TqAuth
                return TqApi(auth=TqAuth(self._username, self._password))
            
            # TqSdk 需要在有事件循环的线程中运行
            self._ensure_thread_loop()
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(_connect),
                self._thread_loop
            )
            self._api = await asyncio.wrap_future(future)
            _logger.info("TqSdk 连接成功: %s", self._username)
            return self._api
    
    async def fetch_bar_data(self, req: HistoryRequest) -> List[BarData]:
        """获取期货历史 K 线数据。"""
        # TqApi 非线程安全：串行化并发请求，避免多线程同时操作同一 api 实例
        async with self._fetch_lock:
            return await self._fetch_bar_data_locked(req)

    async def _fetch_bar_data_locked(self, req: HistoryRequest) -> List[BarData]:
        api = await self._ensure_api()
        
        tq_symbol = _get_tqsdk_symbol(req.symbol, req.exchange)
        if not tq_symbol:
            _logger.warning(f"无法映射 {req.symbol} 到 TqSdk 合约代码，跳过")
            return []
        
        duration = INTERVAL_TO_DURATION.get(req.interval)
        if not duration:
            _logger.warning(f"不支持的周期 {req.interval}，跳过")
            return []
        
        # TqSdk 免费版最多 8000 根
        data_length = 8000
        
        def _fetch():
            klines = api.get_kline_serial(tq_symbol, duration, data_length=data_length)
            # 等待数据就绪（wait_update 可能长时间阻塞，外层用 asyncio.wait_for 兜底超时）
            import time
            deadline = time.time() + 90
            while time.time() < deadline:
                # 最大阻塞 3 秒，留出外层检查取消/超时的机会
                api.wait_update(deadline=time.time() + 3)
                valid = klines.dropna(subset=['close'])
                if len(valid) > 0:
                    return klines
            raise TimeoutError(f"获取 {tq_symbol} K线超时")
        
        try:
            # 在专用线程中执行 TqSdk 操作；wait_for 兜底超时(120s)并使其可被取消
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(_fetch),
                self._thread_loop
            )
            klines = await asyncio.wait_for(asyncio.wrap_future(future), timeout=120)
        except asyncio.TimeoutError:
            _logger.error(f"TqSdk 获取 {tq_symbol} 超时(120s)")
            return []
        except Exception as e:
            _logger.error(f"TqSdk 获取 {tq_symbol} 失败: {e}")
            return []
        
        return self._df_to_bars(klines, req.symbol, req.exchange, req.interval)
    
    @staticmethod
    def _df_to_bars(df: pd.DataFrame, symbol: str, exchange: Exchange, interval: Interval) -> List[BarData]:
        """将 TqSdk DataFrame 转换为 BarData 列表。"""
        bars: List[BarData] = []
        
        # 过滤无效数据（NaN）
        valid = df.dropna(subset=['close'])
        if valid.empty:
            return bars
        
        for _, row in valid.iterrows():
            try:
                # TqSdk datetime 是纳秒时间戳
                dt_ns = int(row['datetime'])
                # 归一为 naive UTC（与全仓数据体系一致，避免 tz-aware/naive 混比报错）
                dt = datetime.fromtimestamp(dt_ns / 1e9, tz=timezone.utc).replace(tzinfo=None)
                
                bars.append(BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=float(row.get('volume', 0)),
                    open_interest=float(row.get('close_oi', 0)),
                    turnover=0.0,  # TqSdk K线暂无成交额字段
                ))
            except (ValueError, TypeError, KeyError) as e:
                _logger.debug(f"解析行失败: {e}")
                continue
        
        return bars
    
    async def close(self):
        """关闭 TqApi 连接。"""
        if self._api is not None:
            def _close():
                self._api.close()
            try:
                future = asyncio.run_coroutine_threadsafe(
                    asyncio.to_thread(_close),
                    self._thread_loop
                )
                await asyncio.wrap_future(future)
            except Exception:
                pass
            self._api = None
            _logger.info("TqSdk 连接已关闭")
