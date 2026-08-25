"""真实数据策略验证：动量 / 缠论3买 / 缠论1买 在股指期货上的有效性回测。

在真实日线数据（data_cache/{symbol}.CFFEX.{interval}.parquet，IF0/IC0/IH0/IM0）
上，对三类策略做框架级回测（BacktestEngine）：
  - 动量策略（MomentumCtaStrategy）：N 日动量阈值进出场，多空均可。
  - 缠论第三类买点（ChanThirdBuyStrategy）：复用 ``strategy/mined.py`` 的
    忠实实现（中枢上沿 ZG + 趋势均线 + 回抽不破）。
  - 缠论第一类买点（ChanFirstBuyStrategy）：本脚本实现的**近似**——下跌趋势
    末端「价格新低但动能（ROC）不创新低」的底背驰检测（确定性规则）。

用法：
    .\\venv\\Scripts\\python.exe scripts\\validate_strategies_index_futures.py \\
        [--symbol IC0] [--interval 1d] [--start 2017-01-01]

说明：
  - 数据不足 / 回测无成交时给出明确提示，不中断其他策略。
  - 缠论1买为确定性近似（真实笔/中枢/背驰需要更高频数据 + 更精细实现）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.core.constant import Exchange, Interval  # noqa: E402
from quantmind.core.contracts import default_size  # noqa: E402
from quantmind.core.object import BarData  # noqa: E402
from quantmind.core.utility import ArrayManager  # noqa: E402
from quantmind.strategy.base import CtaTemplate  # noqa: E402
from quantmind.strategy.mined import ChanThirdBuyStrategy  # noqa: E402
from quantmind.strategy.runners import run_strategy  # noqa: E402


# ---------------------------------------------------------------------------
# 1) 动量策略（多空）
# ---------------------------------------------------------------------------
class MomentumCtaStrategy(CtaTemplate):
    """N 日动量阈值策略：动量 > threshold 做多，< -threshold 做空，之间空仓。"""

    author = "QuantMind 验证"
    parameters = ["window", "threshold", "size", "max_pos"]

    def __init__(self, context, setting=None):
        self.window = 20
        self.threshold = 0.03
        self.size = 1
        self.max_pos = 1.0
        self.am = None
        self.last_target = 0.0
        super().__init__(context, setting)

    def on_bar(self, bar: BarData) -> None:
        if self.am is None:
            self.am = ArrayManager(self.window + 5)
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        closes = self.am.close
        mom = closes[-1] / closes[-self.window] - 1.0
        if mom > self.threshold:
            target = self.max_pos * self.size
        elif mom < -self.threshold:
            target = -self.max_pos * self.size
        else:
            target = 0.0
        if target != self.last_target:
            self.set_target(bar.vt_symbol, target)
            self.last_target = target
            self.pos = target


# ---------------------------------------------------------------------------
# 2) 缠论第一类买点（近似：下跌末端底背驰）
# ---------------------------------------------------------------------------
class ChanFirstBuyStrategy(CtaTemplate):
    """缠论第一类买点（确定性近似）。

    核心：下跌趋势末端出现「价格新低但动能衰减」的底背驰。
    规则：
      - 下跌趋势：close 在 trend_window 均线之下。
      - 背驰检测：连续两次创新低（low_window 内的最低点），后一次价格更低，
        但后一次低点处的 ROC(roc_window) 高于前一次（动能未创新低）→ 买入。
      - 离场：价格上穿 trend_window 均线（趋势反转）或跌破背驰低点（新低延续）。
    """

    author = "QuantMind 验证（近似）"
    parameters = ["trend_window", "low_window", "roc_window", "size", "max_pos"]

    def __init__(self, context, setting=None):
        self.trend_window = 60
        self.low_window = 20
        self.roc_window = 10
        self.size = 1
        self.max_pos = 1.0
        self.am = None
        #: 上一个低点记录：(价格, ROC, index)
        self.prev_low = None
        self.entry_low = None
        self.last_target = 0.0
        super().__init__(context, setting)

    def on_bar(self, bar: BarData) -> None:
        if self.am is None:
            self.am = ArrayManager(self.trend_window + self.roc_window + 5)
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        closes = self.am.close
        lows = self.am.low
        n = len(closes)
        if n < self.trend_window + self.roc_window:
            return

        ma = sum(closes[-self.trend_window:]) / self.trend_window
        last = closes[-1]
        down_trend = last < ma

        # 当前低点检测：最近 low_window 内的最低价（ArrayManager 属性为 list）
        seg = lows[-self.low_window:]
        cur_low = float(min(seg))
        cur_low_idx = n - 1 - seg[::-1].index(cur_low)
        if cur_low_idx - self.roc_window < 0:
            return
        cur_roc = closes[cur_low_idx] / closes[cur_low_idx - self.roc_window] - 1.0

        # 背驰：创新低且动能衰减
        if down_trend and self.prev_low is not None:
            prev_price, prev_roc, _ = self.prev_low
            if cur_low < prev_price and cur_roc > prev_roc:
                # 底背驰 → 做多
                self._set(bar.vt_symbol, self.max_pos * self.size)
                self.entry_low = cur_low
                self.prev_low = (cur_low, cur_roc, cur_low_idx)
                return
        # 更新低点记录（价格创新低时更新）
        if self.prev_low is None or cur_low < self.prev_low[0]:
            self.prev_low = (cur_low, cur_roc, cur_low_idx)

        # 离场：趋势反转 或 跌破背驰低点
        if self.last_target != 0.0:
            if last > ma or (self.entry_low is not None and last < self.entry_low):
                self._set(bar.vt_symbol, 0.0)
        elif self.last_target == 0.0:
            pass

    def _set(self, vt_symbol: str, target: float) -> None:
        if target != self.last_target:
            self.set_target(vt_symbol, target)
            self.last_target = target
            self.pos = target


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_bars(symbol: str, interval: str, start: Optional[str]) -> List[BarData]:
    """从 data_cache parquet 读真实数据 → BarData 列表。"""
    iv = Interval(interval) if interval in ("1d", "1h", "30m", "15m", "5m", "1m") else Interval.DAILY
    path = PROJECT / "data_cache" / f"{symbol}.CFFEX.{iv.value}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if start:
        df = df[df["datetime"] >= pd.Timestamp(start)]
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    bars = [
        BarData(
            symbol=symbol,
            exchange=Exchange.CFFEX,
            datetime=row.datetime.to_pydatetime(),
            interval=iv,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row.get("volume") or 0.0),
            open_interest=float(row.get("open_interest") or 0.0),
            turnover=float(row.get("turnover") or 0.0),
        )
        for _, row in df.iterrows()
    ]
    return bars


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="策略验证：动量/缠论3买/缠论1买 on 股指期货")
    ap.add_argument("--symbol", default="IC0", help="IC0/IF0/IH0/IM0")
    ap.add_argument("--interval", default="1d", help="1d/1h/30m/15m/5m/1m")
    ap.add_argument("--start", default=None, help="起始日期 2017-01-01")
    args = ap.parse_args()

    try:
        bars = load_bars(args.symbol, args.interval, args.start)
    except FileNotFoundError as exc:
        print(f"✗ {exc}")
        return 1
    if len(bars) < 200:
        print(f"✗ 数据不足（{len(bars)} 根 < 200），无法回测")
        return 1

    vt_symbol = f"{args.symbol}.CFFEX"
    # 合约乘数（盈亏 = 点数变化 × 乘数 × 手数），从 contracts 表读取
    multiplier = default_size(vt_symbol)
    sizes = {vt_symbol: multiplier}
    start = bars[0].datetime.strftime("%Y-%m-%d")
    end = bars[-1].datetime.strftime("%Y-%m-%d")
    print(f"标的 {vt_symbol}  周期 {args.interval}  {len(bars)} 根  {start} ~ {end}  "
          f"合约乘数 {multiplier}")
    print("=" * 78)

    strategies = [
        ("动量 (20日, ±3%)", MomentumCtaStrategy,
         {"window": 20, "threshold": 0.03, "size": 1, "max_pos": 1.0}),
        ("缠论3买 (中枢上沿ZG)", ChanThirdBuyStrategy,
         {"trend_window": 60, "break_window": 20, "size": 1, "max_pos": 1.0}),
        ("缠论1买 (底背驰近似)", ChanFirstBuyStrategy,
         {"trend_window": 60, "low_window": 20, "roc_window": 10, "size": 1, "max_pos": 1.0}),
    ]

    for name, cls, setting in strategies:
        print(f"\n▶ {name}")
        try:
            res = run_strategy("backtest", cls, vt_symbol, setting, bars, sizes=sizes)
            rep = res.get("report") or {}
            eq = res.get("equity_curve") or []
            print(f"  总收益   : {rep.get('total_return', 0):+.2%}")
            print(f"  年化收益 : {rep.get('annual_return', 0):+.2%}")
            print(f"  Sharpe   : {rep.get('sharpe', 0):.2f}")
            print(f"  最大回撤 : {rep.get('max_drawdown', 0):.2%}")
            print(f"  交易数   : {res.get('trades', 0)}")
            if eq:
                print(f"  净值区间 : {eq[0].get('equity', 1):.3f} → {eq[-1].get('equity', 1):.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 回测失败: {str(exc)[:120]}")

    print("\n" + "=" * 78)
    print("完成。注：缠论1买为确定性近似；真实缠论需笔/中枢/背驰精细实现。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
