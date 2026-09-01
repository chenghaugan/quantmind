"""策略验证用确定性策略模板：动量 / 缠论1买 / 缠论3买 / 双均线。

供「策略验证」Web 入口（idea → 单品种回测 → 门槛判定 → 有效策略库）
与命令行验证脚本复用。所有策略均为确定性 CtaTemplate（可审计、可复现），
参数化后可直接回测；缠论3买复用 ``strategy/mined.py`` 的忠实实现。

- :class:`MomentumCtaStrategy` —— N 日动量阈值进出场（多空均可）。
- :class:`ChanFirstBuyStrategy` —— 缠论第一类买点（确定性近似：下跌趋势末端
  「价格新低但动能 ROC 不创新低」的底背驰检测）。
- ``chan_third_buy`` —— 复用 :class:`quantmind.strategy.mined.ChanThirdBuyStrategy`。
"""
from __future__ import annotations

from typing import Dict, Type

from ..core.object import BarData
from ..core.utility import ArrayManager
from .base import CtaTemplate
from .mined import ChanThirdBuyStrategy

__all__ = [
    "MomentumCtaStrategy",
    "ChanFirstBuyStrategy",
    "VALIDATION_STRATEGIES",
    "resolve_validate_strategy",
]


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
        mom = closes[-1] / closes[-self.window - 1] - 1.0  # 跨 window 个收益区间（原 [-window] 只有 window-1 个）
        if mom > self.threshold:
            target = self.max_pos * self.size
        elif mom < -self.threshold:
            target = -self.max_pos * self.size
        else:
            target = 0.0
        if target != self.last_target:
            oid = self.set_target(bar.vt_symbol, target)
            if oid == "":
                # 风控拒单：保留 last_target，下一根 bar 重试
                return
            self.last_target = target
            self.pos = target


class ChanFirstBuyStrategy(CtaTemplate):
    """缠论第一类买点（确定性近似）。

    核心：下跌趋势末端出现「价格新低但动能衰减」的底背驰。
    规则：
      - 下跌趋势：close 在 trend_window 均线之下。
      - 背驰检测：连续两次创新低（low_window 内的最低点），后一次价格更低，
        但后一次低点处的 ROC(roc_window) 高于前一次（动能未创新低）→ 买入。
      - 离场：价格上穿 trend_window 均线（趋势反转）或跌破背驰低点（新低延续）。

    注：为确定性近似；真实缠论需笔/线段/中枢的精细实现。
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
                # 拒单时不更新 entry_low/prev_low（预支更新会让本次背驰买点永久丢失）
                if self._set(bar.vt_symbol, self.max_pos * self.size):
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

    def _set(self, vt_symbol: str, target: float) -> bool:
        if target != self.last_target:
            oid = self.set_target(vt_symbol, target)
            if oid == "":
                # 风控拒单：保留 last_target，下一根 bar 重试
                return False
            self.last_target = target
            self.pos = target
        return True


class BollingerRecoverStrategy(CtaTemplate):
    """布林带回穿策略（按用户规则）：

    - 多头入场：收盘跌破下轨后，5 日内收盘重新上穿下轨 → 买入。
    - 多头离场：收盘跌破中轨（中轨卖出）或 亏损达 5%（止损）。
    - 空头入场：收盘突破上轨后，5 日内收盘跌回上轨下方 → 卖空。
    - 空头离场：收盘上穿中轨（中轨平仓）或 亏损达 5%（止损）。

    布林带：中轨=SMA(N)，上下轨=中轨±k×std（总体标准差，ddof=0）。
    """

    author = "QuantMind 验证（布林带回穿）"
    parameters = ["window", "k", "recover_days", "stop_loss", "size", "max_pos"]

    def __init__(self, context, setting=None):
        self.window = 20
        self.k = 2.0
        self.recover_days = 5      # 跌破/突破后回穿的有效天数
        self.stop_loss = 0.05      # 5% 止损
        self.size = 1
        self.max_pos = 1.0
        self.am = None
        self._below = None         # 距跌破下轨的天数（None=未跌破）
        self._above = None         # 距突破上轨的天数（None=未突破）
        self.entry_price = None
        self.last_target = 0.0
        super().__init__(context, setting)

    def on_bar(self, bar: BarData) -> None:
        if self.am is None:
            self.am = ArrayManager(self.window + 5)
        self.am.update_bar(bar)
        if not self.am.inited or len(self.am.close) < self.window:
            return
        closes = self.am.close
        close = closes[-1]
        n = self.window
        middle = self.am.sma(n)
        # 总体标准差（ddof=0，与布林带惯例一致）
        seg = closes[-n:]
        mean = sum(seg) / n
        var = sum((x - mean) ** 2 for x in seg) / n
        sd = var ** 0.5
        upper = middle + self.k * sd
        lower = middle - self.k * sd

        # ---- 回穿状态跟踪 ----
        # 多头入场：跌破下轨后 recover_days 日内收盘回穿上轨下轨
        signal_long = False
        if close < lower:
            self._below = 0
        elif self._below is not None:
            self._below += 1
            if self._below <= self.recover_days and close > lower:
                signal_long = True
                self._below = None
        # 空头入场：突破上轨后 recover_days 日内收盘跌回上轨下
        signal_short = False
        if close > upper:
            self._above = 0
        elif self._above is not None:
            self._above += 1
            if self._above <= self.recover_days and close < upper:
                signal_short = True
                self._above = None

        # ---- 持仓管理 ----
        target = self.last_target
        if target > 0:  # 持多
            if close < middle or (self.entry_price and close < self.entry_price * (1 - self.stop_loss)):
                target = 0.0
        elif target < 0:  # 持空
            if close > middle or (self.entry_price and close > self.entry_price * (1 + self.stop_loss)):
                target = 0.0

        fired = None  # 本根触发的入场信号（拒单时需回滚状态以便重试）
        if target == 0.0:
            if signal_long:
                target = self.max_pos * self.size
                fired = "long"
            elif signal_short:
                target = -self.max_pos * self.size
                fired = "short"

        if target != self.last_target:
            oid = self.set_target(bar.vt_symbol, target)
            if oid == "":
                # 风控拒单：保留 last_target，下一根 bar 重试；
                # 回穿信号回滚为“刚刚触发”状态，否则本次入场永久丢失
                if fired == "long":
                    self._below = 0
                elif fired == "short":
                    self._above = 0
                return
            self.last_target = target
            self.pos = target
            if fired == "long" or fired == "short":
                self.entry_price = close
            if target == 0.0:
                self.entry_price = None


#: 策略验证可用的确定性策略表：名称 → 类
VALIDATION_STRATEGIES: Dict[str, Type[CtaTemplate]] = {
    "momentum": MomentumCtaStrategy,
    "chan_first_buy": ChanFirstBuyStrategy,
    "chan_third_buy": ChanThirdBuyStrategy,
    "bollinger_recover": BollingerRecoverStrategy,
}

#: 策略默认参数（供页面/API 兜底）
DEFAULT_SETTINGS: Dict[str, dict] = {
    "momentum": {"window": 20, "threshold": 0.03, "size": 1, "max_pos": 1.0},
    "chan_first_buy": {"trend_window": 60, "low_window": 20,
                       "roc_window": 10, "size": 1, "max_pos": 1.0},
    "chan_third_buy": {"trend_window": 60, "break_window": 20,
                       "size": 1, "max_pos": 1.0},
    "bollinger_recover": {"window": 20, "k": 2.0, "recover_days": 5,
                           "stop_loss": 0.05, "size": 1, "max_pos": 1.0},
}

#: 预置模板的默认参数搜索网格（参数优化用；围绕 DEFAULT_SETTINGS 默认值展开）
DEFAULT_PARAM_GRIDS: Dict[str, dict] = {
    "momentum": {"window": [10, 20, 30], "threshold": [0.02, 0.03, 0.04]},
    "chan_first_buy": {"trend_window": [40, 60, 80], "low_window": [15, 20, 25],
                       "roc_window": [8, 10, 12]},
    "chan_third_buy": {"trend_window": [40, 60, 80], "break_window": [15, 20, 25]},
    "bollinger_recover": {"window": [15, 20, 30], "k": [1.5, 2.0, 2.5],
                          "stop_loss": [0.03, 0.05, 0.08]},
    "dual_ma": {"fast": [3, 5, 8], "slow": [15, 20, 30]},
}

#: idea → 策略名 关键词识别表（顺序敏感：先匹配更具体的）
_IDEA_KEYWORDS = [
    ("缠论1买", "chan_first_buy"),
    ("缠论一买", "chan_first_buy"),
    ("底背驰", "chan_first_buy"),
    ("一买", "chan_first_buy"),
    ("缠论3买", "chan_third_buy"),
    ("缠论三买", "chan_third_buy"),
    ("第三类买点", "chan_third_buy"),
    ("三买", "chan_third_buy"),
    ("布林", "bollinger_recover"),
    ("bollinger", "bollinger_recover"),
    ("布林带", "bollinger_recover"),
    ("上下轨", "bollinger_recover"),
    ("下轨", "bollinger_recover"),
    ("上轨", "bollinger_recover"),
    ("中轨", "bollinger_recover"),
    ("动量", "momentum"),
    ("momentum", "momentum"),
    ("趋势", "momentum"),
]


def resolve_validate_strategy(idea: str, fallback: str = "") -> str:
    """从 idea 关键词识别策略类型；识别不到返回 ``fallback``（或空串）。"""
    text = (idea or "").lower()
    for kw, name in _IDEA_KEYWORDS:
        if kw in text:
            return name
    return fallback if fallback in VALIDATION_STRATEGIES else ""
