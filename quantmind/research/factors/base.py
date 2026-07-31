"""因子基类与通用工具（参考 vnpy.alpha.dataset 的设计理念）。

每个因子把 ``List[BarData]`` 计算成与 K 线等长的 ``pd.Series``（按时间升序、索引为 bar 时间），
因子值缺失处以 ``NaN`` 填充。因子计算全部基于**历史已知数据**，避免前视。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from ...core.object import BarData


def bars_to_df(bars: List[BarData]) -> pd.DataFrame:
    """把 K 线列表转成 DataFrame（按时间升序），便于因子计算。"""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "open_interest"])
    recs = [
        {
            "datetime": b.datetime,
            "open": b.open_price,
            "high": b.high_price,
            "low": b.low_price,
            "close": b.close_price,
            "volume": b.volume,
            "open_interest": b.open_interest,
            "turnover": b.turnover,
        }
        for b in bars
    ]
    df = pd.DataFrame(recs).sort_values("datetime").reset_index(drop=True)
    return df


@dataclass
class FactorMeta:
    """因子元信息。"""

    name: str
    category: str = "technical"
    description: str = ""


class Factor(ABC):
    """因子抽象基类。"""

    meta: FactorMeta = FactorMeta(name="base")
    params: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def compute(self, bars: List[BarData]) -> pd.Series:
        """返回与 ``bars`` 等长的因子值序列（索引为 bar 序号 0..n-1）。"""
        raise NotImplementedError

    # ---- 便捷构造 ----
    @classmethod
    def make(cls, **params) -> "Factor":
        inst = cls()
        inst.params = params
        return inst


def rolling_zscore(series: pd.Series, window: int = 120) -> pd.Series:
    """滚动 z-score（expanding 兜底），用于多因子标准化。"""
    roll_mean = series.rolling(window, min_periods=20).mean()
    roll_std = series.rolling(window, min_periods=20).std()
    return (series - roll_mean) / (roll_std.replace(0, pd.NA))


def expanding_zscore(series: pd.Series) -> pd.Series:
    """扩张窗口 z-score（只用历史数据，无前视）。"""
    return (series - series.expanding().mean()) / (series.expanding().std().replace(0, pd.NA))
