"""时点切分与市场 regime 标注（对标 AlphaBench 的 train/val/test 防泄漏与牛/熊/震荡分层）。

论文反复强调两个工程要点：
  1. **防泄漏切分**：搜索/训练只应使用 train 期数据做决策，val/test 期仅作最终评估
     （对应 ``cot_search`` 的 ``val_panel`` 独立验证）。
  2. **市场 regime 分层**：同一因子在牛/熊/震荡市的预测力可能截然不同，评测应分
     regime 报告，而非只看全样本均值。

本模块提供：
  - :func:`time_split`：按比例或日期把面板切成 train / val / test 三段（标准时点切分）。
  - :func:`regime_labels`：依据基准指数的滚动收益趋势给每个交易日打标签
    （bull / bear / sideways）。
  - :class:`PanelSplitter`：把切分 + regime 组合成便捷的一次性工具。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .factors.alpha_cs import Panel

_logger = logging.getLogger("quantmind.research.split")

# regime 名称（论文用语：Neutral / Bear / Bull，此处统一中文语义 key）
REGIME_TYPES = ("bull", "bear", "sideways")

__all__ = [
    "time_split",
    "regime_labels",
    "PanelSplitter",
    "REGIME_TYPES",
]


def _slice_panel(panel: Panel, idx: Sequence) -> Panel:
    """按日期索引切片面板（各字段对齐）。"""
    return Panel(
        close=panel.close.loc[idx] if len(panel.close) else panel.close,
        open=panel.open.loc[idx] if len(panel.open) else panel.open,
        high=panel.high.loc[idx] if len(panel.high) else panel.high,
        low=panel.low.loc[idx] if len(panel.low) else panel.low,
        volume=panel.volume.loc[idx] if len(panel.volume) else panel.volume,
        amount=panel.amount.loc[idx] if len(panel.amount) else panel.amount,
    )


def time_split(
    panel: Panel,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    by_date: Optional[Tuple[str, str]] = None,
) -> Tuple[Panel, Panel, Panel]:
    """按时间把面板切成 train / val / test 三段（标准时点切分，避免随机打散引入前视）。

    Args:
        panel: 待切分面板（index 为日期，需升序）。
        train_frac: train 占比（0-1）。
        val_frac: val 占比（0-1）；余下为 test。
        by_date: 可选 (start, end) 字符串覆盖，若提供则按该日期区间裁剪后再比例切分。

    Returns:
        (train_panel, val_panel, test_panel)，三者按时间顺序拼接。
    """
    if panel.close.empty:
        return panel, _empty_like(panel), _empty_like(panel)

    p = panel
    if by_date:
        start, end = by_date
        mask = (p.close.index >= pd.Timestamp(start)) & (p.close.index <= pd.Timestamp(end))
        p = _slice_panel(p, p.close.index[mask])

    dates = list(p.close.index)
    n = len(dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_dates = dates[:n_train]
    val_dates = dates[n_train:n_train + n_val]
    test_dates = dates[n_train + n_val:]
    return (_slice_panel(p, train_dates), _slice_panel(p, val_dates),
            _slice_panel(p, test_dates))


def _empty_like(panel: Panel) -> Panel:
    empty = panel.close.iloc[0:0]
    return Panel(close=empty, open=empty.copy(), high=empty.copy(),
                 low=empty.copy(), volume=empty.copy(), amount=empty.copy())


def regime_labels(
    close: pd.Series,
    window: int = 20,
    slope_threshold: float = 0.0,
    flat_window: int = 5,
) -> pd.Series:
    """根据基准价格序列给每个交易日打 regime 标签。

    用滚动对数收益斜率与波动判断趋势方向：
      - 斜率 > threshold 且显著 → bull（上涨）
      - 斜率 < -threshold 且显著 → bear（下跌）
      - 否则 → sideways（震荡）

    Args:
        close: 基准指数/面板均值的收盘价序列（index=日期）。
        window: 趋势判断的滚动窗口。
        slope_threshold: 归一化斜率阈值（默认 0；可设 >0 提高归入 bull/bear 的难度）。
        flat_window: 近 flat_window 日涨跌幅度小则视为 sideways。

    Returns:
        ``pd.Series``（index=日期），值为 bull/bear/sideways。
    """
    log_p = np.log(close.replace(0, np.nan))
    # 滚动窗口的线性斜率（单位：每期对数收益）
    slope = log_p.rolling(window, min_periods=max(3, window // 2)).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan,
        raw=True,
    )
    # 近 flat_window 期的累计对数收益（用于识别震荡）
    recent_chg = log_p.diff(flat_window).abs()

    out = pd.Series("sideways", index=close.index, dtype=object)
    for idx, s in slope.items():
        if pd.isna(s):
            out[idx] = "sideways"
            continue
        if recent_chg.get(idx, 0) < 1e-4:
            out[idx] = "sideways"
        elif s > slope_threshold:
            out[idx] = "bull"
        elif s < -slope_threshold:
            out[idx] = "bear"
        else:
            out[idx] = "sideways"
    return out


@dataclass
class SplitResult:
    """切分 + regime 的完整结果。"""

    train: Panel
    val: Panel
    test: Panel
    regime_train: pd.Series = field(default_factory=pd.Series)
    regime_val: pd.Series = field(default_factory=pd.Series)
    regime_test: pd.Series = field(default_factory=pd.Series)
    splits: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_train": len(self.train.dates),
            "n_val": len(self.val.dates),
            "n_test": len(self.test.dates),
            "regime_train": dict(self.regime_train.value_counts()),
            "regime_val": dict(self.regime_val.value_counts()),
            "regime_test": dict(self.regime_test.value_counts()),
        }


class PanelSplitter:
    """组合时点切分 + regime 标注的一次性工具。

    用法::

        splitter = PanelSplitter(train_frac=0.6, val_frac=0.2, regime_window=20)
        res = splitter.split(panel, benchmark_series=panel.close.mean(axis=1))
        # res.train / res.val / res.test 为三段面板；
        # res.regime_* 为每段的 regime 标签。
    """

    def __init__(
        self,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        regime_window: int = 20,
        regime_threshold: float = 0.0,
    ) -> None:
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold

    def split(
        self,
        panel: Panel,
        benchmark_series: Optional[pd.Series] = None,
    ) -> SplitResult:
        """切分面板并给每段标注 regime。

        Args:
            panel: 待切分面板。
            benchmark_series: 用于判断 regime 的基准序列；默认取面板各标的均值收盘。

        Returns:
            :class:`SplitResult`。
        """
        train, val, test = time_split(panel, self.train_frac, self.val_frac)
        bench = benchmark_series if benchmark_series is not None else panel.close.mean(axis=1)
        reg_all = regime_labels(bench, window=self.regime_window,
                                slope_threshold=self.regime_threshold)
        return SplitResult(
            train=train, val=val, test=test,
            regime_train=reg_all.reindex(train.dates),
            regime_val=reg_all.reindex(val.dates),
            regime_test=reg_all.reindex(test.dates),
            splits=[len(train.dates), len(val.dates), len(test.dates)],
        )
