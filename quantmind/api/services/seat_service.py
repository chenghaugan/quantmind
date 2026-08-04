"""席位因子服务：加载席位 CSV → 计算 F1-F8 → 评估 IC/分组收益。"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.constant import Exchange, Interval
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from ...research.factors.seat_futures import (
    compute_seat_factors,
    seat_df_from_tradingagents,
    _SEAT_FACTORS,
)
from ..schemas import SeatFactorRequest

_logger = logging.getLogger(__name__)


class SeatService:
    """封装席位因子计算与评估逻辑。"""

    def __init__(self, dm: DataManager) -> None:
        self.dm = dm

    @staticmethod
    def list_factors() -> Dict[str, str]:
        """返回 F1-F8 因子名与简介。"""
        return {
            "F1_net_position": "净持仓（各席位净持仓矩阵）",
            "F2_net_change": "净持仓变化（一阶差分）",
            "F3_net_ratio": "净持仓占比（除以总持仓）",
            "F4_concentration": "多空持仓比代理（资金集中度）",
            "F5_net_change_rate": "净持仓变化率（百分比变化）",
            "F6_net_accel": "净持仓二阶变化（加速度）",
            "F7_net_zscore": "净持仓 Z-score（滚动标准化）",
            "F8_seat_sentiment": "席位情绪综合（rank 均值）",
        }

    async def compute(self, req: SeatFactorRequest) -> Dict[str, Any]:
        """加载席位数据 → 计算因子 → 评估 IC/分组收益。"""
        if req.factor not in _SEAT_FACTORS:
            raise ValueError(f"未知席位因子: {req.factor}（可选: {_SEAT_FACTORS}）")

        try:
            # 从 CSV 加载席位净持仓矩阵
            seat_df, total_oi = seat_df_from_tradingagents(req.seat_data_root, req.symbol)
            n_seats = seat_df.shape[1]
            n_dates = seat_df.shape[0]

            # 计算因子序列（聚合为单列）
            factors_dict = compute_seat_factors(seat_df, total_oi, aggregate=req.aggregate)
            factor_series = factors_dict[req.factor]

            # 获取价格数据并计算未来收益
            price_df = await self._get_price_data(req.symbol, req.exchange, req.interval)
            if price_df is None or len(price_df) == 0:
                return {"factor": req.factor, "error": "无法获取价格数据"}

            # 计算未来收益
            returns = price_df["close"].pct_change().shift(-req.forward_periods)

            # 对齐因子序列与收益序列（按日期索引）
            aligned = pd.DataFrame({
                "factor": factor_series,
                "return": returns,
            }).dropna()

            if len(aligned) < 10:
                return {"factor": req.factor, "error": f"有效样本不足（仅 {len(aligned)} 个）"}

            # 时序 IC：因子值与未来收益的 Pearson 相关系数
            ic_series = aligned["factor"].rolling(60, min_periods=20).corr(aligned["return"])
            ic_mean = float(ic_series.mean()) if not ic_series.isna().all() else 0.0
            ic_std = float(ic_series.std()) if not ic_series.isna().all() else 0.0
            ir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_positive_ratio = float((ic_series > 0).mean()) if not ic_series.isna().all() else 0.0

            # 分组收益：按因子值分位数分组，计算各组平均收益
            aligned["quantile"] = pd.qcut(
                aligned["factor"],
                q=req.n_groups,
                labels=False,
                duplicates="drop",
            )
            group_returns = aligned.groupby("quantile")["return"].mean()
            top_quantile_return = float(group_returns.iloc[-1]) if len(group_returns) > 0 else 0.0
            bottom_quantile_return = float(group_returns.iloc[0]) if len(group_returns) > 0 else 0.0
            long_short_return = top_quantile_return - bottom_quantile_return

            # 综合得分
            composite_score = ic_mean * (ir if ir > 0 else 0) * ic_positive_ratio

            return {
                "factor": req.factor,
                "n_seats": n_seats,
                "n_dates": n_dates,
                "n_samples": len(aligned),
                "ic_mean": ic_mean,
                "ir": ir,
                "ic_positive_ratio": ic_positive_ratio,
                "top_quantile_return": top_quantile_return,
                "long_short_return": long_short_return,
                "composite_score": composite_score,
            }
        except FileNotFoundError as e:
            _logger.warning("席位数据缺失: %s", e)
            return {"factor": req.factor, "error": f"席位数据文件缺失: {e}"}
        except Exception as e:
            _logger.exception("席位因子计算失败")
            return {"factor": req.factor, "error": f"计算失败: {e}"}

    async def _get_price_data(
        self, symbol: str, exchange: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """从 DataManager 获取价格数据并转为 DataFrame。"""
        try:
            req = HistoryRequest(
                symbol=symbol,
                exchange=Exchange(exchange.upper()),
                interval=Interval(interval or "1d"),
            )
            bars = await self.dm.get_bar_data(req)
            if not bars:
                return None

            df = pd.DataFrame([{
                "datetime": b.datetime,
                "close": b.close_price,
            } for b in bars])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            return df
        except Exception as e:
            _logger.warning("获取价格数据失败: %s", e)
            return None
