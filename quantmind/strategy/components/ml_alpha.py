"""MLAlphaModel: 机器学习 Alpha 模型组件

将 ML 模型集成到策略组件框架中。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import AlphaModel
from ..context import StrategyContext

logger = logging.getLogger(__name__)


class MLAlphaModel(AlphaModel):
    """基于机器学习的 Alpha 模型"""

    def __init__(self, model_path: Optional[str] = None):
        """初始化 ML Alpha 模型

        Args:
            model_path: 预训练模型路径（joblib 格式）
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = []

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """加载预训练模型"""
        import joblib

        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return

        try:
            data = joblib.load(path)
            self.model = data.get("model")
            self.feature_names = data.get("feature_names", [])
            logger.info(f"ML model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def on_init(self, context: StrategyContext) -> None:
        """策略初始化时调用"""
        if self.model is None and self.model_path:
            self._load_model(self.model_path)

    def on_bar(self, bar) -> Optional[float]:
        """处理每根 K 线，返回目标仓位

        注意：实际选股逻辑需要在多标的场景下实现，这里只是示例接口。
        真实的 ML 选股应该在截面维度上进行排序和选择。

        Args:
            bar: 当前 K 线数据

        Returns:
            目标仓位（-1 到 1 之间），None 表示不交易
        """
        if self.model is None:
            return None

        # 这里需要根据实际的因子计算逻辑来构造特征
        # 示例：假设我们有动量因子
        # 实际使用时需要与因子计算模块集成
        return None

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """预测截面分数（用于选股排序）

        Args:
            features: 特征 DataFrame，index 为标的代码

        Returns:
            预测分数 Series
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        scores = self.model.predict(features)
        return pd.Series(scores, index=features.index)

    def select_top_k(
        self,
        features: pd.DataFrame,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[str]:
        """选择 Top-K 标的

        Args:
            features: 特征 DataFrame
            k: 选择数量
            threshold: 分数阈值

        Returns:
            选中的标的代码列表
        """
        scores = self.predict_scores(features)
        scores = scores[scores > threshold]
        top_k = scores.nlargest(k)
        return top_k.index.tolist()


__all__ = ["MLAlphaModel"]
