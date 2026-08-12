"""MLRanker: LambdaRank 排序模型（选股专用）

直接优化排序指标（NDCG），比回归更适合选股场景。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


@dataclass
class MLRankerConfig:
    """MLRanker 配置"""
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    lag_periods: int = 1
    n_estimators: int = 100
    learning_rate: float = 0.1
    eval_at: list[int] = None

    def __post_init__(self):
        if self.eval_at is None:
            self.eval_at = [5, 10, 20]


class MLRanker:
    """LambdaRank 排序模型"""

    def __init__(self, config: Optional[MLRankerConfig] = None):
        if not _HAS_LGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.config = config or MLRankerConfig()
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_names: list[str] = []

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "forward_return",
        group_col: str = "date",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """准备特征（带滞后处理）

        Args:
            df: 包含因子和目标的 DataFrame
            target_col: 目标列名
            group_col: 分组列名（用于 LambdaRank）

        Returns:
            X, y, groups
        """
        # 滞后处理：避免前视偏差
        lag = self.config.lag_periods
        feature_cols = [c for c in df.columns if c not in [target_col, group_col]]

        X = df[feature_cols].shift(lag).iloc[lag:]
        y = df[target_col].iloc[lag:]
        groups = df[group_col].iloc[lag:]

        self.feature_names = feature_cols
        return X, y, groups

    def time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """时间序列分割（按分组边界切分，避免跨组泄漏）"""
        unique_groups = groups.unique()
        n_groups = len(unique_groups)

        train_end = int(n_groups * self.config.train_ratio)
        val_end = int(n_groups * (self.config.train_ratio + self.config.val_ratio))

        train_groups = unique_groups[:train_end]
        val_groups = unique_groups[train_end:val_end]
        test_groups = unique_groups[val_end:]

        train_mask = groups.isin(train_groups)
        val_mask = groups.isin(val_groups)
        test_mask = groups.isin(test_groups)

        return {
            "train": (X[train_mask], y[train_mask], groups[train_mask]),
            "val": (X[val_mask], y[val_mask], groups[val_mask]),
            "test": (X[test_mask], y[test_mask], groups[test_mask]),
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        group_val: Optional[pd.Series] = None,
    ) -> None:
        """训练 LambdaRank 模型"""
        # 计算 group 大小
        train_group_sizes = group_train.value_counts().sort_index().values

        fit_params = {
            "group": train_group_sizes,
        }

        if X_val is not None and y_val is not None and group_val is not None:
            val_group_sizes = group_val.value_counts().sort_index().values
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_group"] = [val_group_sizes]

        self.model = lgb.LGBMRanker(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            verbosity=-1,
        )
        self.model.fit(X_train, y_train, **fit_params)

    def predict_rank(self, X: pd.DataFrame) -> pd.Series:
        """预测排序分数"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        scores = self.model.predict(X)
        return pd.Series(scores, index=X.index)

    def evaluate_ndcg(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        group_test: pd.Series,
    ) -> dict[str, float]:
        """评估 NDCG 指标"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        scores = self.model.predict(X_test)

        ndcg_scores = {}
        for k in self.config.eval_at:
            ndcg = self._compute_ndcg(y_test, scores, group_test, k)
            ndcg_scores[f"ndcg@{k}"] = ndcg

        return ndcg_scores

    def _compute_ndcg(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        groups: pd.Series,
        k: int,
    ) -> float:
        """计算 NDCG@k"""
        ndcg_list = []

        for group in groups.unique():
            mask = groups == group
            y_g = y_true[mask].values
            pred_g = y_pred[mask.values]

            if len(y_g) < 2:
                continue

            # 按预测分数排序
            order = np.argsort(-pred_g)
            y_g_sorted = y_g[order]

            # DCG
            dcg = np.sum((2 ** y_g_sorted[:k] - 1) / np.log2(np.arange(1, min(k, len(y_g_sorted)) + 1) + 1))

            # 理想排序
            ideal_order = np.argsort(-y_g)
            y_g_ideal = y_g[ideal_order]
            idcg = np.sum((2 ** y_g_ideal[:k] - 1) / np.log2(np.arange(1, min(k, len(y_g_ideal)) + 1) + 1))

            if idcg > 0:
                ndcg_list.append(dcg / idcg)

        return np.mean(ndcg_list) if ndcg_list else 0.0

    def feature_importance(self) -> pd.Series:
        """特征重要性"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names, name="importance")

    def save(self, path: str) -> None:
        """保存模型"""
        import joblib
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """加载模型"""
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info(f"Model loaded from {path}")
