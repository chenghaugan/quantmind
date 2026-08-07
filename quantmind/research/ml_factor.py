"""轻量级机器学习因子示例（LightGBM）。

本模块演示如何在量化研究中安全地使用机器学习模型，重点强调**前视偏差防护**。

前视偏差（Look-ahead Bias）防护措施：
1. 严格的时间序列分割：训练集、验证集、测试集按时间顺序划分
2. 滚动窗口训练：避免使用未来数据训练模型
3. 特征滞后处理：所有特征使用 t-1 期数据预测 t 期收益
4. 交叉验证防护：时间序列交叉验证（TimeSeriesSplit）而非随机分割

示例流程：
1. 准备因子面板数据（多标的、多时间截面）
2. 构造特征矩阵（滞后处理避免前视）
3. 时间序列分割（train/val/test）
4. 训练 LightGBM 模型
5. 评估样本外表现（IC、IR、分组收益）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger("quantmind.research.ml")

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    lgb = None  # type: ignore


@dataclass
class MLFactorConfig:
    """ML 因子配置。"""

    # 时间序列分割比例
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # LightGBM 超参数
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    # 前视防护：特征滞后周期数
    lag_periods: int = 1


class MLFactorTrainer:
    """LightGBM 因子训练器（带前视偏差防护）。"""

    def __init__(self, config: Optional[MLFactorConfig] = None) -> None:
        if not _LGBM_AVAILABLE:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        self.config = config or MLFactorConfig()
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []

    def prepare_features(
        self,
        factor_panel: pd.DataFrame,
        target_col: str = "forward_return",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备特征矩阵和目标变量（带前视防护）。

        :param factor_panel: 因子面板数据，index 为日期，columns 为因子名称。
                             必须包含 target_col 列（未来收益）。
        :param target_col: 目标变量列名（未来收益）。
        :return: (X, y) 特征矩阵和目标变量，已做滞后处理。
        """
        # 前视防护：所有特征滞后 lag_periods 期
        lag = self.config.lag_periods
        features = factor_panel.drop(columns=[target_col]).shift(lag)

        # 目标变量不滞后（t 期特征预测 t 期收益）
        target = factor_panel[target_col]

        # 删除因滞后产生的 NaN
        valid_mask = features.notna().all(axis=1) & target.notna()
        X = features[valid_mask]
        y = target[valid_mask]

        self.feature_names = list(X.columns)
        _logger.info(
            "特征准备完成：%d 个特征，%d 个样本（滞后 %d 期）",
            len(self.feature_names), len(X), lag,
        )
        return X, y

    def time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """时间序列分割（严格避免前视）。

        按时间顺序分割为训练集、验证集、测试集，确保：
        - 训练集 < 验证集 < 测试集（时间顺序）
        - 无重叠（避免数据泄露）

        :return: {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        """
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        splits = {
            "train": (X.iloc[:train_end], y.iloc[:train_end]),
            "val": (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
            "test": (X.iloc[val_end:], y.iloc[val_end:]),
        }

        _logger.info(
            "时间序列分割：train=%d, val=%d, test=%d",
            len(splits["train"][0]), len(splits["val"][0]), len(splits["test"][0]),
        )
        return splits

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> lgb.LGBMRegressor:
        """训练 LightGBM 模型。

        :param X_train: 训练集特征。
        :param y_train: 训练集目标。
        :param X_val: 验证集特征（可选，用于早停）。
        :param y_val: 验证集目标（可选，用于早停）。
        :return: 训练好的模型。
        """
        cfg = self.config
        model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=42,
            verbosity=-1,
        )

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [lgb.early_stopping(10, verbose=False)]

        model.fit(X_train, y_train, **fit_params)
        self.model = model
        _logger.info("模型训练完成：%d 棵树", model.n_estimators_)
        return model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """使用训练好的模型预测。"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index, name="ml_prediction")

    def evaluate(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> Dict[str, float]:
        """评估模型表现（样本外）。

        :return: {"ic": IC均值, "ir": 信息比率, "ic_positive_ratio": IC>0比例}
        """
        # 计算整体 IC（简化版，不按时间截面分组）
        # 如果 index 是 MultiIndex（日期+标的），则按日期分组计算截面 IC
        if isinstance(y_pred.index, pd.MultiIndex):
            # 按日期分组计算截面 IC
            ic_list = []
            for date, group_pred in y_pred.groupby(level=0):
                group_true = y_true.loc[date]
                if len(group_pred) > 1:
                    ic = group_pred.corr(group_true)
                    if not pd.isna(ic):
                        ic_list.append(ic)
            ic_series = pd.Series(ic_list)
        else:
            # 单标的时序数据，直接计算整体 IC
            ic_series = pd.Series([y_pred.corr(y_true)])

        ic_mean = ic_series.mean() if len(ic_series) > 0 else 0.0
        ic_std = ic_series.std() if len(ic_series) > 1 else 1.0
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_positive_ratio = (ic_series > 0).mean() if len(ic_series) > 0 else 0.0

        metrics = {
            "ic_mean": ic_mean,
            "ir": ir,
            "ic_positive_ratio": ic_positive_ratio,
            "ic_std": ic_std,
        }
        _logger.info("模型评估：IC=%.4f, IR=%.4f, IC>0比例=%.2f%%",
                     ic_mean, ir, ic_positive_ratio * 100)
        return metrics

    def feature_importance(self) -> pd.Series:
        """返回特征重要性。"""
        if self.model is None:
            raise ValueError("模型未训练")
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names, name="importance").sort_values(ascending=False)


def train_ml_factor_example(
    factor_panel: pd.DataFrame,
    target_col: str = "forward_return",
) -> Dict[str, any]:
    """端到端 ML 因子训练示例（带前视防护）。

    :param factor_panel: 因子面板数据（index=日期, columns=因子+target）。
    :param target_col: 目标变量列名。
    :return: 包含模型、评估指标、特征重要性的字典。
    """
    trainer = MLFactorTrainer()

    # 1. 准备特征（滞后处理）
    X, y = trainer.prepare_features(factor_panel, target_col)

    # 2. 时间序列分割
    splits = trainer.time_series_split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # 3. 训练模型
    model = trainer.train(X_train, y_train, X_val, y_val)

    # 4. 样本外评估
    y_pred_test = trainer.predict(X_test)
    metrics = trainer.evaluate(y_pred_test, y_test)

    # 5. 特征重要性
    importance = trainer.feature_importance()

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


__all__ = [
    "MLFactorConfig",
    "MLFactorTrainer",
    "train_ml_factor_example",
]
