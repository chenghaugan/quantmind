"""ML 因子模块测试。"""
import pytest
import pandas as pd
import numpy as np

from quantmind.research.ml_factor import (
    MLFactorConfig,
    MLFactorTrainer,
    train_ml_factor_example,
    _LGBM_AVAILABLE,
)


@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="LightGBM 未安装")
class TestMLFactorTrainer:
    """ML 因子训练器测试。"""

    def test_prepare_features_with_lag(self):
        """特征准备：滞后处理避免前视。"""
        # 构造测试数据
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        factor_panel = pd.DataFrame({
            "factor1": np.random.randn(100),
            "factor2": np.random.randn(100),
            "forward_return": np.random.randn(100),
        }, index=dates)

        trainer = MLFactorTrainer(MLFactorConfig(lag_periods=1))
        X, y = trainer.prepare_features(factor_panel, "forward_return")

        # 验证滞后处理：X 的长度应该比原始数据少 lag_periods
        assert len(X) == 99  # 100 - 1
        assert len(y) == 99
        assert "factor1" in X.columns
        assert "factor2" in X.columns

    def test_time_series_split(self):
        """时间序列分割：严格避免前视。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        X = pd.DataFrame({
            "factor1": np.random.randn(100),
            "factor2": np.random.randn(100),
        }, index=dates)
        y = pd.Series(np.random.randn(100), index=dates)

        config = MLFactorConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        trainer = MLFactorTrainer(config)
        splits = trainer.time_series_split(X, y)

        # 验证分割比例
        assert len(splits["train"][0]) == 60
        assert len(splits["val"][0]) == 20
        assert len(splits["test"][0]) == 20

        # 验证时间顺序：train < val < test
        train_end = splits["train"][0].index[-1]
        val_start = splits["val"][0].index[0]
        val_end = splits["val"][0].index[-1]
        test_start = splits["test"][0].index[0]

        assert train_end < val_start
        assert val_end < test_start

    def test_train_and_predict(self):
        """训练和预测：端到端流程。"""
        # 构造合成数据
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        
        # 因子与目标有弱相关性
        factor1 = np.random.randn(n_samples)
        factor2 = np.random.randn(n_samples)
        forward_return = 0.3 * factor1 + 0.2 * factor2 + np.random.randn(n_samples) * 0.5

        factor_panel = pd.DataFrame({
            "factor1": factor1,
            "factor2": factor2,
            "forward_return": forward_return,
        }, index=dates)

        trainer = MLFactorTrainer(MLFactorConfig(n_estimators=10))
        X, y = trainer.prepare_features(factor_panel, "forward_return")
        splits = trainer.time_series_split(X, y)

        # 训练
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        model = trainer.train(X_train, y_train, X_val, y_val)

        # 预测
        X_test, _ = splits["test"]
        y_pred = trainer.predict(X_test)

        assert len(y_pred) == len(X_test)
        assert y_pred.name == "ml_prediction"

    def test_evaluate_metrics(self):
        """评估指标：IC、IR、IC>0 比例。"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y_pred = pd.Series(np.random.randn(100), index=dates)
        y_true = pd.Series(np.random.randn(100), index=dates)

        trainer = MLFactorTrainer()
        metrics = trainer.evaluate(y_pred, y_true)

        assert "ic_mean" in metrics
        assert "ir" in metrics
        assert "ic_positive_ratio" in metrics
        assert -1 <= metrics["ic_mean"] <= 1
        assert 0 <= metrics["ic_positive_ratio"] <= 1

    def test_feature_importance(self):
        """特征重要性：返回排序后的特征重要性。"""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        factor_panel = pd.DataFrame({
            "factor1": np.random.randn(n_samples),
            "factor2": np.random.randn(n_samples),
            "forward_return": np.random.randn(n_samples),
        }, index=dates)

        trainer = MLFactorTrainer(MLFactorConfig(n_estimators=5))
        X, y = trainer.prepare_features(factor_panel, "forward_return")
        splits = trainer.time_series_split(X, y)
        X_train, y_train = splits["train"]
        trainer.train(X_train, y_train)

        importance = trainer.feature_importance()
        assert len(importance) == 2
        assert "factor1" in importance.index
        assert "factor2" in importance.index


@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="LightGBM 未安装")
class TestTrainMLFactorExample:
    """端到端 ML 因子训练示例测试。"""

    def test_end_to_end_training(self):
        """端到端训练：完整流程。"""
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        factor_panel = pd.DataFrame({
            "factor1": np.random.randn(n_samples),
            "factor2": np.random.randn(n_samples),
            "forward_return": np.random.randn(n_samples),
        }, index=dates)

        result = train_ml_factor_example(factor_panel, "forward_return")

        assert "model" in result
        assert "metrics" in result
        assert "feature_importance" in result
        assert result["n_train"] > 0
        assert result["n_test"] > 0
