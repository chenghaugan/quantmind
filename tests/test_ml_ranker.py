"""MLRanker 单元测试"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from quantmind.research.ml_ranker import MLRanker, MLRankerConfig


def test_ml_ranker_config():
    """测试 MLRankerConfig 默认值"""
    config = MLRankerConfig()
    assert config.train_ratio == 0.6
    assert config.val_ratio == 0.2
    assert config.test_ratio == 0.2
    assert config.lag_periods == 1
    assert config.n_estimators == 100


def test_ml_ranker_init():
    """测试 MLRanker 初始化"""
    ranker = MLRanker()
    assert ranker.model is None
    assert ranker.feature_names == []


def test_ml_ranker_prepare_features():
    """测试特征准备（带滞后）"""
    ranker = MLRanker(MLRankerConfig(lag_periods=1))

    # 构造测试数据
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(100),
        "factor2": np.random.randn(100),
        "forward_return": np.random.randn(100),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")

    # 验证滞后处理
    assert len(X) == 99  # 滞后1期后少一个样本
    assert len(y) == 99
    assert len(groups) > 0  # 按日期分组
    assert ranker.feature_names == ["factor1", "factor2"]


def test_ml_ranker_time_series_split():
    """测试时间序列分割"""
    ranker = MLRanker(MLRankerConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2))

    # 构造测试数据
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(100),
        "forward_return": np.random.randn(100),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # 验证分割比例（允许小误差）
    total = len(X)
    train_size = len(splits["train"][0])
    val_size = len(splits["val"][0])
    test_size = len(splits["test"][0])

    assert train_size + val_size + test_size == total
    assert abs(train_size / total - 0.6) < 0.1
    assert abs(val_size / total - 0.2) < 0.1
    assert abs(test_size / total - 0.2) < 0.1


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not installed"
)
def test_ml_ranker_train_predict():
    """测试模型训练和预测"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    ranker = MLRanker(MLRankerConfig(n_estimators=10))

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]

    # 训练
    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)

    assert ranker.model is not None
    assert len(ranker.feature_names) == 2

    # 预测
    X_test, _, _ = splits["test"]
    predictions = ranker.predict_rank(X_test)

    assert len(predictions) == len(X_test)
    assert isinstance(predictions, pd.Series)


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not installed"
)
def test_ml_ranker_evaluate_ndcg():
    """测试 NDCG 评估"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    ranker = MLRanker(MLRankerConfig(n_estimators=10, eval_at=[5, 10]))

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(200),
        "forward_return": np.random.randn(200),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]
    X_test, y_test, g_test = splits["test"]

    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)
    ndcg = ranker.evaluate_ndcg(X_test, y_test, g_test)

    assert "ndcg@5" in ndcg
    assert "ndcg@10" in ndcg
    assert 0 <= ndcg["ndcg@5"] <= 1
    assert 0 <= ndcg["ndcg@10"] <= 1


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not installed"
)
def test_ml_ranker_save_load():
    """测试模型保存和加载"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    ranker = MLRanker(MLRankerConfig(n_estimators=10))

    # 构造测试数据并训练
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    ranker.train(X_train, y_train, g_train)

    # 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save(str(model_path))

        assert model_path.exists()

        # 加载
        ranker2 = MLRanker()
        ranker2.load(str(model_path))

        assert ranker2.model is not None
        assert ranker2.feature_names == ranker.feature_names

        # 预测应该一致
        X_test, _, _ = splits["test"]
        pred1 = ranker.predict_rank(X_test)
        pred2 = ranker2.predict_rank(X_test)

        pd.testing.assert_series_equal(pred1, pred2)


def test_ml_ranker_feature_importance():
    """测试特征重要性"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    ranker = MLRanker(MLRankerConfig(n_estimators=10))

    # 构造测试数据并训练
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    })

    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    ranker.train(X_train, y_train, g_train)

    importance = ranker.feature_importance()

    assert isinstance(importance, pd.Series)
    assert len(importance) == 2
    assert "factor1" in importance.index
    assert "factor2" in importance.index
