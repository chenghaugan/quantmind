"""MLAlphaModel 单元测试"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from quantmind.strategy.components.ml_alpha import MLAlphaModel


def test_ml_alpha_model_init():
    """测试 MLAlphaModel 初始化"""
    model = MLAlphaModel()
    assert model.model is None
    assert model.feature_names == []


def test_ml_alpha_model_init_with_path():
    """测试带模型路径的初始化"""
    model = MLAlphaModel(model_path="/nonexistent/path.joblib")
    # 文件不存在，模型应该为 None
    assert model.model is None


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not installed"
)
def test_ml_alpha_model_predict_scores():
    """测试预测分数"""
    try:
        import lightgbm
        import joblib
    except ImportError:
        pytest.skip("LightGBM or joblib not installed")

    # 创建一个简单的模型并保存
    from quantmind.research.ml_ranker import MLRanker, MLRankerConfig

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
    ranker.train(X_train, y_train, g_train)

    # 保存模型
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save(str(model_path))

        # 加载模型
        alpha_model = MLAlphaModel(model_path=str(model_path))
        assert alpha_model.model is not None

        # 预测分数
        X_test, _, _ = splits["test"]
        scores = alpha_model.predict_scores(X_test)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(X_test)


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not installed"
)
def test_ml_alpha_model_select_top_k():
    """测试 Top-K 选股"""
    try:
        import lightgbm
        import joblib
    except ImportError:
        pytest.skip("LightGBM or joblib not installed")

    from quantmind.research.ml_ranker import MLRanker, MLRankerConfig

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
    ranker.train(X_train, y_train, g_train)

    # 保存模型
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save(str(model_path))

        # 加载模型
        alpha_model = MLAlphaModel(model_path=str(model_path))

        # 构造测试特征（模拟多只股票）
        test_features = pd.DataFrame({
            "factor1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "factor2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }, index=["stock1", "stock2", "stock3", "stock4", "stock5"])

        # 选择 Top-3
        selected = alpha_model.select_top_k(test_features, k=3)

        assert len(selected) == 3
        assert all(s in test_features.index for s in selected)


def test_ml_alpha_model_select_top_k_no_model():
    """测试未加载模型时的 Top-K 选股"""
    alpha_model = MLAlphaModel()

    test_features = pd.DataFrame({
        "factor1": [1.0, 2.0, 3.0],
    }, index=["stock1", "stock2", "stock3"])

    with pytest.raises(ValueError, match="Model not loaded"):
        alpha_model.select_top_k(test_features, k=2)


def test_ml_alpha_model_predict_scores_no_model():
    """测试未加载模型时的预测"""
    alpha_model = MLAlphaModel()

    test_features = pd.DataFrame({
        "factor1": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError, match="Model not loaded"):
        alpha_model.predict_scores(test_features)
