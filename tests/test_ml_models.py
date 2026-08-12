"""ML 选股模型层测试。"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantmind.research.ml_ranker import MLRanker, MLRankerConfig
from quantmind.strategy.components.ml_alpha import MLAlphaModel
from quantmind.core.object import BarData
from quantmind.core.constant import Exchange, Interval


def test_ml_ranker_config():
    """MLRankerConfig 默认值正确。"""
    config = MLRankerConfig()
    assert config.train_ratio == 0.6
    assert config.val_ratio == 0.2
    assert config.test_ratio == 0.2
    assert config.n_estimators == 200
    assert config.lag_periods == 1


def test_ml_ranker_init():
    """MLRanker 初始化。"""
    ranker = MLRanker()
    assert ranker.model is None
    assert ranker.feature_names == []


def test_ml_ranker_prepare_features():
    """MLRanker.prepare_features 特征准备。"""
    # 构造测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(100),
        "factor2": np.random.randn(100),
        "forward_return": np.random.randn(100),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker()
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")

    assert len(X) > 0
    assert len(y) > 0
    assert len(groups) > 0
    assert ranker.feature_names == ["factor1", "factor2"]


def test_ml_ranker_time_series_split():
    """MLRanker.time_series_split 时间序列分割。"""
    # 构造测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(100),
        "factor2": np.random.randn(100),
        "forward_return": np.random.randn(100),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker()
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) == 3  # (X, y, groups)
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3


def test_ml_ranker_train():
    """MLRanker.train 训练模型。"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker(MLRankerConfig(n_estimators=10))  # 减少迭代次数加速测试
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]

    model = ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)
    assert model is not None
    assert ranker.model is not None
    assert len(ranker.feature_names) == 2


def test_ml_ranker_predict():
    """MLRanker.predict_rank 预测。"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker(MLRankerConfig(n_estimators=10))
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]

    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)

    X_test, y_test, g_test = splits["test"]
    predictions = ranker.predict_rank(X_test)
    assert len(predictions) == len(X_test)


def test_ml_ranker_evaluate_ndcg():
    """MLRanker.evaluate_ndcg 评估 NDCG。"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker(MLRankerConfig(n_estimators=10))
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]
    X_test, y_test, g_test = splits["test"]

    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)
    ndcg = ranker.evaluate_ndcg(X_test, y_test, g_test)

    assert "ndcg@5" in ndcg
    assert "ndcg@10" in ndcg
    assert "ndcg@20" in ndcg


def test_ml_ranker_feature_importance():
    """MLRanker.feature_importance 特征重要性。"""
    try:
        import lightgbm
    except ImportError:
        pytest.skip("LightGBM not installed")

    # 构造测试数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker(MLRankerConfig(n_estimators=10))
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]

    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)
    importance = ranker.feature_importance()

    assert len(importance) == 2
    assert "factor1" in importance.index
    assert "factor2" in importance.index


def test_ml_alpha_model_init():
    """MLAlphaModel 初始化。"""
    model = MLAlphaModel(top_k=10, score_threshold=0.5)
    assert model.top_k == 10
    assert model.score_threshold == 0.5
    assert model._model is None
    assert model._init_done is False


def test_ml_alpha_model_on_init():
    """MLAlphaModel.on_init 初始化。"""
    model = MLAlphaModel()
    model.on_init()
    assert model._init_done is True


def test_ml_alpha_model_on_bar():
    """MLAlphaModel.on_bar 处理 bar。"""
    model = MLAlphaModel()
    model.on_init()

    # 构造测试 bar
    bar = BarData(
        symbol="600000",
        exchange=Exchange.SSE,
        datetime=datetime(2024, 1, 1),
        interval=Interval.DAILY,
        open_price=10.0,
        high_price=10.5,
        low_price=9.5,
        close_price=10.2,
        volume=1000000,
    )

    # 没有模型时返回 None
    score = model.on_bar(bar)
    assert score is None


def test_ml_alpha_model_get_predictions():
    """MLAlphaModel.get_predictions 获取预测。"""
    model = MLAlphaModel()
    model.on_init()

    predictions = model.get_predictions()
    assert isinstance(predictions, dict)
    assert len(predictions) == 0


def test_ml_alpha_model_get_top_k():
    """MLAlphaModel.get_top_k 获取 TopK。"""
    model = MLAlphaModel(top_k=5)
    model.on_init()

    # 手动设置一些预测
    model._predictions = {
        "600000.SSE": 0.8,
        "600001.SSE": 0.6,
        "600002.SSE": 0.9,
        "600003.SSE": 0.7,
        "600004.SSE": 0.5,
        "600005.SSE": 0.4,
    }

    top_k = model.get_top_k()
    assert len(top_k) == 5
    assert top_k[0] == "600002.SSE"  # 最高分


def test_ml_alpha_model_save_load(tmp_path):
    """MLAlphaModel 模型保存加载。"""
    try:
        import lightgbm
        import joblib
    except ImportError:
        pytest.skip("LightGBM or joblib not installed")

    # 训练一个简单模型
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    data = {
        "date": dates,
        "factor1": np.random.randn(200),
        "factor2": np.random.randn(200),
        "forward_return": np.random.randn(200),
    }
    df = pd.DataFrame(data)

    ranker = MLRanker(MLRankerConfig(n_estimators=10))
    X, y, groups = ranker.prepare_features(df, target_col="forward_return", group_col="date")
    splits = ranker.time_series_split(X, y, groups)

    X_train, y_train, g_train = splits["train"]
    X_val, y_val, g_val = splits["val"]

    ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)

    # 保存模型
    model_path = tmp_path / "test_model.joblib"
    ranker.save(str(model_path))
    assert model_path.exists()

    # 加载模型
    ranker2 = MLRanker()
    ranker2.load(str(model_path))
    assert ranker2.model is not None
    assert ranker2.feature_names == ranker.feature_names


def test_ml_service_init():
    """MLService 初始化。"""
    from quantmind.api.services.ml_service import MLService

    service = MLService()
    assert service.model_dir.exists()
    assert len(service._trainers) == 0
    assert len(service._rankers) == 0


def test_ml_service_list_models():
    """MLService.list_models 列出模型。"""
    from quantmind.api.services.ml_service import MLService

    service = MLService()
    models = service.list_models()
    assert isinstance(models, list)


def test_ml_service_predict_not_found():
    """MLService.predict 模型不存在。"""
    from quantmind.api.services.ml_service import MLService

    service = MLService()
    features = pd.DataFrame({"factor1": [1.0, 2.0], "factor2": [3.0, 4.0]})

    with pytest.raises(ValueError, match="模型不存在"):
        service.predict("nonexistent_model", features)
