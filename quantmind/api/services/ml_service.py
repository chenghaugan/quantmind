"""ML 模型服务：训练、预测、评估。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...research.ml_factor import MLFactorTrainer, MLFactorConfig
from ...research.ml_ranker import MLRanker, MLRankerConfig

_logger = logging.getLogger("quantmind.api.ml_service")


class MLService:
    """ML 模型训练/预测/评估服务。"""

    def __init__(self) -> None:
        self.model_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.model_dir.mkdir(exist_ok=True)
        self._trainers: Dict[str, MLFactorTrainer] = {}
        self._rankers: Dict[str, MLRanker] = {}

    def train_regressor(
        self,
        model_id: str,
        factor_panel: pd.DataFrame,
        target_col: str = "forward_return",
        config: Optional[MLFactorConfig] = None,
    ) -> Dict[str, Any]:
        """训练 LightGBM 回归模型。"""
        trainer = MLFactorTrainer(config)
        X, y = trainer.prepare_features(factor_panel, target_col)
        splits = trainer.time_series_split(X, y)

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        model = trainer.train(X_train, y_train, X_val, y_val)
        y_pred = trainer.predict(X_test)
        metrics = trainer.evaluate(y_pred, y_test)
        importance = trainer.feature_importance()

        # 保存模型
        import joblib
        model_path = self.model_dir / f"{model_id}.joblib"
        joblib.dump(
            {"model": model, "feature_names": trainer.feature_names},
            model_path,
        )
        self._trainers[model_id] = trainer

        _logger.info("回归模型 %s 训练完成，IC=%.4f, IR=%.4f", model_id, metrics["ic_mean"], metrics["ir"])

        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_importance": importance.head(10).to_dict(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def train_ranker(
        self,
        model_id: str,
        factor_panel: pd.DataFrame,
        target_col: str = "forward_return",
        group_col: str = "date",
        config: Optional[MLRankerConfig] = None,
    ) -> Dict[str, Any]:
        """训练 LambdaRank 排序模型。"""
        ranker = MLRanker(config)
        X, y, groups = ranker.prepare_features(factor_panel, target_col, group_col)
        splits = ranker.time_series_split(X, y, groups)

        X_train, y_train, g_train = splits["train"]
        X_val, y_val, g_val = splits["val"]
        X_test, y_test, g_test = splits["test"]

        model = ranker.train(X_train, y_train, g_train, X_val, y_val, g_val)
        ndcg = ranker.evaluate_ndcg(X_test, y_test, g_test)
        importance = ranker.feature_importance()

        # 保存模型
        ranker.save(str(self.model_dir / f"{model_id}_ranker.joblib"))
        self._rankers[model_id] = ranker

        _logger.info("排序模型 %s 训练完成，NDCG@10=%.4f", model_id, ndcg.get("ndcg@10", 0))

        return {
            "model_id": model_id,
            "model_path": str(self.model_dir / f"{model_id}_ranker.joblib"),
            "ndcg": ndcg,
            "feature_importance": importance.head(10).to_dict(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        model_type: str = "regressor",
    ) -> pd.Series:
        """使用训练好的模型预测。"""
        if model_type == "ranker":
            if model_id not in self._rankers:
                # 尝试加载
                path = self.model_dir / f"{model_id}_ranker.joblib"
                if path.exists():
                    ranker = MLRanker()
                    ranker.load(str(path))
                    self._rankers[model_id] = ranker
                else:
                    raise ValueError(f"模型不存在: {model_id}")
            return self._rankers[model_id].predict_rank(features)
        else:
            if model_id not in self._trainers:
                # 尝试加载
                import joblib
                path = self.model_dir / f"{model_id}.joblib"
                if path.exists():
                    data = joblib.load(path)
                    trainer = MLFactorTrainer()
                    trainer.model = data["model"]
                    trainer.feature_names = data["feature_names"]
                    self._trainers[model_id] = trainer
                else:
                    raise ValueError(f"模型不存在: {model_id}")
            return self._trainers[model_id].predict(features)

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有已训练/加载的模型。"""
        models = []
        for mid in self._trainers:
            models.append({"model_id": mid, "type": "regressor"})
        for mid in self._rankers:
            models.append({"model_id": mid, "type": "ranker"})
        # 扫描模型目录
        for path in self.model_dir.glob("*.joblib"):
            mid = path.stem.replace("_ranker", "")
            mtype = "ranker" if path.stem.endswith("_ranker") else "regressor"
            if not any(m["model_id"] == mid for m in models):
                models.append({"model_id": mid, "type": mtype, "path": str(path)})
        return models


__all__ = ["MLService"]
