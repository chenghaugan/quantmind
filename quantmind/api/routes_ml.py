"""ML 模型 API 路由（训练/预测/评估）。"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    """训练请求体。"""
    model_id: str
    model_type: str = "regressor"  # regressor | ranker
    factor_specs: List[Dict[str, Any]] = []  # FactorSpec 列表
    target_col: str = "forward_return"
    group_col: str = "date"


class PredictRequest(BaseModel):
    """预测请求体。"""
    model_id: str
    model_type: str = "regressor"
    features: Dict[str, List[float]]  # {feature_name: [values]}


@router.get("/models", summary="列出所有模型")
async def list_models() -> Dict[str, Any]:
    """列出所有已训练/加载的模型。"""
    from .app import get_ml_service
    models = get_ml_service().list_models()
    return {"models": models}


@router.post("/train", summary="训练模型", status_code=201)
async def train_model(payload: TrainRequest) -> Dict[str, Any]:
    """训练 LightGBM 回归或 LambdaRank 排序模型。

    注意：实际训练需要传入因子面板数据（通过 factor_specs 指定因子，
    后端从 DataManager 获取数据并计算因子值）。
    """
    from .app import get_ml_service
    try:
        # 这里简化处理：实际需要从 DataManager 获取数据并计算因子
        # 暂时返回提示信息
        return {
            "message": "训练接口已就绪，但需要传入因子面板数据。",
            "model_id": payload.model_id,
            "model_type": payload.model_type,
            "usage": "请通过 API 客户端构造 factor_panel 后调用 train_regressor/train_ranker",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", summary="模型预测")
async def predict(payload: PredictRequest) -> Dict[str, Any]:
    """使用训练好的模型进行预测。"""
    from .app import get_ml_service
    import pandas as pd

    try:
        features = pd.DataFrame(payload.features)
        predictions = get_ml_service().predict(
            payload.model_id,
            features,
            payload.model_type,
        )
        return {
            "model_id": payload.model_id,
            "predictions": predictions.tolist(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}", summary="删除模型")
async def delete_model(model_id: str) -> Dict[str, bool]:
    """删除指定模型（从内存和磁盘）。"""
    from .app import get_ml_service
    import os

    ml_service = get_ml_service()

    # 从内存移除
    if model_id in ml_service._trainers:
        del ml_service._trainers[model_id]
    if model_id in ml_service._rankers:
        del ml_service._rankers[model_id]

    # 从磁盘删除
    model_dir = ml_service.model_dir
    for suffix in [".joblib", "_ranker.joblib"]:
        path = model_dir / f"{model_id}{suffix}"
        if path.exists():
            os.remove(path)

    return {"deleted": True}


__all__ = ["router"]
