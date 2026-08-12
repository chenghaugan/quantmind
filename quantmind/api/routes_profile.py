"""投资者偏好画像 API 路由。"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", summary="列出所有 Profile")
async def list_profiles() -> Dict[str, List[Dict[str, Any]]]:
    """列出所有投资者偏好画像（预定义 + 自定义）。"""
    from .app import app
    service = app.state.profile_service
    return {"profiles": service.list_profiles()}


@router.get("/{profile_id}", summary="获取单个 Profile")
async def get_profile(profile_id: str) -> Dict[str, Any]:
    """获取指定 Profile 详情。"""
    from .app import app
    service = app.state.profile_service
    result = service.get_profile(profile_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return result


@router.post("", summary="创建自定义 Profile", status_code=201)
async def create_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """创建自定义投资者偏好画像。"""
    from .app import app
    service = app.state.profile_service
    try:
        return service.create_profile(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{profile_id}", summary="更新自定义 Profile")
async def update_profile(profile_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """更新自定义 Profile。预定义 Profile 不可修改。"""
    from .app import app
    service = app.state.profile_service
    try:
        result = service.update_profile(profile_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{profile_id}", summary="删除自定义 Profile")
async def delete_profile(profile_id: str) -> Dict[str, bool]:
    """删除自定义 Profile。预定义 Profile 不可删除。"""
    from .app import app
    service = app.state.profile_service
    try:
        deleted = service.delete_profile(profile_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        return {"deleted": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{profile_id}/apply", summary="应用 Profile")
async def apply_profile(profile_id: str) -> Dict[str, Any]:
    """应用 Profile 到当前会话（返回参数映射）。"""
    from .app import app
    service = app.state.profile_service
    result = service.apply_profile(profile_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return result


__all__ = ["router"]
