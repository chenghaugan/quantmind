"""认证授权路由"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    USERS_DB,
)
from .schemas_auth import LoginRequest, TokenResponse, UserInfo

router = APIRouter(prefix="/auth", tags=["认证授权"])
security = HTTPBearer()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """用户登录，返回 JWT token"""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return UserInfo(
        username=current_user["username"],
        role=current_user["role"],
        created_at=current_user.get("created_at", "unknown")
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """用户登出（客户端删除 token 即可）"""
    return {"message": "登出成功"}


@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """验证 token 是否有效"""
    return {
        "valid": True,
        "username": current_user["username"],
        "role": current_user["role"]
    }


@router.get("/users")
async def list_users(current_user: dict = Depends(get_current_user)):
    """列出所有用户（仅管理员）"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    
    users = []
    for username, user_data in USERS_DB.items():
        users.append({
            "username": username,
            "role": user_data["role"]
        })
    
    return {"users": users}
