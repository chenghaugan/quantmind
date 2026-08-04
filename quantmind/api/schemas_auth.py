"""认证授权相关的数据模型"""
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token 类型")
    expires_in: int = Field(..., description="过期时间（秒）")


class UserInfo(BaseModel):
    """用户信息"""
    username: str = Field(..., description="用户名")
    role: str = Field(..., description="角色")
    created_at: str = Field(..., description="创建时间")
