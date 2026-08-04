"""认证授权 API 测试"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from quantmind.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_login_success(client):
    """登录成功"""
    r = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


def test_login_wrong_password(client):
    """登录失败（错误密码）"""
    r = client.post("/auth/login", json={"username": "admin", "password": "wrongpwd"})
    assert r.status_code == 401
    assert "用户名或密码错误" in r.json().get("detail", "")


def test_login_user_not_found(client):
    """登录失败（用户不存在）"""
    r = client.post("/auth/login", json={"username": "nonexistent", "password": "testpwd"})
    assert r.status_code == 401


def test_me_authenticated(client):
    """获取当前用户信息（需认证）"""
    # 先登录获取 token
    login = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = login.json()["access_token"]
    
    # 使用 token 访问 /me
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"


def test_me_unauthenticated(client):
    """获取当前用户信息（未认证）"""
    r = client.get("/auth/me")
    assert r.status_code == 401  # FastAPI HTTPBearer 标准行为


def test_me_invalid_token(client):
    """获取当前用户信息（无效 token）"""
    r = client.get("/auth/me", headers={"Authorization": "Bearer invalid_token"})
    assert r.status_code == 401


def test_validate_token(client):
    """验证 token 有效性"""
    login = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = login.json()["access_token"]
    
    r = client.get("/auth/validate", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["username"] == "admin"


def test_logout(client):
    """用户登出"""
    login = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = login.json()["access_token"]
    
    r = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["message"] == "登出成功"


def test_list_users_admin(client):
    """列出用户（管理员权限）"""
    login = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    token = login.json()["access_token"]
    
    r = client.get("/auth/users", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "users" in body
    assert len(body["users"]) >= 1
    assert any(u["username"] == "admin" for u in body["users"])
