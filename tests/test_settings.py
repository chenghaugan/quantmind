"""AI 设置服务（SettingsService）单元测试：JSON 持久化 + .env 双向同步（离线）。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from quantmind.api.services import settings_service as ss
from quantmind.api.services.settings_service import SettingsService


def _make_service(tmp_path: Path) -> SettingsService:
    """构造指向临时目录的 SettingsService，避免污染仓库。"""
    svc = SettingsService()
    svc.path = tmp_path / "config" / "ai_settings.json"
    env = tmp_path / ".env"
    svc._env_path = lambda: env  # type: ignore[method-assign]
    # 阻断对真实环境变量 / 项目根 .env 的读入，保证无其它配置时默认即 mock（保持用例隔离）
    for k in list(os.environ):
        if k.startswith("QM_"):
            os.environ.pop(k, None)
    ss.get_settings = lambda: _CleanSettings()
    svc.data = svc._load()  # 用临时路径重算（模拟启动时路径即就绪）
    return svc


class _CleanSettings:
    """隔离环境下的“默认”设置：所有 llm_* 都回到代码默认值（provider=mock）。"""

    def __getattr__(self, name: str):
        defaults = {
            "llm_provider": "mock",
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "",
            "llm_temperature": 0.7,
        }
        return defaults.get(name, "")


def test_defaults_mock(tmp_path: Path):
    """无任何配置时默认 mock 且来源为 default。"""
    svc = _make_service(tmp_path)
    g = svc.get()
    assert g["provider"] == "mock"
    assert g["source"] == "default"


def test_save_writes_json_and_env_preserving_other_keys(tmp_path: Path):
    """保存同时写 json 与 .env，且 .env 中其它键与注释被保留。"""
    env = tmp_path / ".env"
    env.write_text('QM_DB_URL="postgresql://x"\n# 注释保留\nFOO=bar\n', encoding="utf-8")
    svc = _make_service(tmp_path)
    out = svc.save({
        "provider": "openai", "api_key": "sk-real-123",
        "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
        "temperature": 0.5,
    })
    assert out["synced_env"] is True
    assert svc.path.exists()
    content = env.read_text(encoding="utf-8")
    assert 'QM_DB_URL="postgresql://x"' in content
    assert "FOO=bar" in content
    assert "QM_LLM_API_KEY=sk-real-123" in content
    assert svc.source() == "json"


def test_read_falls_back_to_env_when_json_missing(tmp_path: Path):
    """删除 json 后，新实例从 .env 回退读取（优先级修复的回归测试）。"""
    env = tmp_path / ".env"
    env.write_text('QM_LLM_PROVIDER=openai\nQM_LLM_API_KEY="sk-real-123"\n'
                   'QM_LLM_BASE_URL=https://api.deepseek.com/v1\nQM_LLM_MODEL=deepseek-chat\n'
                   'QM_LLM_TEMPERATURE=0.5\n', encoding="utf-8")
    svc = _make_service(tmp_path)
    g = svc.get()
    assert g["source"] == "env"
    assert g["provider"] == "openai"
    assert g["api_key"] == "sk-real-123"
    assert g["model"] == "deepseek-chat"
    assert g["temperature"] == 0.5  # 字符串被转换为 float


def test_resave_overwrites_same_env_keys(tmp_path: Path):
    """重复保存只覆盖同 key，不产生重复行。"""
    env = tmp_path / ".env"
    env.write_text("QM_LLM_PROVIDER=openai\nQM_LLM_API_KEY=sk-old\n", encoding="utf-8")
    svc = _make_service(tmp_path)
    svc.save({"provider": "deepseek", "api_key": "sk-new"})
    lines = [l for l in env.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("QM_LLM_")]
    assert len(lines) == 6
    assert any("sk-new" in l for l in lines)
    assert not any("sk-old" in l for l in lines)


def test_bad_temperature_falls_back(tmp_path: Path):
    """非法温度值回退到 0.7。"""
    svc = _make_service(tmp_path)
    out = svc.save({"provider": "mock", "temperature": "not-a-number"})
    assert out["temperature"] == 0.7
