"""全局配置（Pydantic Settings，读取 .env / 环境变量）。"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QM_", env_file=".env", extra="ignore"
    )

    db_url: str = "postgresql://qm:quantmind@timescaledb:5432/quantmind"
    redis_url: str = "redis://redis:6379/0"
    llm_provider: str = "mock"
    api_url: str = "http://api:8000"
    log_level: str = "INFO"
    local_data_root: str = ""  # 本地真实数据根目录（如 china-futures CSV 克隆路径）；非空时注册本地文件源（期货真实数据优先）


@lru_cache
def get_settings() -> Settings:
    return Settings()
