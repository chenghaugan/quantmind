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
    local_stock_root: str = ""  # 本地 A 股数据根目录（如 astock-data-toolkit / a-stock-data 的 Parquet/CSV）；非空时注册 ChinaAStockParquetFeed（优先于 mootdx）
    local_hk_root: str = ""  # 本地港股数据根目录（东方财富/第三方导出的港股日频 Parquet/CSV）；非空时注册 ChinaHKAStockParquetFeed（优先于 em_hk）
    local_option_root: str = ""  # 本地期权数据根目录（股指/ETF/商品期权日频 Parquet/CSV）；非空时注册 ChinaOptionParquetFeed（优先于 akshare_option）
    seat_data_root: str = ""  # 期货席位持仓排名数据根目录（如 TradingAgents_for_Futures 的 qihuo/database/positioning）；非空时启用 F1-F8 真实席位因子


@lru_cache
def get_settings() -> Settings:
    return Settings()
