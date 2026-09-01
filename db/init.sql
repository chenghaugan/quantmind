-- QuantMind TimescaleDB 初始化
-- 由 docker-compose 在首次启动 TimescaleDB 时自动执行

-- 启用 timescaledb 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 原始 K 线表（按 (symbol, exchange, interval) 区分多周期；此处以统一 ts 列存多周期）
CREATE TABLE IF NOT EXISTS bars (
    symbol       TEXT        NOT NULL,
    exchange     TEXT        NOT NULL,
    interval     TEXT        NOT NULL DEFAULT '1d',
    ts           TIMESTAMPTZ NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    turnover     DOUBLE PRECISION,
    PRIMARY KEY (symbol, exchange, interval, ts)
);

-- 转为超表（按时间分区，1 天一个 chunk）
SELECT create_hypertable('bars', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- 压缩策略：7 天前的 chunk 自动压缩
ALTER TABLE bars SET (timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, exchange, interval');
SELECT add_compression_policy('bars', INTERVAL '7 days', if_not_exists => TRUE);

-- 多周期连续聚合（bars_5m/1h/1d）：已按用户决策移除 —— 应用行情读取走本地
-- parquet 仓库，不查询这些视图；如将来需要直接从库中查多周期 K 线，
-- 参考历史版本重建视图 + add_continuous_aggregate_policy 刷新策略。

-- 常用索引
CREATE INDEX IF NOT EXISTS idx_bars_sym_exch_ts
    ON bars (symbol, exchange, interval, ts DESC);
