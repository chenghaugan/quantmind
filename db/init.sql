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

-- 多周期连续聚合：1m -> 5m / 1h / 1d
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5m
WITH (timescaledb.continuous) AS
SELECT symbol, exchange, '5m'::text AS interval,
       time_bucket('5 minutes', ts) AS ts,
       FIRST(open, ts) AS open, MAX(high) AS high, MIN(low) AS low,
       LAST(close, ts) AS close, SUM(volume) AS volume,
       LAST(open_interest, ts) AS open_interest, SUM(turnover) AS turnover
FROM bars
WHERE interval = '1m'
GROUP BY symbol, exchange, time_bucket('5 minutes', ts)
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1h
WITH (timescaledb.continuous) AS
SELECT symbol, exchange, '1h'::text AS interval,
       time_bucket('1 hour', ts) AS ts,
       FIRST(open, ts) AS open, MAX(high) AS high, MIN(low) AS low,
       LAST(close, ts) AS close, SUM(volume) AS volume,
       LAST(open_interest, ts) AS open_interest, SUM(turnover) AS turnover
FROM bars
WHERE interval = '1m'
GROUP BY symbol, exchange, time_bucket('1 hour', ts)
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1d
WITH (timescaledb.continuous) AS
SELECT symbol, exchange, '1d'::text AS interval,
       time_bucket('1 day', ts) AS ts,
       FIRST(open, ts) AS open, MAX(high) AS high, MIN(low) AS low,
       LAST(close, ts) AS close, SUM(volume) AS volume,
       LAST(open_interest, ts) AS open_interest, SUM(turnover) AS turnover
FROM bars
WHERE interval = '1m'
GROUP BY symbol, exchange, time_bucket('1 day', ts)
WITH NO DATA;

-- 刷新策略（每 1m 聚合视图每 5 分钟刷新；可按需调整）
SELECT add_continuous_aggregate_policy('bars_5m',
    start_offset => INTERVAL '1 hour', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);

-- 常用索引
CREATE INDEX IF NOT EXISTS idx_bars_sym_exch_ts
    ON bars (symbol, exchange, interval, ts DESC);
