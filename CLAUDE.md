# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QuantMind (`quantmind/`) is an AI-driven, multi-asset quant research & trading framework for the Chinese market (futures / A-shares / HK / options). The application lives in the `quantmind/` subdirectory of the workspace root `E:\BaiduSyncdisk\学习\量化投资经理`. The project is Python 3.13, packaged as a single editable package `quantmind`. Its positioning docs and design PDFs sit in the workspace root, not in this repo.

It ships **three surfaces that share one engine**:
- **CLI** (`python -m quantmind.cli`) — Typer + Rich, for scripting/automation.
- **API** (`quantmind.api.app`) — FastAPI backend on :8000.
- **Web** (`quantmind.web.streamlit_app`) — Streamlit frontend on :8501.

## Commands

```
# Install (Windows local, no Docker). Use the workbuddy-managed Python 3.13.
scripts\bootstrap_windows.bat

# Run tests (pytest; asyncio_mode auto, pythonpath=. already configured)
.venv\Scripts\python.exe -m pytest                 # whole suite (~491 tests)
.venv\Scripts\python.exe -m pytest tests/test_backtest.py
.venv\Scripts\python.exe -m pytest tests/test_backtest.py -k "paper"   # single test

# Local dev servers (two terminals)
uvicorn quantmind.api.app:app --host 0.0.0.0 --port 8000          # terminal 1
streamlit run quantmind/web/streamlit_app.py --server.port 8501   # terminal 2

# Full stack (Docker: timescaledb + redis + api + web)
docker compose up        # hot-reload via source volume mounts; API swagger at :8000/docs
```

There is no configured linter/formatter — follow PEP 8, type-annotate signatures, prefer immutable/dataclass patterns. Tests are mocked/offline by default (`tests/helpers.py` uses `MockFeed`); real-data and LLM paths degrade gracefully to mocks when no key is set.

## Architecture

Layered from data up, sharing an `EventEngine` event bus driven through an async `DataManager`:

- **`quantmind/core/`** — vnpy-inspired fundamentals: `event.py` (EventType: BAR/SIGNAL/ORDER/TRADE/POSITION/RISK…), `engine.py` (EventEngine), `gateway.py`, `constant.py` (Exchange/Interval/Direction/Offset), `contracts.py`.
- **`quantmind/data/`** — data access. `feed/registry.py` (`build_default_registry`) registers per-asset-class feeds; `feed/base.py` defines `HistoryRequest`/feed protocol. `feed/mock.py` is the offline fallback. `manager.py` (`DataManager`) is the async query entry point. `store/` has Timescale/Redis/InMemory stores plus `DiskBarCache` (Parquet write-cache → second-scope repeat queries, `data_cache/`).
- **`quantmind/research/`** — factor study. Factor types live in `factors/` (`alpha101`, `alpha191`, `technical`, `seat_futures`, WorldQuant-style `alpha_cs` panel factors and DSL `panel_expr.py`). `evaluator.py` (IC/IR/decay/monotonicity), `search/` (co/ea/tot factor-mining algos), `judge.py` (LLM judging), `dedup.py`, `split.py` (train/val/test anti-leak), `cross_sectional_backtest.py`, and `orchestrator.py` (end-to-end AI pipeline).
- **`quantmind/strategy/`** — `runners.py` `run_strategy(mode, ...)` runs the same strategy code across `backtest | paper | live`. Built-in strategies (`dual_ma`, `multifactor`, `vol_target`, `pair`) plus `components/` composable framework (alpha/universe/risk/portfolio building blocks). `backtest/` has engine, broker, `cost.py` cost model, `walkforward.py`, `optimizer.py`.
- **`quantmind/risk/`** — pre-trade risk: `engine.py` (order checks), `limits.py` (limit tiers), `portfolio.py`, `turbulence.py`, `calendar.py` (trading-session gates, Beijing time).
- **`quantmind/ai/`** — LLM layer with a pluggable `provider.py` (`build_provider()`; provider = mock | openai-compatible incl. DeepSeek). `agent.py` (ResearchAgent: idea → spec/factors/strategy code), `factor_gen.py`, `codegen.py`, `sandbox.py` (code-safety), `prompts.py`.
- **`quantmind/live/`** — trading gateways (`ctp/xtp/ib`), `order_manager.py`, `sim.py`, `reconcile.py`. `paper/engine.py` + `promotion.py` (LifecycleManager: RESEARCH→BACKTEST→PAPER→LIVE gate).
- **`quantmind/api/`** — FastAPI app + `services/` (one service class per domain: Data/Factor/Backtest/Research/Risk/Seat/Search/Knowledge…), `ws.py` (WebSocket push), `auth.py`/`routes_auth.py`, `scheduler.py`.
- **`quantmind/web/`** — Streamlit: `streamlit_app.py` + numbered pages `1_仪表盘.py … 21_因子衰减监控.py`; `web/utils/` (api_client, charts, theme).

The end-to-end flow wired by `research/orchestrator.py` and the CLI `e2e` command: **idea → AI research → evidence → factor mining (seed → co/ea/tot search → dedup → OOS backtest) → composite alpha → strategy code → lifecycle registration → paper run → (live)**. The CLI `factor_pipeline`/`cs` commands and `examples/end_to_end_factor_pipeline.py` are runnable references for the research chain.

## Key data-flow conventions

- Multi-asset symbols are `vt_symbol = "{symbol}.{EXCHANGE}"` (e.g. `rb0.SHFE`, `600519.SSE`, `00700.HKEX`); contract sizing via `core/contracts.default_size(vt)`.
- Real-data root env vars (`QM_LOCAL_*_ROOT`, `QM_SEAT_DATA_ROOT`) register local filesystem feeds that take priority; otherwise AKShare/yfinance/mootdx fall back to `mock`.
- Lifecycle state machine lives in `quantmind/paper/promotion.py`; the API `/lifecycle` endpoints gate promotion and `run_strategy(mode=paper)` reuses the same strategy code for paper replay.

## Configuration & environment

Config is Pydantic Settings in `quantmind/config.py` (env prefix `QM_`), read from `.env`. Copy `.env.example`. Key settings: `QM_DB_URL`, `QM_REDIS_URL`, `QM_LLM_PROVIDER` (default `mock`), `QM_LLM_API_KEY`/`QM_LLM_BASE_URL`/`QM_LLM_MODEL`, and the `QM_LOCAL_*_ROOT` data roots. All APIs/web work with no keys: provider defaults to mock. `.env` and `quantmind/config/` (may hold plaintext keys) are gitignored — never commit them.

## Notes / gotchas

- Requires **Python 3.13** (`requires-python = ">=3.13"`). Bootstrap prefers `.workbuddy`'s managed 3.13.12.
- `mootdx` realtime feed is an **optional extra** and conflicts with the pinned `httpx>=0.27` — do not add it by default; A-shares realtime falls back to akshare.
- Time handling requires `Asia/Shanghai` tzdata (Docker installs it).
- There is no `conftest.py`; test-time imports rely on `[tool.pytest.ini_options] pythonpath = ["."]`. Tests use offline mock feeds unless explicitly exercising real data.
