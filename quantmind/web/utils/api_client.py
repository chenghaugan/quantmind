"""API 客户端

性能关键：
- 默认 URL 用 127.0.0.1 而非 localhost（避免 Windows IPv6 解析超时 ~2s/请求）
- 使用全局 httpx.Client 长连接池（复用 TCP 连接，首请求建立后后续请求毫秒级）
"""

import os
import httpx
from typing import Any, Dict, List, Optional

# 直连 IPv4，避免 localhost 解析到 [::1] 超时
API_URL = os.getenv("QM_API_URL", "http://127.0.0.1:8000").rstrip("/")

# 全局长连接客户端（连接池复用，比每次 new Client 快 10-100 倍）
_HTTP = httpx.Client(timeout=30)


class APIClient:
    """QuantMind REST API 客户端"""

    @staticmethod
    def _close():
        try:
            _HTTP.close()
        except Exception:
            pass

    @staticmethod
    def get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict:
        try:
            r = _HTTP.get(f"{API_URL}{path}", params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"error": f"HTTP 错误: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def post(path: str, json: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict:
        try:
            r = _HTTP.post(f"{API_URL}{path}", json=json or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"error": f"HTTP 错误: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def put(path: str, json: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict:
        try:
            r = _HTTP.put(f"{API_URL}{path}", json=json or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"error": f"HTTP 错误: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def delete(path: str, timeout: int = 30) -> Dict:
        try:
            r = _HTTP.delete(f"{API_URL}{path}", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"error": f"HTTP 错误: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def health(timeout: int = 5) -> Dict:
        return APIClient.get("/health", timeout=timeout)

    @staticmethod
    def data(symbol: str, exchange: str, interval: str = "1d",
             start: Optional[str] = None, end: Optional[str] = None,
             page: int = 1, page_size: int = 1000,
             timeout: int = 30) -> Dict:
        """查询行情。返回 ``{"data": [...], "pagination": {...}}`` 或 ``{"error": ...}``。"""
        params: Dict[str, Any] = {
            "symbol": symbol, "exchange": exchange, "interval": interval,
            "page": max(1, int(page)), "page_size": max(1, min(1000, int(page_size))),
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return APIClient.get("/data", params=params, timeout=timeout)

    @staticmethod
    def bars(symbol: str, exchange: str, interval: str = "1d",
             start: Optional[str] = None, end: Optional[str] = None,
             limit: int = 1000, timeout: int = 30) -> list:
        """只要 K 线数组的便捷封装：出错或空数据统一返回 ``[]``。"""
        res = APIClient.data(symbol, exchange, interval, start, end,
                             page=1, page_size=limit, timeout=timeout)
        if not isinstance(res, dict) or "error" in res:
            return []
        return res.get("data") or []

    # ---- 数据质量 ----
    @staticmethod
    def data_quality(symbol: str, exchange: str, interval: str = "1d",
                     start: Optional[str] = None, end: Optional[str] = None,
                     freshness_days: Optional[int] = None, timeout: int = 60) -> Dict:
        params: Dict[str, Any] = {"symbol": symbol, "exchange": exchange, "interval": interval}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if freshness_days:
            params["freshness_days"] = freshness_days
        return APIClient.get("/data/quality", params=params, timeout=timeout)

    @staticmethod
    def research(idea: str, asset_class: Optional[str] = None) -> Dict:
        json = {"idea": idea}
        if asset_class:
            json["asset_class"] = asset_class
        return APIClient.post("/research", json=json, timeout=60)

    @staticmethod
    def factor(symbol: str, exchange: str, factor: str,
               expression: Optional[str] = None, window: int = 20,
               forward_periods: int = 1) -> Dict:
        json = {
            "symbol": symbol,
            "exchange": exchange,
            "factor": factor,
            "window": window,
            "forward_periods": forward_periods,
        }
        if expression:
            json["expression"] = expression
        return APIClient.post("/factor", json=json, timeout=60)

    @staticmethod
    def backtest(strategy: str, symbol: str, exchange: str,
                 mode: str = "backtest", setting: Optional[Dict] = None,
                 capital: float = 1e6, commission: float = 2e-4,
                 cost: bool = False) -> Dict:
        json = {
            "strategy": strategy,
            "symbol": symbol,
            "exchange": exchange,
            "mode": mode,
            "capital": capital,
            "commission": commission,
            "cost": cost,
        }
        if setting:
            json["setting"] = setting
        return APIClient.post("/backtest", json=json, timeout=120)

    @staticmethod
    def order(vt_symbol: str, direction: str, volume: int,
              offset: str = "开", price: float = 0.0) -> Dict:
        return APIClient.post("/order", json={
            "vt_symbol": vt_symbol,
            "direction": direction,
            "offset": offset,
            "volume": volume,
            "price": price,
        })

    @staticmethod
    def orders(timeout: int = 10) -> Dict:
        return APIClient.get("/orders", timeout=timeout)

    @staticmethod
    def positions(timeout: int = 10) -> Dict:
        return APIClient.get("/positions", timeout=timeout)

    @staticmethod
    def cancel_order(order_id: str, timeout: int = 10) -> Dict:
        return APIClient.delete(f"/order/{order_id}", timeout=timeout)

    @staticmethod
    def lifecycle(strategy_id: str, to: str, metrics: Dict, note: str = "") -> Dict:
        json = {"strategy_id": strategy_id, "to": to, "metrics": metrics}
        if note:
            json["note"] = note
        return APIClient.post("/lifecycle", json=json)

    @staticmethod
    def feeds(timeout: int = 5) -> Dict:
        return APIClient.get("/feeds", timeout=timeout)

    @staticmethod
    def factors(timeout: int = 5) -> Dict:
        return APIClient.get("/factors", timeout=timeout)

    # ------------------------------------------------------------------
    # 因子衰减监控（对标 Vibe-Trading strategy-dev-manager）
    # ------------------------------------------------------------------
    @staticmethod
    def factor_decay(timeout: int = 10) -> Dict:
        """列出所有已扫描因子的衰减状态（GET /factors/decay）。"""
        return APIClient.get("/factors/decay", timeout=timeout)

    @staticmethod
    def factor_decay_scan(timeout: int = 120) -> Dict:
        """触发全量因子衰减扫描（POST /factors/decay/scan）。"""
        return APIClient.post("/factors/decay/scan", json={}, timeout=timeout)

    @staticmethod
    def strategies(timeout: int = 10) -> Dict:
        return APIClient.get("/strategies", timeout=timeout)

    @staticmethod
    def strategy_register(name: str, code: str, idea: str = "", timeout: int = 30) -> Dict:
        """注册 AI 生成策略（POST /strategies/register）。"""
        return APIClient.post("/strategies/register",
                              json={"name": name, "code": code, "idea": idea},
                              timeout=timeout)

    # ------------------------------------------------------------------
    # AI 模型设置
    @staticmethod
    def ai_settings(timeout: int = 10) -> Dict:
        return APIClient.get("/settings/ai", timeout=timeout)

    @staticmethod
    def ai_settings_save(payload: Dict, timeout: int = 10) -> Dict:
        return APIClient.put("/settings/ai", json=payload, timeout=timeout)

    @staticmethod
    def ai_settings_test(payload: Optional[Dict] = None, timeout: int = 30) -> Dict:
        return APIClient.post("/settings/ai/test",
                              json=payload or {}, timeout=timeout)

    @staticmethod
    def walkforward(strategy: str, symbol: str, exchange: str,
                   train_window: int, test_window: int, step: int,
                   capital: float = 1e6, cost: bool = False,
                   timeout: int = 180) -> Dict:
        json = {
            "strategy": strategy,
            "symbol": symbol,
            "exchange": exchange,
            "train_window": train_window,
            "test_window": test_window,
            "step": step,
            "capital": capital,
            "cost": cost,
        }
        return APIClient.post("/walkforward", json=json, timeout=timeout)

    # ------------------------------------------------------------------
    # 风控
    # ------------------------------------------------------------------
    @staticmethod
    def risk_profiles(timeout: int = 10) -> Dict:
        return APIClient.get("/risk/profiles", timeout=timeout)

    @staticmethod
    def risk_check(payload: Dict[str, Any], timeout: int = 20) -> Dict:
        return APIClient.post("/risk/check", json=payload, timeout=timeout)

    @staticmethod
    def risk_calendar(day: Optional[str] = None, symbol: str = "rb0",
                      exchange: str = "SHFE", horizon: int = 14,
                      timeout: int = 10) -> Dict:
        params: Dict[str, Any] = {"symbol": symbol, "exchange": exchange, "horizon": horizon}
        if day:
            params["day"] = day
        return APIClient.get("/risk/calendar", params=params, timeout=timeout)

    # ------------------------------------------------------------------
    # 参数优化
    # ------------------------------------------------------------------
    @staticmethod
    def optimize_space(strategy: str = "dual_ma", timeout: int = 10) -> Dict:
        return APIClient.get("/optimize/space", params={"strategy": strategy}, timeout=timeout)

    @staticmethod
    def optimize(payload: Dict[str, Any], timeout: int = 300) -> Dict:
        return APIClient.post("/optimize", json=payload, timeout=timeout)

    @staticmethod
    def optimize_optuna(payload: Dict[str, Any], timeout: int = 300) -> Dict:
        return APIClient.post("/optimize/optuna", json=payload, timeout=timeout)

    # ------------------------------------------------------------------
    # 截面研究
    # ------------------------------------------------------------------
    @staticmethod
    def cs_factors(timeout: int = 10) -> Dict:
        return APIClient.get("/cross-section/factors", timeout=timeout)

    @staticmethod
    def cross_section(payload: Dict[str, Any], timeout: int = 300) -> Dict:
        return APIClient.post("/cross-section", json=payload, timeout=timeout)

    # ------------------------------------------------------------------
    # 因子搜索（co / ea / tot）
    # ------------------------------------------------------------------
    @staticmethod
    def search_factor(payload: Dict[str, Any], timeout: int = 600) -> Dict:
        """因子迭代搜索（POST /factor/search）。"""
        return APIClient.post("/factor/search", json=payload, timeout=timeout)

    @staticmethod
    def factor_pipeline(payload: Dict[str, Any], timeout: int = 900) -> Dict:
        """端到端因子挖掘流水线（POST /factor/pipeline）：挖掘→去冗余→OOS回测→复合组合。"""
        return APIClient.post("/factor/pipeline", json=payload, timeout=timeout)

    @staticmethod
    def factor_e2e_start(payload: Dict[str, Any], timeout: int = 30) -> Dict:
        """端到端流水线（异步启动，POST /factor/e2e/start）：立即返回 task_id。

        任务在后台执行，用 ``factor_e2e_status(task_id)`` 轮询进度/结果，
        彻底规避长跑单次请求超时问题。
        """
        return APIClient.post("/factor/e2e/start", json=payload, timeout=timeout)

    @staticmethod
    def factor_e2e_status(task_id: str, timeout: int = 30) -> Dict:
        """查询端到端流水线后台任务状态（GET /factor/e2e/status/{task_id}）。"""
        return APIClient.get(f"/factor/e2e/status/{task_id}", timeout=timeout)

    @staticmethod
    def factor_e2e(payload: Dict[str, Any], timeout: int = 900) -> Dict:
        """端到端流水线（同步，POST /factor/e2e，向后兼容）。

        长跑可能超出单请求超时；新代码请优先用 ``factor_e2e_start`` + ``factor_e2e_status``。
        """
        return APIClient.post("/factor/e2e", json=payload, timeout=timeout)

    @staticmethod
    def knowledge_search(query: str, top_k: int = 10,
                         kind: Optional[str] = None, timeout: int = 30) -> Dict:
        """知识库语义检索（POST /knowledge/search）。"""
        json: Dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if kind:
            json["kind"] = kind
        return APIClient.post("/knowledge/search", json=json, timeout=timeout)

    @staticmethod
    def knowledge_list(kind: Optional[str] = None, timeout: int = 30) -> Dict:
        """知识库条目列表（GET /knowledge）。"""
        params: Dict[str, Any] = {}
        if kind:
            params["kind"] = kind
        return APIClient.get("/knowledge", params=params, timeout=timeout)

    # ------------------------------------------------------------------
    # 席位因子（商品期货）
    # ------------------------------------------------------------------
    @staticmethod
    def seat_factors(timeout: int = 10) -> Dict:
        return APIClient.get("/seat-factors", timeout=timeout)

    @staticmethod
    def seat_factor(payload: Dict[str, Any], timeout: int = 120) -> Dict:
        return APIClient.post("/seat-factor", json=payload, timeout=timeout)

    # ------------------------------------------------------------------
    # 数据管理与本地数据配置
    # ------------------------------------------------------------------
    @staticmethod
    def data_roots(timeout: int = 10) -> Dict:
        return APIClient.get("/settings/data", timeout=timeout)

    @staticmethod
    def data_roots_save(payload: Dict[str, Any], timeout: int = 15) -> Dict:
        return APIClient.put("/settings/data", json=payload, timeout=timeout)

    @staticmethod
    def data_download(payload: Dict[str, Any], timeout: int = 600) -> Dict:
        return APIClient.post("/data/download", json=payload, timeout=timeout)

    @staticmethod
    def data_files(timeout: int = 15) -> Dict:
        return APIClient.get("/data/files", timeout=timeout)

    # ------------------------------------------------------------------
    # 本地行情仓库（Parquet 写缓存）
    # ------------------------------------------------------------------
    @staticmethod
    def cache_stats(timeout: int = 15) -> Dict:
        """本地行情仓库概览（含逐标的明细）。GET /data/cache。"""
        return APIClient.get("/data/cache", timeout=timeout)

    @staticmethod
    def cache_purge(timeout: int = 30) -> Dict:
        """清空本地行情仓库（删除全部 .parquet）。DELETE /data/cache。"""
        return APIClient.delete("/data/cache", timeout=timeout)

    @staticmethod
    def cache_warm(symbols: list, exchange: str = "SHFE",
                   start: Optional[str] = None, end: Optional[str] = None,
                   timeout: int = 600) -> Dict:
        """预热：把指定标的多真实源拉进本地行情仓库。POST /data/cache/warm。"""
        payload: Dict[str, Any] = {"symbols": symbols, "exchange": exchange}
        if start:
            payload["start"] = start
        if end:
            payload["end"] = end
        return APIClient.post("/data/cache/warm", json=payload, timeout=timeout)

    @staticmethod
    def cache_history(limit: int = 50, timeout: int = 15) -> Dict:
        """本地行情仓库刷新执行历史（最新在前）。GET /data/cache/history。"""
        return APIClient.get(f"/data/cache/history?limit={int(limit)}", timeout=timeout)

    @staticmethod
    def cache_refresh(timeout: int = 600) -> Dict:
        """手动触发全量刷新：把仓库内所有标的从真实源重拉回写。POST /data/cache/refresh。"""
        return APIClient.post("/data/cache/refresh", json=None, timeout=timeout)

    @staticmethod
    def cache_warm_market(timeout: int = 900) -> Dict:
        """全市场（A股+港股）增量预热：未缓存标的分批拉入行情仓库。POST /data/cache/market-warm。"""
        return APIClient.post("/data/cache/market-warm", json=None, timeout=timeout)


    # ------------------------------------------------------------------
    # 告警通知配置
    # ------------------------------------------------------------------
    @staticmethod
    def alert_settings(timeout: int = 10) -> Dict:
        return APIClient.get("/settings/alert", timeout=timeout)

    @staticmethod
    def alert_settings_save(payload: Dict[str, Any], timeout: int = 15) -> Dict:
        return APIClient.put("/settings/alert", json=payload, timeout=timeout)

    # ------------------------------------------------------------------
    # LLM 策略挖掘
    # ------------------------------------------------------------------
    @staticmethod
    def strategy_mining_architect(
        factors: List[Dict[str, Any]],
        constraint: Optional[str] = None,
        template_preference: Optional[str] = None,
        symbol: str = "rb0",
        exchange: str = "SHFE",
        timeout: int = 60,
    ) -> Dict:
        """LLM 策略架构师（POST /strategy-mining/architect）。"""
        payload: Dict[str, Any] = {
            "factors": factors,
            "symbol": symbol,
            "exchange": exchange,
        }
        if constraint:
            payload["constraint"] = constraint
        if template_preference:
            payload["template_preference"] = template_preference
        return APIClient.post("/strategy-mining/architect", json=payload, timeout=timeout)

    @staticmethod
    def strategy_mining_auto_backtest(
        spec: Dict[str, Any],
        strategy_id: Optional[str] = None,
        max_iterations: int = 3,
        min_sharpe: float = 0.5,
        max_drawdown: float = -0.30,
        timeout: int = 120,
    ) -> Dict:
        """自动回测循环（POST /strategy-mining/auto-backtest）。"""
        payload: Dict[str, Any] = {
            "spec": spec,
            "max_iterations": max_iterations,
            "min_sharpe": min_sharpe,
            "max_drawdown": max_drawdown,
        }
        if strategy_id:
            payload["strategy_id"] = strategy_id
        return APIClient.post("/strategy-mining/auto-backtest", json=payload, timeout=timeout)
