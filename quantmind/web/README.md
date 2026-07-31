"""Web 前端升级路径说明。"""
# QuantMind Web 前端

MVP 采用 **Streamlit**（`streamlit_app.py`）作为 Web 统一界面，零 JS、Python 原生、
改码即刷新（`--server.runOnSave true`），最快覆盖研究/回测/监控/行情面板。

## 升级路径：React 19 + Vite（参考 Vibe-Trading）

当需要生产级 UX（复杂交易执行、拖拽策略编排、富交互图表）时，升级为：

- `web/`（React 19 + Vite，dev server 带 HMR 热更新）
- 通过 REST（`/data`、`/research`、`/backtest`）与 WebSocket（`/ws`）对接 `api` 服务
- docker-compose 中 `web` 服务改为 `npm run dev`（Vite HMR）

后端 `api/` 无需改动——其 REST + WebSocket 契约对两种前端一致。
