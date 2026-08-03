# QuantMind Web 界面使用说明

## 架构
- **后端** FastAPI(`quantmind/api/app.py`,v0.2.0):8000 端口,接入真实引擎(EventEngine / DataManager / LifecycleManager)。
- **前端** Streamlit(`quantmind/web/streamlit_app.py`):8501 端口,通过 REST + WebSocket 调后端。
- 部署:`docker-compose.yml` 含 `timescaledb + redis + api + web` 四服务。

## 前端功能(6 个页)
1. **行情** —— 按合约/交易所查历史,`st.line_chart` 画线。
2. **AI 研究** —— 输入投资想法 → 规格/因子/策略代码(经 `ResearchAgent`)。
3. **因子** —— 因子名或表达式评估(IC/IR/衰减/分位收益)。
4. **回测/模拟/实盘** —— 切 `mode`(backtest/paper/live)+ 网关(ctp/xtp/ib)。
5. **生命周期** —— 策略晋升闸门(BACKTEST→PAPER→APPROVED→LIVE)。
6. **监控** —— WebSocket(`/ws`)实时事件推送 + 手动下单入口。

## 后端接口
`/health` `/feeds` `/data` `/research` `/factor` `/backtest` `/strategies` `/order` `/lifecycle` `/ws`(WebSocket 推送 bar/signal/position/trade/account/log)。交互式文档:`http://localhost:8000/docs`(Swagger)。

## 本机启动(推荐)
```bat
.venv\Scripts\activate
:: 终端1:后端
uvicorn quantmind.api.app:app --host 0.0.0.0 --port 8000
:: 终端2:前端
streamlit run quantmind/web/streamlit_app.py --server.port 8501
```
浏览器打开 **http://localhost:8501**。

## Docker 启动(全栈)
```bash
docker compose up        # 自动起 timescaledb+redis+api+web
# 访问 http://localhost:8501
```

## 状态(2026-08-03 验证)
- 镜像 Web 依赖齐全(`fastapi 0.141.1 / uvicorn 0.52.0 / streamlit 1.60.0`)。
- 沙箱实测后端可运行:`HEALTH_OK`(6s),`/data` 拉到真实行情,`/backtest` 跑通真实成本回测。
- 前端 GUI 需在**用户本机**浏览器访问(沙箱端口隔离,agent 侧起的服务用户看不到界面)。
- 待查: `/data` 的 600000 显示约 143 元(浦发实际约 7-8 元)且 datetime 带奇怪时分秒,疑似 akshare 接口解析/复权/时间戳处理需校准(不影响功能可用性)。
