# 真实数据源接入运行手册

QuantMind 默认用 `MockFeed` 兜底，所有端到端验证目前都是**离线合成数据**。
要接入真实行情、让因子 IC / 回测曲线具备真实参考价值，请在本机或容器内
（需联网环境）按以下步骤操作。

> 注意：当前 WorkBuddy 沙箱环境离线，以下步骤无法在沙箱内执行，必须在你自己的
> 联网机器或 `docker compose` 容器内完成。

## 1. 安装数据源库

```bash
cd quantmind
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[llm]"
# 真实行情依赖（代码内为延迟导入，离线不装也能跑 Mock）
pip install akshare mootdx
# 港股 / 期权如需：pip install yfinance
```

## 2. 字段核对要点

- **期货**：`ak.futures_main_sina` / `ak.futures_zh_daily_sina`（注意主力换月、夜盘）
- **A 股**：`ak.stock_zh_a_hist`
- **港股**：`ak.stock_hk_hist` / `yfinance`
- **期权**：目前仅接入 OHLCV 序列，期权链（expiry/strike）建模未做

重点核对：期货主力换月是否连续、港股是否为延时行情、复权方式是否与回测一致。

## 3. 实跑自检

```bash
python -m quantmind.cli smoke     # 拉真实数据并按字段映射自检
python -m quantmind.cli e2e       # 全链路（无真实数据时仍自动回退 MockFeed）
python -m pytest                  # 回归测试
```

## 4. 期货席位因子 F1-F8（需真实席位数据）

见 `config/seat_sources.example.yaml`：复制为 `config/seat_sources.yaml` 并填充
数据源 / API key（从环境变量读取）后，`compute_seat_factors` 才能跑出真实 IC。
公开来源可选交易所每日持仓排名；付费来源按 `paid_api` 字段接入。

## 5. 容器化（已具备 Dockerfile / docker-compose.yml / .env.example）

```bash
cp .env.example .env              # 填 QM_DB_PASSWORD / QM_LLM_PROVIDER / key
docker compose up --build         # 首次 build 需联网拉基础镜像
# 浏览器开 http://localhost:8501 ；API 文档 http://localhost:8000/docs
```
