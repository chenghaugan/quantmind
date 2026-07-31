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

## 6. 本地文件源（china-futures 仓库，CC0，最稳）

无需安装 akshare、无 API 限频，把已克隆的 CSV 仓库直接接入 feed 层即可吃真实数据。
已落地 `LocalFileFeed` / `ChinaFuturesCSVFeed`（`quantmind/data/feed/`），离线可编、离线可测。

```bash
# 克隆（CC0 公共领域；6 大交易所全品种 5min，2015-04~2025-06；约数 GB）
git clone https://github.com/Freddy-Hexas/china-futures-5min-2015-2025
# 设置本地数据根目录（该目录下需有 5min/ 子目录）
export QM_LOCAL_DATA_ROOT=/abs/path/to/china-futures-5min-2015-2025
python -m quantmind.cli info          # 确认 "数据源" 行出现 china_futures_csv(本地,优先)
python -m quantmind.cli factor --symbol IC0 --exchange CFFEX        # 主连自动拼接
python -m quantmind.cli factor --symbol IC2401 --exchange CFFEX     # 具体交割合约
python -m quantmind.cli backtest --symbol rb0 --exchange SHFE       # 商品期货主连回测
```

要点：
- 仓库结构 `5min/<交易所>/<品种>/<品种>YYMM.csv`（交易所：CFFEX/CZCE/DCE/GFEX/INE/SHFE）。
- 请求约定：具体交割合约 `IC2401.CFFEX`（symbol=IC2401）；主连/连续 `IC0.CFFEX` 或
  `IC9999.CFFEX`（自动拼接该品种所有交割月 CSV 为简单主力连续）。大小写不敏感。
- 频率：仓库为 5min；因子/回测主要用日频，feed 自动做 5min→日频降采样（按 UTC 自然日
  聚合，等价于中国交易日，因 UTC 日界≈北京时间 08:00，位于夜盘后、日盘前）。
- 时区：CSV 北京时间为 naive，自动减 8h 转 UTC 存储（与体系一致）。
- 主连是**简单主力连续**（按交割月月末窗口衔接拼接，重叠期归近月），非成交量加权换月，
  价格可能有跳变；如需严格主力连续，后续可接 `ak.futures_main_sina` 或自行构造。
- 文件缺失时自动降级 AKShare → Mock，全链路仍可跑。
- A 股（astock-data-toolkit 落 Parquet）/ 港股 / 期权仍走原有实时源；其本地 Parquet 接入
  可复用 `LocalFileFeed`（实现 `_resolve_paths` 即可），后续按需扩展。
```
