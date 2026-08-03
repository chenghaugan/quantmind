# 真实数据端到端验证 (2026-08-03)

## 结论
**无需 Docker Desktop、无需本机 clash 代理** —— agent 沙箱实测可直连外网,直接在沙箱容器内即可拉真实数据回测。

## 运行命令
```bash
cd quantmind
docker run --rm -v "$(pwd)":/app quantmind:latest \
  python -m quantmind.cli backtest --symbol 600000 --exchange SSE --strategy dual_ma --cost
```
(容器默认网络模式即直连外网,不要加 `--network none`。)

## 结果:A 股 600000 真实回测 ✅
| 项 | 值 |
|---|---|
| 数据区间 | 2025-08-03 ~ 2026-08-02(约一年真实交易日,非 mock) |
| 交易笔数 | 23 |
| total_return | -0.0001(dual_ma 震荡市 + 真实成本,真实结果) |
| commission | 115.00 |
| stamp_tax(卖出千1) | 3.21 |
| total_cost | 118.21 |
| MockFeed 降级 | 无(日志未出现 mock 兜底) |

## 已知降级(非阻塞,不影响 A 股主链路)
- `akshare_future`(期货):`Length mismatch: Expected axis has 0 elements` —— **akshare 期货日线接口结构变动,需修复**。
- `akshare_option`(期权):`module 'akshare' has no attribute 'option_szse_50etf_daily'` —— akshare 已移除该接口(已知)。
- `mootdx_astock`:未安装 mootdx,自动回退 akshare(正常)。

## 本机直跑(用户侧,可选)
用户本机也可用 `scripts/bootstrap_windows.bat` 建 venv + 装依赖后直接 `python -m quantmind.cli ...`,同样直连本机网络拉数据,无需 Docker。
