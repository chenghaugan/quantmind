# efinance 数据源集成说明

## 概述

已集成 efinance（东方财富）作为期货数据的备选数据源，与现有的 akshare（新浪）并存。

## 数据源对比

| 特性 | akshare（新浪） | efinance（东方财富） |
|------|----------------|---------------------|
| **日线数据** | 2017年至今（~9.5年） | 理论上更长（待验证） |
| **分钟数据限制** | 1023 根硬限制 | 无明确限制（待验证） |
| **1分钟数据** | 约 5 天 | 理论上更长 |
| **5分钟数据** | 约 1 个月 | 理论上更长 |
| **15分钟数据** | 约 3 个月 | 理论上更长 |
| **30分钟数据** | 约 6 个月 | 理论上更长 |
| **60分钟数据** | 约 1 年 | 理论上更长 |
| **持仓量** | 有 | 暂无 |
| **稳定性** | 稳定 | 依赖网络环境 |

## 使用方法

### 1. 下载脚本

```bash
# 使用 auto 模式（默认）：分钟数据优先 efinance，失败回退 akshare
python scripts/download_index_futures.py --source auto

# 强制使用 efinance
python scripts/download_index_futures.py --source efinance

# 强制使用 akshare
python scripts/download_index_futures.py --source akshare

# 指定周期
python scripts/download_index_futures.py --periods 1d,60m,30m,15m,5m,1m --source auto
```

### 2. 在代码中使用

```python
from quantmind.data.feed import EfinanceFeed, AkShareFuturesFeed
from quantmind.data.feed.base import HistoryRequest
from quantmind.core.constant import Exchange, Interval

# 使用 efinance
feed = EfinanceFeed()
req = HistoryRequest(
    symbol="IF0",
    exchange=Exchange.CFFEX,
    interval=Interval.MINUTE,
)
bars = await feed.fetch_bar_data(req)

# 使用 akshare
feed = AkShareFuturesFeed()
bars = await feed.fetch_bar_data(req)
```

### 3. 数据源注册

efinance 已自动注册到数据源注册表，优先级为 11（在 akshare 的 10 之后）：

```python
from quantmind.data.feed import build_default_registry

registry = build_default_registry()
# 数据源优先级：
# 5: 本地期货CSV（如果配置）
# 10: akshare（新浪）
# 11: efinance（东方财富）
# 20: mootdx（A股）
# ...
```

## 支持的品种

### 股指期货主力（已实现）

- IF0（沪深300主力）→ 行情ID: 8.IF0
- IC0（中证500主力）→ 行情ID: 8.IC0
- IH0（上证50主力）→ 行情ID: 8.IH0
- IM0（中证1000主力）→ 行情ID: 8.IM0

### 商品期货主力（待实现）

目前 efinance 数据源仅支持股指期货主力，商品期货需要后续扩展行情 ID 动态查询功能。

## 注意事项

1. **网络环境**：efinance 依赖东方财富 API，某些网络环境可能无法访问
2. **数据完整性**：efinance 期货数据暂无持仓量字段
3. **行情 ID 映射**：商品期货的行情 ID 需要动态查询，暂未实现
4. **错误处理**：auto 模式下，efinance 失败会自动回退到 akshare

## 测试验证

换到可以访问东方财富 API 的网络环境后，运行以下命令验证：

```bash
# 测试 efinance 数据获取
python scripts/test_efinance.py

# 对比 akshare 和 efinance 的数据量
python scripts/download_index_futures.py --source efinance --periods 1m
python scripts/download_index_futures.py --source akshare --periods 1m
```

## 文件清单

- `quantmind/data/feed/efinance_feed.py` - efinance 数据源实现
- `quantmind/data/feed/__init__.py` - 数据源注册
- `scripts/download_index_futures.py` - 下载脚本（已添加 --source 参数）
- `scripts/test_efinance.py` - efinance 测试脚本

## 后续扩展

1. 实现商品期货行情 ID 动态查询
2. 添加更多数据源（Tushare Pro、BaoStock 等）
3. 优化数据源切换策略（根据数据量自动选择最优源）
