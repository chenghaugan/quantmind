"""Web 端常量定义"""

EXCHANGES = {
    "期货": ["SHFE", "CFFEX", "DCE", "CZCE", "INE", "GFEX"],
    "A股": ["SSE", "SZSE"],
    "港股": ["HKEX"],
}

ALL_EXCHANGES = [ex for exs in EXCHANGES.values() for ex in exs]

EXCHANGE_NAMES = {
    "SHFE": "上海期货交易所",
    "CFFEX": "中国金融期货交易所",
    "DCE": "大连商品交易所",
    "CZCE": "郑州商品交易所",
    "INE": "上海国际能源中心",
    "GFEX": "广州期货交易所",
    "SSE": "上海证券交易所",
    "SZSE": "深圳证券交易所",
    "HKEX": "香港交易所",
}

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

INTERVAL_NAMES = {
    "1m": "1 分钟", "5m": "5 分钟", "15m": "15 分钟", "30m": "30 分钟",
    "1h": "1 小时", "4h": "4 小时", "1d": "日线", "1w": "周线",
}

#: 常用标的速选（点一下就能填进输入框）
SYMBOL_PRESETS = {
    "期货": [
        ("rb0", "SHFE", "螺纹钢主连"),
        ("hc0", "SHFE", "热卷主连"),
        ("cu0", "SHFE", "沪铜主连"),
        ("au0", "SHFE", "沪金主连"),
        ("i0", "DCE", "铁矿石主连"),
        ("m0", "DCE", "豆粕主连"),
        ("MA0", "CZCE", "甲醇主连"),
        ("IF0", "CFFEX", "沪深300期指"),
        ("sc0", "INE", "原油主连"),
        ("si0", "GFEX", "工业硅主连"),
    ],
    "A股": [
        ("600519", "SSE", "贵州茅台"),
        ("601318", "SSE", "中国平安"),
        ("600036", "SSE", "招商银行"),
        ("000001", "SZSE", "平安银行"),
        ("000858", "SZSE", "五粮液"),
        ("300750", "SZSE", "宁德时代"),
    ],
    "港股": [
        ("00700", "HKEX", "腾讯控股"),
        ("09988", "HKEX", "阿里巴巴"),
        ("03690", "HKEX", "美团"),
        ("00939", "HKEX", "建设银行"),
    ],
}

#: 截面研究默认标的篮子
CS_BASKETS = {
    "黑色系（SHFE）": (["rb0", "hc0", "ss0", "wr0"], "SHFE"),
    "有色金属（SHFE）": (["cu0", "al0", "zn0", "pb0", "ni0", "sn0"], "SHFE"),
    "贵金属+能化（SHFE）": (["au0", "ag0", "fu0", "bu0", "ru0"], "SHFE"),
    "大商所农产品": (["m0", "y0", "p0", "a0", "c0"], "DCE"),
    "A股白马": (["600519", "601318", "600036", "600900", "601899"], "SSE"),
}

#: AI 研究可选资产类别
ASSET_CLASS_CHOICES = ["期货", "A股", "港股", "期权"]
ASSET_CLASS_DESC = {
    "期货": "商品 / 金融期货",
    "A股": "沪深主板 / 双创",
    "港股": "香港市场",
    "期权": "ETF / 商品期权",
}

STRATEGIES = {
    "dual_ma": {
        "name": "双均线",
        "desc": "经典趋势跟踪，快慢均线交叉产生多空信号",
        "params": {"fast": 5, "slow": 20},
    },
    "multifactor": {
        "name": "多因子",
        "desc": "动量 + 均值回归 + 波动率打分合成",
        "params": {"specs": [{"name": "momentum_20", "weight": 1.0}]},
    },
    "vol_target": {
        "name": "波动率目标",
        "desc": "动态调整仓位，使组合波动率稳定在目标水平",
        "params": {"target_vol": 0.20, "lookback": 20},
    },
    "pair": {
        "name": "配对交易",
        "desc": "价差 Z-Score 均值回归",
        "params": {"window": 30, "entry_z": 1.5, "exit_z": 0.3},
    },
}

LIFECYCLE_STATES = ["IDEA", "RESEARCH", "BACKTEST", "PAPER", "APPROVED", "LIVE"]

LIFECYCLE_DESC = {
    "IDEA": "想法登记：只有一句话假设，尚未验证",
    "RESEARCH": "因子/信号研究：IC、IR、衰减是否站得住",
    "BACKTEST": "历史回测：含真实成本，看夏普与回撤",
    "PAPER": "模拟盘：实时行情跑单，验证工程链路",
    "APPROVED": "风控评审通过：限额、合约、时段都已确认",
    "LIVE": "实盘运行：真金白银，受风控闸门实时管控",
}

GATEWAYS = {
    "ctp": "CTP（期货）",
    "xtp": "XTP（A股）",
    "ib": "IB（全球）",
}

#: 回测绩效字段中文名
PERF_LABELS = {
    "total_return": "总收益率",
    "annual_return": "年化收益",
    "sharpe": "夏普比率",
    "sharpe_ratio": "夏普比率",
    "max_drawdown": "最大回撤",
    "calmar": "卡玛比率",
    "win_rate": "胜率",
    "profit_factor": "盈亏比",
    "volatility": "年化波动",
    "trade_count": "成交笔数",
    "total_trades": "成交笔数",
    "avg_trade": "笔均盈亏",
    "final_equity": "期末净值",
    "total_commission": "总手续费",
    "total_slippage": "总滑点",
}
