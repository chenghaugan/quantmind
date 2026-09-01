"""合约元数据：默认合约乘数（用于回测/模拟的盈亏计算）。

真实场景应从交易所/网关合约查询获取；此处提供国内常见品种默认值。
"""
from __future__ import annotations

# 商品/金融期货主力合约乘数（每点价值，元）
_SIZE_TABLE = {
    # 金融期货（中金所）
    "IF": 300.0, "IC": 300.0, "IH": 300.0, "IM": 200.0,
    "T": 10000.0, "TF": 10000.0, "TS": 20000.0,
    # 商品期货（SHFE/DCE/CZCE/INE/GFEX）
    "RB": 10.0, "HC": 10.0, "CU": 5.0, "AL": 5.0, "ZN": 5.0, "PB": 5.0,
    "NI": 1.0, "SN": 1.0, "AU": 1000.0, "AG": 15.0, "BU": 10.0, "RU": 10.0,
    "SP": 10.0, "FU": 10.0, "I": 100.0, "J": 100.0, "JM": 60.0, "JM2": 60.0,
    "PG": 20.0, "EB": 5.0, "EG": 10.0, "MA": 10.0, "TA": 5.0, "FG": 20.0,
    "SA": 20.0, "UR": 20.0, "SR": 10.0, "CF": 5.0, "OI": 10.0, "RM": 10.0,
    "Y": 10.0, "P": 10.0, "M": 10.0, "A": 10.0, "C": 10.0, "CS": 10.0,
    "PP": 5.0, "L": 5.0, "V": 5.0, "JD": 10.0, "LH": 16.0, "AP": 10.0,
    "CJ": 5.0, "PK": 5.0, "SF": 5.0, "SM": 5.0, "SI": 5.0, "LC": 1000.0,
    "EC": 50.0, "PS": 5.0,
    # 期权（股指/商品，每点价值较小）
    "IO": 100.0, "MO": 100.0, "HO": 100.0, "TFO": 100.0,
    "CUO": 5.0, "RUO": 10.0, "AUO": 1000.0, "M_O": 10.0, "C_O": 10.0,
    "SO": 10.0, "TAO": 5.0, "MAO": 10.0, "RBO": 10.0, "PGO": 20.0,
    "A_O": 10.0, "CO": 10.0,
}


import re


def default_size(vt_symbol: str) -> float:
    """按合约代码返回默认乘数（其余默认 1）。

    兼容带月份代码/期权后缀的合约：如 ``T2409.CFFEX`` → T（10000）、
    ``I2501.DCE`` → I（100）、``IO2409-C-3900`` → IO（100）。
    """
    sym = vt_symbol.split(".")[0].upper()
    # 剥离数字/后缀，提取字母根（如 T2409 → T、IO2409-C-3900 → IO）
    m = re.match(r"[A-Za-z]+", sym)
    root = m.group(0) if m else sym
    # 依次尝试：完整代码、字母根、递减前缀
    candidates = [sym, root, root[:2], root[:1]]
    for key in candidates:
        if key and key in _SIZE_TABLE:
            return _SIZE_TABLE[key]
    return 1.0
