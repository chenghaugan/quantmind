"""展示层时间格式化：数据层统一 UTC 存储，界面统一北京时间。

用法（页面内）：
    from utils.timefmt import to_bj
    df["datetime"] = to_bj(df["datetime"])
"""
from __future__ import annotations

import pandas as pd

TZ_BJ = "Asia/Shanghai"


def to_bj(values):
    """把 UTC 时间（序列/列表/Index）转成北京时间，并去掉时区标记。

    去掉 tz 标记是为了 plotly 图表与 dataframe 直接展示为北京时间字符串，
    而不显示 +08:00 后缀。
    """
    ts = pd.to_datetime(pd.Series(values), utc=True)
    return ts.dt.tz_convert(TZ_BJ).dt.tz_localize(None)
