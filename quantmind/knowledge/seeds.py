"""知识库内置种子：经典交易方法论（公开通用知识）。

提供 :func:`ensure_seed_data` 幂等落库：表内已存在相同 ``title`` 的记录则不重复写入，
可被 CLI / 测试 / 初始化脚本调用，作为「领域知识获取层」的库内资料底料。
"""
from __future__ import annotations

import logging
from typing import List

from .store import KnowledgeStore

_logger = logging.getLogger("quantmind.knowledge.seeds")

__all__ = ["METHODOLOGY_SEEDS", "ensure_seed_data"]


#: 内置方法论条目：title / concept（一句话核心）/ summary（简要）/ content（要点）
#: source 统一为 "seed"（表示内置种子，非联网/非用户来源）。
METHODOLOGY_SEEDS: List[dict] = [
    {
        "title": "缠论第三类买点",
        "concept": "价格回抽不重新进入中枢区间形成的次级别买点，属于中枢后的启动确认信号。",
        "summary": "走势中枢由至少三段次级别走势重叠区间构成；第三类买点是中枢形成后，价格回抽不再进入中枢区间（不跌破中枢上沿 ZG）而形成的次级别买点。",
        "content": (
            "缠论（缠中说禅）以中枢为核心组织走势结构。相关基础概念：笔（相邻顶底分型之间的连线）、"
            "线段（至少三笔构成）、中枢（至少三段次级别走势的公共重叠区间，上沿记为 ZG、下沿记为 ZD）。\n"
            "三类买点：\n"
            "1) 第一类买点：下跌趋势中底背弛点（价格新低但动能减弱），趋势反转的起点。\n"
            "2) 第二类买点：第一类买点之后的回调不创新低点，次级别二次确认。\n"
            "3) 第三类买点：一段走势中枢形成后，价格回抽不重新进入中枢区间（回折不跌破中枢上沿 ZG 之下、"
            "即在 ZG 上方企稳），从而形成次级别上升的确认买点。\n"
            "量化含义：可在中枢上沿 ZG 之上，用「回抽不破 ZG」作为趋势延续的过滤条件，"
            "配合动量/突破类因子布局趋势方向。"
        ),
        "source": "seed",
        "tags": ["技术分析", "缠论", "趋势", "买点", "中枢"],
        # 机器可读：可忠实实现，映射到 chan_third_buy 因子 kind（策略层有确定性参考实现）。
        "meta": {
            "implementable": True,
            "kind": "chan_third_buy",
            "operator": "chan_third_buy",
            "evidence": "seed",
        },
    },
    {
        "title": "黄金分割线（斐波那契回撤）",
        "concept": "用斐波那契比例（0.618/0.5/0.382 等）标记行情回调中的关键支撑/阻力位，作为回踩买入的参考区间。",
        "summary": "在一段明显趋势（从阶段高点到低点，或低点到高点）中，回调往往在 0.382/0.5/0.618 等斐波那契回撤位企稳；可把这些价位作为趋势回踩的支撑带。",
        "content": (
            "将一段已完成的明确波动（阶段高 H、低 L）称为起止区间，回撤位 = H - (H - L) * r，r∈{0.382,0.5,0.618}。\n"
            "用法：\n"
            "1) 上升趋势中，价格回踩到 0.618/0.5 回撤位附近且止跌（如收盘站回该位上方）→ 视为潜在趋势延续买点。\n"
            "2) 0.618 是强弱分界：回撤未破 0.618 并重拾升势，趋势仍强；跌破 0.618 则回撤加深。\n"
            "量化含义：可构造「回撤位 + 方向确认」因子——在回踩黄金分割支撑带且出现方向性止跌确认（动量转正/收复前一日高）时做多，跌破关键分割位则离场。"
        ),
        "source": "seed",
        "tags": ["技术分析", "黄金分割", "斐波那契", "回撤", "支撑"],
        # 机器可读：可忠实实现（映射到均值回归 kind，回踩支撑带买入）。
        "meta": {
            "implementable": True,
            "kind": "mean_reversion",
            "operator": "fib_retracement",
            "evidence": "seed",
        },
    },
    {
        "title": "威科夫量价分析",
        "concept": "用成交量与价格的配合关系识别吸筹(markup)、派发(distribution)等主力行为阶段。",
        "summary": "威科夫（Wyckoff）方法通过量价关系把市场行为划分为吸筹→上升→派发→下降四阶段，并识别 Supply/Demand 的失衡（如 Spring、Sign of Strength）。",
        "content": (
            "威科夫量价分析把市场行为拆解为四个阶段：\n"
            "1) Accumulation（吸筹）：聪明资金低位收集，量能萎缩但价格窄幅震荡，常伴随 Spring（假跌破后收回）。\n"
            "2) Markup（上升）：需求主导，价升量增，突破吸筹区间上沿。\n"
            "3) Distribution（派发）：顶部放量滞涨，主力出货，可能出现 Upthrust（假突破后回落）。\n"
            "4) Markdown（下降）：供应主导，价跌量增。\n"
            "量化含义：可构造「量价配合」因子——价格创阶段新高且成交显著放量（Demand 主导），"
            "或在吸筹末期识别放量反转（Spring）作为潜在拐点信号。"
        ),
        "source": "seed",
        "tags": ["威科夫", "量价", "主力行为", "吸筹", "派发"],
    },
    {
        "title": "海龟交易法则",
        "concept": "基于唐奇安通道突破的趋势跟踪系统，以明确规则管理入场、止损与加仓。",
        "summary": "海龟交易法则依赖唐奇安通道（Donchian Channel）突破入场，采用 ATR 仓位管理、固定倍数止损与金字塔加仓的机械式趋势跟踪体系。",
        "content": (
            "海龟交易法是经典机械趋势跟踪系统，核心规则：\n"
            "1) 入场：价格突破 N 日唐奇安通道上沿（20 日新高）做多、跌破下沿做空。\n"
            "2) 退出：反向突破（如 10 日低点）离场。\n"
            "3) 仓位管理：用 ATR（真实波幅均值）以固定风险倍数（如 2%）倒推头寸规模，波动适配。\n"
            "4) 加仓：趋势延续时按固定间隔金字塔加仓，同时也相应上移止损。\n"
            "量化含义：唐奇安通道突破可直接实现为\n"
            "突破因子（当前价相对过去 N 日最高/最低的位置），叠加 ATR 归一化控制波动，"
            "属于典型的 momentum / trend 因子族。"
        ),
        "source": "seed",
        "tags": ["海龟", "唐奇安通道", "趋势跟踪", "ATR", "突破"],
    },
    {
        "title": "TD 序列（DeMark Sequential）",
        "concept": "通过连续计数 13 根 K 线的价格对比，识别趋势衰竭的潜在拐点。",
        "summary": "DeMark Sequential 由 Tom DeMark 提出：先记录 Setup（连续 9 根满足收盘价相对 4 根前更高/更低的 K 线），再计数 Countdown（连续 13 根），在 13 处往往出现趋势衰竭、反转概率上升。",
        "content": (
            "TD Sequential 用于定位趋势中的潜在反转点：\n"
            "1) Setup：连续 9 根 K 线的收盘价逐根高于（买入 Setup）/低于（卖出 Setup）4 根前的收盘价。\n"
            "2) Countdown：在 Setup 完成后，继续计数满足条件的 K 线直至第 13 根。\n"
            "3) 在第 13 根附近，趋势动能趋于衰竭，反转（TD Buy/TD Sell）概率上升。\n"
            "量化含义：TD 计数（已完成的 Setup 根数、距 Countdown 13 的距离）可作为\n"
            "「趋势强度/衰竭」的时序计数因子，常与超买超卖或均值回归逻辑结合使用。"
        ),
        "source": "seed",
        "tags": ["TD序列", "DeMark", "反转", "趋势衰竭", "拐点"],
    },
    {
        "title": "均线多头排列",
        "concept": "短期均线在中期、长期均线之上依次排列，代表上升趋势的健康结构。",
        "summary": "均线多头排列指短周期均线 > 中周期 > 长周期（如 MA5 > MA10 > MA20），并通常伴随价格位于各均线上方，是趋势多头的经典结构确认。",
        "content": (
            "均线多头排列是趋势确认的经典技术形态：\n"
            "1) 定义：MA(短) > MA(中) > MA(长)，且价格通常站上所有均线，均线向上发散。\n"
            "2) 对比空头排列：MA(短) < MA(中) < MA(长) 则代表下行趋势。\n"
            "3) 意义：均线系统反映不同周期参与者的持仓成本，多头排列说明各周期成本依次抬高、"
            "买盘占据主导，常作为动量/趋势因子或交易过滤条件。\n"
            "量化含义：可用均线差分或排序实现\n"
            "「均线多头排列」结构因子（如 max(MA_short)-min(MA_long) 与 MA_mid 的组合），"
            "或作为趋势过滤层叠加到其他因子之上。"
        ),
        "source": "seed",
        "tags": ["均线", "多头排列", "趋势", "移动平均", "结构形态"],
    },
    {
        "title": "布林带均值回归",
        "concept": "价格偏离均线超过两倍标准差被视为超买/超卖，存在向中枢回归的倾向。",
        "summary": "布林带（Bollinger Bands）由中轨（均线）± k 倍标准差构成；价格触及/突破上下轨通常被解读为均值回归信号。",
        "content": (
            "布林带由中轨（N 日均线）与上下轨（中轨 ± k 倍 N 日标准差）构成：\n"
            "1) 带宽（Bandwidth）反映波动率水平，带宽收窄常预示价格即将选择方向。\n"
            "2) 突破上轨通常被解读为短期超买，突破下轨为短期超卖——结合走势可作均值回归交易。\n"
            "3) %B = (价 - 下轨)/(上轨 - 下轨)，将价格位置归一化到 [-1,1] 附近。\n"
            "量化含义：%B 偏离或 (价-均线)/标准差 的标准化值是典型的\n"
            "mean_reversion 类因子；当趋势明确时也可把带宽方向作为突破/波动信号。"
        ),
        "source": "seed",
        "tags": ["布林带", "均值回归", "波动率", "超买超卖"],
    },
]


def ensure_seed_data(store: KnowledgeStore | None = None) -> int:
    """幂等写入内置方法论种子，返回本次实际写入的条数。

    以 ``title`` 判重：表内已存在同名 ``title`` 的方法论记录则跳过。
    传入 ``store=None`` 时使用默认库（``KnowledgeStore()``）。
    """
    store = store or KnowledgeStore()
    written = 0
    for seed in METHODOLOGY_SEEDS:
        title = (seed.get("title") or "").strip()
        if not title:
            continue
        existing = store.search(title, top_k=1, kind="methodology")
        if any((h.get("metadata") or {}).get("title") == title for h in existing):
            # 已存在：仅回填机器可读 meta（seed 提供且库存为空时），不改动用户内容。
            if seed.get("meta") and not (existing[0].get("metadata") or {}).get("meta"):
                if store.update_methodology_meta(title, seed["meta"]):
                    written += 1
            continue
        store.ingest_methodology(
            title=title,
            concept=seed.get("concept", ""),
            summary=seed.get("summary", ""),
            content=seed.get("content", ""),
            source=seed.get("source", "seed"),
            tags=seed.get("tags") or [],
            meta=seed.get("meta"),
        )
        written += 1
    if written:
        _logger.info("知识库已写入 %d 条方法论种子", written)
    return written
