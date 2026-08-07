"""新增因子族测试：gtja191 / qlib158 / academic + alpha101 补全 + registry 全量注册。

覆盖借鉴 HKUDS/Vibe-Trading「5 大族」思路新增的因子族核心子集：
  - gtja191   国泰君安短周期价量因子（25 个，A股风格）
  - qlib158   常用量价技术指标（20 个）
  - academic  学术风格因子·价格代理版（11 个）
  - alpha101  世界量化因子补全至 60 个
"""
from __future__ import annotations

import pytest

from quantmind.research import (
    build_factor_registry,
    list_gtja191, build_gtja191_factor,
    list_qlib158, build_qlib158_factor,
    list_academic, build_academic_factor,
    list_alpha101,
)
from tests.helpers import load_bars


# ---------------------------------------------------------------- 数量断言
def test_gtja191_count():
    assert len(list_gtja191()) == 25


def test_qlib158_count():
    assert len(list_qlib158()) == 20


def test_academic_count():
    assert len(list_academic()) == 11


def test_alpha101_expanded():
    # 原 40 个，补全后应包含新增的代表性因子
    names = list_alpha101()
    assert len(names) == 60
    for n in ("alpha009", "alpha023", "alpha027", "alpha041", "alpha050",
              "alpha095"):
        assert n in names


# ---------------------------------------------------------------- 冒烟：每个因子 compute 等长且有有限值
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "builder,names",
    [
        (build_gtja191_factor, list_gtja191()),
        (build_qlib158_factor, list_qlib158()),
        (build_academic_factor, list_academic()),
    ],
    ids=["gtja191", "qlib158", "academic"],
)
async def test_family_factors_compute(builder, names):
    bars = await load_bars()
    n = len(bars)
    for name in names:
        f = builder(name)
        s = f.compute(bars)
        assert len(s) == n, f"{name}: 长度 {len(s)} != {n}"
        finite = s.dropna()
        # 至少 80% 为有限值（含 compute 末尾 fillna(0) 后的有效区）
        assert finite.notna().sum() / n > 0.8, f"{name} 有效值过少"
        assert f.meta.name == name


# ---------------------------------------------------------------- 代表性因子分类正确
def test_family_categories():
    assert build_gtja191_factor("gtja191_001").meta.category == "gtja191"
    assert build_qlib158_factor("qlib_rsi_14").meta.category == "qlib158"
    assert build_academic_factor("acad_bab").meta.category == "academic"


# ---------------------------------------------------------------- registry 全量注册
def test_registry_registers_new_families():
    reg = build_factor_registry()
    by_name = {f["name"]: f for f in reg.list_factors()}
    # 各新族核心子集应全部纳入 registry
    for name in list_gtja191():
        assert name in by_name, f"gtja191 {name} 未注册"
    for name in list_qlib158():
        assert name in by_name, f"qlib158 {name} 未注册"
    for name in list_academic():
        assert name in by_name, f"academic {name} 未注册"
    # alpha101 精选子集扩展后应含新增代表因子
    for name in ("alpha041", "alpha050", "alpha095"):
        assert name in by_name, f"alpha101 {name} 未注册"
    # 总注册数应显著大于原 28
    assert len(by_name) >= 80
