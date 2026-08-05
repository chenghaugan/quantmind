"""T2 judge 任务测试（research/judge.py）。

覆盖：信号分类 / 两两选择 / ranking / scoring 四个任务都能产出结构化预测，
且与 ground-truth 对照的评分函数（acc / jaccard / mae）正常工作。
默认用 MockProvider（离线确定性兜底）。
"""
from __future__ import annotations

import pytest

from quantmind.ai.provider import MockProvider
from quantmind.research.judge import (
    judge_signal,
    judge_pairwise,
    judge_ranking,
    judge_scoring,
    score_signal_accuracy,
    score_pairwise_accuracy,
    score_ranking,
    score_scoring,
)


@pytest.mark.asyncio
async def test_signal_classification_offline():
    """简单 vs 复杂表达式 → signal/noise 判定（离线兜底可跑通，来自 Mock 兜底）。"""
    res = await judge_signal("Mean($close, 20)", provider=MockProvider())
    assert res.task == "signal_classification"
    assert res.prediction in ("signal", "noise")


@pytest.mark.asyncio
async def test_pairwise_offline():
    """两个候选中应返回 0 或 1 之一。"""
    res = await judge_pairwise("Mean($close, 5)", "Std($close, 20)", provider=MockProvider())
    assert res.task == "pairwise_selection"
    assert res.prediction in (0, 1)


@pytest.mark.asyncio
async def test_ranking_offline():
    """ranking 返回下标列表（长度 ≤ top_k）。"""
    exprs = ["Mean($close,5)", "Std($close,20)", "Corr($close,$volume,10)", "Log($close)"]
    res = await judge_ranking(exprs, provider=MockProvider(), top_k=2)
    assert res.task == "ranking"
    assert isinstance(res.prediction, list)
    assert len(res.prediction) <= 2
    assert all(isinstance(i, int) for i in res.prediction)


@pytest.mark.asyncio
async def test_scoring_offline():
    """scoring 返回各维度 1-5 评分 dict。"""
    res = await judge_scoring("Corr($close, $volume, 10)", provider=MockProvider())
    assert res.task == "scoring"
    assert isinstance(res.prediction, dict)
    for k in ("Signal", "Performance", "Stability", "WinRate", "Skewness"):
        assert k in res.prediction
        assert 1 <= int(res.prediction[k]) <= 5


def test_signal_scoring_accuracy():
    """信号分类评分：对则 correct=True 且 score=1。"""
    res = score_signal_accuracy(
        type("R", (), {"prediction": "signal", "raw": ""})(), truth_is_signal=True)
    assert res.correct is True and res.score == 1.0
    res2 = score_signal_accuracy(
        type("R", (), {"prediction": "noise", "raw": ""})(), truth_is_signal=True)
    assert res2.correct is False and res2.score == 0.0


def test_pairwise_scoring_accuracy():
    """pairwise 评分：对则 correct=True。"""
    res = score_pairwise_accuracy(
        type("R", (), {"prediction": 1, "raw": ""})(), better_index=1)
    assert res.correct is True and res.score == 1.0


def test_ranking_jaccard():
    """ranking 评分：完全重合 → score 1，部分重合 → (0,1)，无重合 → 0。"""
    res = score_ranking(
        type("R", (), {"prediction": [0, 1], "raw": ""})(), true_order=[0, 1, 2], top_k=2)
    assert res.score == 1.0
    res2 = score_ranking(
        type("R", (), {"prediction": [0, 1], "raw": ""})(), true_order=[1, 2, 3], top_k=2)
    # 真实 top-2 = {1,2}；预测 {0,1} → 交 {1} 并 {0,1,2} → 1/3
    assert 0.0 < res2.score < 1.0
    res3 = score_ranking(
        type("R", (), {"prediction": [0], "raw": ""})(), true_order=[1, 2], top_k=1)
    assert res3.score == 0.0


def test_scoring_mae():
    """scoring 评分：精确 → score=1；偏离 → 1 - MAE/4。"""
    truth = {"Signal": 3, "Performance": 3, "Stability": 3, "WinRate": 3, "Skewness": 3}
    res = score_scoring(
        type("R", (), {"prediction": {k: 3 for k in truth}, "raw": ""})(), truth=truth)
    assert res.score == 1.0
    res2 = score_scoring(
        type("R", (), {"prediction": {k: 1 for k in truth}, "raw": ""})(), truth=truth)
    assert 0.0 < res2.score < 1.0
