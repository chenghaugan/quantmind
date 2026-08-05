"""LLM「评估器 / judge」能力评测（对标 AlphaBench T2 与 T4）。

论文第二大发现：所有 LLM 在零样本因子评估（factor evaluation）上都表现极差。
本模块把评估能力拆成四个可测的原子/组合任务，并给出与「真实回测 IC」
对照的评分函数，用于量化评估 LLM 是否可充当 factor judge：

  - :class:`SignalClassificationTask`（T4a）：判断单个表达式是「信号」还是「噪声」。
  - :class:`PairwiseSelectionTask`（T4b）：两个候选中选出预期表现更好者。
  - :class:`RankingTask`（T2 ranking）：从 N 个候选中选出预期 top-K。
  - :class:`ScoringTask`（T2 scoring）：对单个因子给出 Signal/Performance/Stability/
    WinRate/Skewness 评分。

所有任务都可通过可插拔 ``LLMProvider`` 运行；离线/无 key 时用内置规则
（结构复杂度 / 算子 / 回测 IC 启发式）兜底，使评测框架可跑通、可测试。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..ai.provider import LLMProvider, MockProvider

_logger = logging.getLogger("quantmind.research.judge")

__all__ = [
    "SignalClassificationTask",
    "PairwiseSelectionTask",
    "RankingTask",
    "ScoringTask",
    "JudgeResult",
    "judge_signal",
    "judge_pairwise",
    "judge_ranking",
    "judge_scoring",
    "score_signal_accuracy",
    "score_pairwise_accuracy",
    "score_ranking",
    "score_scoring",
]


# -- 提示词 ---------------------------------------------------------------
_SIGNAL_SYSTEM = (
    "You are an expert quantitative researcher. A 'signal' factor has a meaningful, "
    "nonrandom relationship to future returns; a 'noise' factor is essentially random "
    "with no predictive structure. Given ONE factor expression, decide whether it is "
    "more likely signal or noise. Respond with ONLY a JSON object: "
    '{"signal": "signal"|"noise"}'
)

_PAIRWISE_SYSTEM = (
    "You are an expert quantitative researcher. Given TWO candidate alpha factor "
    "expressions, pick the one more likely to predict future stock returns (higher IC). "
    "Respond with ONLY a JSON object: {\"choice\": 0|1}  (0 = the first factor, "
    "1 = the second factor)."
)

_RANK_SYSTEM = (
    "You are an expert quantitative researcher. Given a pool of candidate factor "
    "expressions, identify the top-K factors most likely to have true predictive power "
    "for stock returns. Respond with ONLY a JSON object: {\"top\": [indices...]} where "
    "indices refer to the 0-based order of the factors as presented."
)

_SCORING_SYSTEM = (
    "You are an expert quantitative researcher performing zero-shot factor scoring. "
    "Given ONE factor expression, predict its quality on 1-5 integer scales. "
    "Respond with ONLY a JSON object: {\"scores\": {\"Signal\": k, \"Performance\": k, "
    '"Stability": k, "WinRate": k, "Skewness": k}}'
)


# -- 结果容器 ---------------------------------------------------------------
@dataclass
class JudgeResult:
    """一次 judge 任务的输出与可选的 ground-truth 对照评分。"""

    task: str
    raw: str = ""
    prediction: object = None
    ground_truth: object = None
    correct: Optional[bool] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "score": self.score,
        }


# -- 任务实现（provider + 提示词 -> 结构化预测） -----------------------------
async def _chat_json(provider: LLMProvider, system: str, user: str, allow):
    """调用 provider 并尝试解析 JSON；失败返回 None。"""
    try:
        text = await provider.chat(system, user)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("judge LLM 调用失败: %s", exc)
        return None, ""
    blob = re.search(r"\{.*\}", text, re.S)
    if not blob:
        return None, text
    try:
        return json.loads(blob.group(0)), text
    except json.JSONDecodeError:
        return None, text


def _mock_signal(expr: str) -> bool:
    """离线兜底：结构复杂/含多样化算子视为 signal。"""
    depth = expr.count("(")
    has_ts = any(t in expr for t in ("Mean", "Std", "Corr", "Rank", "Delta", "TsRank", "Slope"))
    return bool(depth >= 2 and has_ts)


def _mock_pairwise(a: str, b: str) -> int:
    """离线兜底：取结构更复杂者。"""
    return 0 if a.count("(") >= b.count("(") else 1


def _mock_ranking(exprs: List[str], k: int) -> List[int]:
    order = sorted(range(len(exprs)), key=lambda i: -exprs[i].count("("))
    return order[:k]


def _mock_scoring(expr: str) -> Dict[str, int]:
    complexity = min(expr.count("("), 5)
    return {
        "Signal": 1 + (1 if complexity >= 2 else 0),
        "Performance": 1 + (1 if complexity >= 3 else 0),
        "Stability": 1 + (1 if "Mean" in expr or "Std" in expr else 0),
        "WinRate": 1 + (1 if complexity >= 3 else 0),
        "Skewness": 2 + (1 if "Skew" in expr or "power" in expr else 0),
    }


async def judge_signal(expr: str, provider: Optional[LLMProvider] = None) -> JudgeResult:
    """判断单个表达式是信号还是噪声。"""
    provider = provider or MockProvider()
    data, raw = await _chat_json(provider, _SIGNAL_SYSTEM, f"Factor expression: {expr}",
                                 allow=("signal", "noise"))
    if isinstance(data, dict) and data.get("signal") in ("signal", "noise"):
        pred = data["signal"]
    else:
        pred = "signal" if _mock_signal(expr) else "noise"
    return JudgeResult(task="signal_classification", raw=raw, prediction=pred)


async def judge_pairwise(a: str, b: str, provider: Optional[LLMProvider] = None) -> JudgeResult:
    """从两个候选中选出预期表现更好者。"""
    provider = provider or MockProvider()
    user = f"Factor 0: {a}\nFactor 1: {b}"
    data, raw = await _chat_json(provider, _PAIRWISE_SYSTEM, user, allow=(0, 1))
    if isinstance(data, dict) and data.get("choice") in (0, 1):
        pred = int(data["choice"])
    else:
        pred = _mock_pairwise(a, b)
    return JudgeResult(task="pairwise_selection", raw=raw, prediction=pred)


async def judge_ranking(
    expressions: Sequence[str],
    provider: Optional[LLMProvider] = None,
    top_k: int = 3,
) -> JudgeResult:
    """从因子池中选出预期 top-K。"""
    provider = provider or MockProvider()
    pool = [
        f"[{i}] {expr}" for i, expr in enumerate(expressions)
    ]
    user = "Candidate factors:\n" + "\n".join(pool) + f"\nSelect top-{min(top_k, len(expressions))}."
    data, raw = await _chat_json(provider, _RANK_SYSTEM, user, allow="top")
    if isinstance(data, dict) and isinstance(data.get("top"), list):
        pred = [int(i) for i in data["top"] if isinstance(i, int) or str(i).isdigit()]
    else:
        pred = _mock_ranking(list(expressions), min(top_k, len(expressions)))
    return JudgeResult(task="ranking", raw=raw, prediction=pred)


async def judge_scoring(expr: str, provider: Optional[LLMProvider] = None) -> JudgeResult:
    """对单个因子给出质量评分（1-5）。"""
    provider = provider or MockProvider()
    data, raw = await _chat_json(provider, _SCORING_SYSTEM, f"Factor expression: {expr}",
                                 allow="scores")
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        pred = {k: int(v) for k, v in data["scores"].items()}
    else:
        pred = _mock_scoring(expr)
    return JudgeResult(task="scoring", raw=raw, prediction=pred)


# -- 评分（与 ground-truth 对照） -------------------------------------------
def score_signal_accuracy(result: JudgeResult, truth_is_signal: bool) -> JudgeResult:
    """信号分类正确率。``truth_is_signal`` 由真实回测 IC 或标签给出。"""
    result.ground_truth = "signal" if truth_is_signal else "noise"
    result.correct = bool(result.prediction == result.ground_truth)
    result.score = 1.0 if result.correct else 0.0
    return result


def score_pairwise_accuracy(result: JudgeResult, better_index: int) -> JudgeResult:
    """pairwise 选择正确率。``better_index`` 为真实更优者下标。"""
    result.ground_truth = better_index
    result.correct = bool(result.prediction == better_index)
    result.score = 1.0 if result.correct else 0.0
    return result


def score_ranking(result: JudgeResult, true_order: Sequence[int], top_k: int = 1) -> JudgeResult:
    """ranking 正确率：预测 top-k 与真实 top-k 的 Jaccard 重合度 [0,1]。"""
    pred = set(int(i) for i in (result.prediction or []) if isinstance(i, int))
    if not pred:
        result.ground_truth = list(true_order[:top_k])
        result.score = 0.0
        result.correct = False
        return result
    truth = set(int(i) for i in true_order[:top_k])
    inter = len(pred & truth)
    union = len(pred | truth) if pred | truth else 1
    result.ground_truth = list(true_order[:top_k])
    result.score = inter / union
    result.correct = bool(pred == truth)
    return result


def score_scoring(result: JudgeResult, truth: Dict[str, int]) -> JudgeResult:
    """scoring 评分 MAE（归一化到 [0,1]，越小越好：1 - MAE/4）。"""
    pred = result.prediction if isinstance(result.prediction, dict) else {}
    if not pred:
        result.ground_truth = truth
        result.score = 0.0
        result.correct = False
        return result
    keys = list(truth.keys())
    mae = sum(abs(int(pred.get(k, 0)) - int(truth[k])) for k in keys) / max(len(keys), 1)
    result.ground_truth = truth
    result.score = max(0.0, min(1.0, 1.0 - mae / 4.0))
    result.correct = bool(mae < 1.0)
    return result
