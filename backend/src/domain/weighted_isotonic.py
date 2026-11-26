"""按入选概率倒数加权的 PAVA 保序拟合。

旧班名单按抽样设计带入选概率：高分保底片段概率为 1，随机抽查片段
概率小于 1。直接把返听率当总体率会高估返听比例，因此每条结论按
入选概率倒数（1/π_i）加权后再做保序回归，把观察样本还原回总体。
"""

from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedBlock:
    score: float
    weight: float
    positives: float

    @property
    def rate(self):
        return self.positives / self.weight


@dataclass(frozen=True)
class WeightedIsotonicCurve:
    """以观察分值为结点的单调不减阶梯曲线，结点间线性插值。"""

    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    total_weight: float
    source: str  # fitted=加权拟合，baseline=沿用基准

    def predict(self, score):
        right = bisect_left(self.scores, score)
        if right == 0:
            return self.probabilities[0]
        if right == len(self.scores):
            return self.probabilities[-1]
        if score == self.scores[right]:
            return self.probabilities[right]
        left = right - 1
        low, high = self.probabilities[left], self.probabilities[right]
        width = self.scores[right] - self.scores[left]
        return low + (score - self.scores[left]) / width * (high - low)


def identity_curve():
    """无训练样本时沿用的基准曲线：校准概率等于原始风险分。"""
    knots = (0.0, 1.0)
    return WeightedIsotonicCurve(knots, knots, total_weight=0.0, source='baseline')


def fit_weighted_isotonic(samples, min_samples=8, baseline=None):
    """加权保序拟合；有效样本不足 ``min_samples`` 时沿用基准曲线。

    ``samples`` 每项为 ``(raw_score, label, inclusion_probability)``，
    label 取 0/1。权重为入选概率倒数，同分片段先合并再进入 PAVA，
    相邻块加权返听率违反单调不减时向前合并。
    """
    if min_samples < 1:
        raise ValueError('最小训练样本数必须为正整数')
    fallback = baseline or identity_curve()
    if len(samples) < min_samples:
        return fallback

    grouped = {}
    order = []
    total_weight = 0.0
    for score, label, inclusion in samples:
        if not 0.0 < inclusion <= 1.0:
            raise ValueError('入选概率必须在 (0, 1] 内才能计算倒数权重')
        if label not in (0, 1):
            raise ValueError('人工返听结论必须为 0 或 1')
        weight = 1.0 / inclusion
        if score not in grouped:
            grouped[score] = [0.0, 0.0]
            order.append(score)
        grouped[score][0] += weight
        grouped[score][1] += weight * label
        total_weight += weight

    blocks = [WeightedBlock(score, grouped[score][0], grouped[score][1]) for score in sorted(order)]
    if len(blocks) < 2:
        raise ValueError('训练集至少需要两种不同原始风险分')
    if all(block.positives == 0.0 for block in blocks) or all(
        block.positives == block.weight for block in blocks
    ):
        raise ValueError('训练集须同时包含需要返听和无需返听的人工确认结论')

    pooled = []
    for block in blocks:
        right = block
        while pooled:
            left = pooled[-1]
            # 交叉相乘比较加权率，避免浮点除法；等率块合并为同一平台。
            if left.positives * right.weight < right.positives * left.weight:
                break
            pooled.pop()
            right = WeightedBlock(
                right.score, left.weight + right.weight, left.positives + right.positives
            )
        pooled.append(right)

    scores = tuple(block.score for block in pooled)
    probabilities = tuple(block.rate for block in pooled)
    return WeightedIsotonicCurve(scores, probabilities, total_weight, source='fitted')
