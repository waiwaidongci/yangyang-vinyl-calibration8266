"""Weighted isotonic fit and interpolation on observed score knots."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat


@dataclass(frozen=True)
class Pool:
    first: int
    last: int
    count: int
    positives: int

    @property
    def probability(self):
        return self.positives / self.count


@dataclass(frozen=True)
class Calibration:
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    pools: tuple[Pool, ...]

    def predict(self, score):
        right = bisect_left(self.scores, score)
        if right == 0:
            return self.probabilities[0]
        if right == len(self.scores):
            return self.probabilities[-1]
        if score == self.scores[right]:
            return self.probabilities[right]
        left = right - 1
        fraction = (score - self.scores[left]) / (self.scores[right] - self.scores[left])
        low, high = self.probabilities[left], self.probabilities[right]
        return low + fraction * (high - low)

    def outside_support(self, score):
        return score < self.scores[0] or score > self.scores[-1]


def fit_calibration(samples):
    """Each sample has one vote, including samples with the same raw score."""
    counts = defaultdict(int)
    positives = defaultdict(int)
    for score, label in samples:
        counts[score] += 1
        positives[score] += label

    scores = tuple(sorted(counts))
    if len(scores) < 2:
        raise ValueError('训练集至少需要两种不同原始风险分')
    if sum(positives.values()) in (0, sum(counts.values())):
        raise ValueError('训练集须同时包含需要返听和无需返听的人工确认结论')

    pools = []
    for index, score in enumerate(scores):
        right = Pool(index, index, counts[score], positives[score])
        while pools:
            left = pools[-1]
            # Integer comparisons preserve ties exactly and combine equal plateaus.
            if left.positives * right.count < right.positives * left.count:
                break
            pools.pop()
            right = Pool(left.first, right.last, left.count + right.count, left.positives + right.positives)
        pools.append(right)

    probabilities = tuple(
        probability
        for pool in pools
        for probability in repeat(pool.probability, pool.last - pool.first + 1)
    )
    return Calibration(scores, probabilities, tuple(pools))
