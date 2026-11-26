"""Holdout-only probability diagnostics with fixed probability bins."""

from bisect import bisect_right
from dataclasses import dataclass
from math import fsum


@dataclass(frozen=True)
class HoldoutObservation:
    record_id: str
    scored_on: str
    confirmed_on: str
    raw_score: float
    probability: float
    label: int
    outside_support: bool


def score_holdout(rows, model):
    return [
        HoldoutObservation(
            row['id'], row['scored_on'], row['confirmed_on'], row['raw_score'],
            model.predict(row['raw_score']), row['needs_relisten'], model.outside_support(row['raw_score']),
        )
        for row in rows
    ]


def brier_score(predictions):
    return fsum((probability - label) ** 2 for probability, label in predictions) / len(predictions)


def reliability_rows(predictions, bin_count, series):
    edges = [index / bin_count for index in range(bin_count + 1)]
    interior_edges = edges[1:-1]
    bins = [[] for _ in range(bin_count)]
    for probability, label in predictions:
        bins[bisect_right(interior_edges, probability)].append((probability, label))

    rows = []
    for index, observations in enumerate(bins):
        count = len(observations)
        mean_prediction = fsum(probability for probability, _ in observations) / count if count else None
        positives = sum(label for _, label in observations)
        event_rate = positives / count if count else None
        rows.append({
            'series': series,
            'bin_from': edges[index],
            'bin_to': edges[index + 1],
            'count': count,
            'positives': positives,
            'mean_prediction': mean_prediction,
            'event_rate': event_rate,
            'absolute_gap': abs(mean_prediction - event_rate) if count else None,
        })
    return rows
