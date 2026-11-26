"""Build a chronological holdout and freeze the labels available at its start."""

from dataclasses import dataclass
from decimal import Decimal
from math import ceil

from .records import MIN_HOLDOUT, MIN_TRAIN


@dataclass(frozen=True)
class ChronologicalSplit:
    train: list[dict]
    holdout: list[dict]
    pending: list[dict]
    cutoff: str
    late_confirmations: int


def split_records(rows, fraction):
    history = sorted(
        (row for row in rows if row['needs_relisten'] is not None),
        key=lambda row: (row['scored_on'], row['id']),
    )
    desired_holdout = max(MIN_HOLDOUT, ceil(len(history) * Decimal(str(fraction))))
    cutoff = history[-desired_holdout]['scored_on']
    train = []
    holdout = []
    late_confirmations = 0
    for row in history:
        if row['scored_on'] >= cutoff:
            holdout.append(row)
        elif row['confirmed_on'] < cutoff:
            train.append(row)
        else:
            late_confirmations += 1
    if len(train) < MIN_TRAIN:
        raise ValueError(
            f'{cutoff} 留出日前仅有 {len(train)} 条已确认的训练片段，'
            f'至少需要 {MIN_TRAIN} 条。请调小留出比例或补充较早的已确认记录。'
        )
    pending = [row for row in rows if row['needs_relisten'] is None]
    return ChronologicalSplit(train, holdout, pending, cutoff, late_confirmations)
