"""Deterministic demo export for the archive's November transfer shifts."""

from datetime import date, timedelta
from random import Random


SCORES = (0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.86, 0.94)
OUTCOME_RATES = (0.03, 0.10, 0.23, 0.18, 0.44, 0.57, 0.83, 0.89)
FIRST_DAY = date(2026, 11, 2)
PENDING_SCORES = (
    (0.98, 0.20, 0.81, 0.59, 0.04, 0.90),
    (0.25, 0.73, 0.60, 0.46, 0.87, 0.92),
    (0.35, 0.56, 0.76, 0.81, 0.16, 0.96),
    (0.43, 0.73, 0.83, 0.92, 0.66, 0.05),
)


def demo_rows():
    rng = Random(261125)
    rows = []
    for offset in range(16):
        scored_on = FIRST_DAY + timedelta(days=offset)
        for slot in range(6):
            band = (3 * offset + slot) % len(SCORES)
            label = int(rng.random() < OUTCOME_RATES[band])
            lag = (offset + slot) % 3
            confirmed_on = scored_on + timedelta(days=lag)
            rows.append(_clip(offset, slot, SCORES[band], confirmed_on.isoformat(), label))
    for offset, scores in enumerate(PENDING_SCORES, start=16):
        for slot, score in enumerate(scores):
            rows.append(_clip(offset, slot, score, None, None))
    return rows


def _clip(offset, slot, score, confirmed_on, label):
    start = 42 + slot % 3 * 116 + offset % 4 * 5
    return {
        'id': f'VNL-{offset * 6 + slot + 1:04d}',
        'record_ref': f'ZL-DR-{offset + 8:03d}',
        'side': 'A' if slot < 3 else 'B',
        'start_seconds': start,
        'end_seconds': start + 18 + slot % 3 * 4,
        'scored_on': (FIRST_DAY + timedelta(days=offset)).isoformat(),
        'raw_score': score,
        'confirmed_on': confirmed_on,
        'needs_relisten': label,
    }
