"""返听开班工作台：名额锁定、结论时点、IPW 再加权与快照。

业务时序：
1. ``open_batch`` 开班：从首班起取连续收齐结论的旧班，按入选概率
   倒数加权重新拟合保序曲线；有效样本不足则沿用基准/上一条冻结曲线。
2. ``lock_batch_list`` 签发：按校准概率锁定高分保底名单、抽查名额和
   每个片段的入选概率；同一班重试只返回锁定时点的名单。
3. ``record_conclusion`` 登记人工结论，支持乱序补录；开班与补录并发
   时按名单锁定时点可见的结论取数。
4. ``export_batch`` 导出严格沿用锁定的入选概率。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from random import Random

from .relisten_sampling import Clip, LockedList, lock_review_list
from .weighted_isotonic import (
    WeightedIsotonicCurve,
    fit_weighted_isotonic,
    identity_curve,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Batch:
    batch_id: str
    locked_list: LockedList | None = None
    # clip_id -> (0/1 结论, 结论时间)
    conclusions: dict[str, tuple[int, str]] = field(default_factory=dict)
    fitted_curve: WeightedIsotonicCurve | None = None
    curve_source: str = ''
    opened_at: str = ''
    training_batches: tuple[str, ...] = ()


class RelistenWorkbench:
    """按班管理名单、结论和开班校准曲线的内存工作台，可快照持久化。"""

    def __init__(self, min_training_samples=8, baseline=None, seed=261126):
        if min_training_samples < 1:
            raise ValueError('最小训练样本数必须为正整数')
        self.min_training_samples = min_training_samples
        self.baseline = baseline or identity_curve()
        self._rng = Random(seed)
        self._batches: dict[str, Batch] = {}
        self._order: list[str] = []

    # ----- 开班与再加权 -----

    def open_batch(self, batch_id, at=None):
        """开新班并用连续收齐结论的旧班重新拟合保序曲线。

        只有从首班起、按开班顺序连续全部收齐结论的班级进入训练集；
        任一班未收齐即截断，之后的班级即使收齐也不参与本次拟合。
        有效片段数低于阈值，或加权样本不满足拟合条件时沿用基准曲线。
        """
        if batch_id in self._batches:
            raise ValueError(f'班级 {batch_id} 已开班')
        at = at or _utc_now()

        eligible = []
        for previous_id in self._order:
            previous = self._batches[previous_id]
            if not self._is_complete(previous):
                break
            eligible.append(previous)

        samples = [
            (selected.raw_score, previous.conclusions[selected.clip_id][0],
             selected.inclusion_probability)
            for previous in eligible
            for selected in previous.locked_list.clips
        ]
        try:
            curve = fit_weighted_isotonic(samples, self.min_training_samples, self.baseline)
        except ValueError:
            # 结论单一或分值单一等无法拟合的情形，按业务口径沿用基准。
            curve = self.baseline

        batch = Batch(
            batch_id=batch_id,
            fitted_curve=curve,
            curve_source=curve.source,
            opened_at=at,
            training_batches=tuple(batch.batch_id for batch in eligible),
        )
        self._batches[batch_id] = batch
        self._order.append(batch_id)
        return {
            'batch_id': batch_id,
            'curve_source': curve.source,
            'training_batches': batch.training_batches,
            'training_samples': len(samples),
            'opened_at': at,
        }

    @staticmethod
    def _is_complete(batch):
        locked = batch.locked_list
        if locked is None:
            return False
        return all(clip.clip_id in batch.conclusions for clip in locked.clips)

    def curve_for(self, batch_id):
        """返回开班时冻结的曲线；未开班的班级不允许签发。"""
        return self._require_batch(batch_id).fitted_curve

    # ----- 名单签发与锁定 -----

    def lock_batch_list(self, batch_id, clips, quota, at=None):
        """用开班冻结曲线计算校准概率，并锁定名单和入选概率。

        名单在首次签发时锁定；重复调用（重试）直接返回同一份名单，
        不重新抽样、不改变入选概率，即使期间有旧班补录结论。
        """
        batch = self._require_batch(batch_id)
        if batch.locked_list is not None:
            return batch.locked_list
        at = at or _utc_now()
        calibrated = [
            Clip(
                clip_id=clip['id'],
                raw_score=float(clip['raw_score']),
                probability=batch.fitted_curve.predict(float(clip['raw_score'])),
            )
            for clip in clips
        ]
        locked = lock_review_list(batch_id, calibrated, quota, self._rng, at)
        batch.locked_list = locked
        return locked

    # ----- 结论登记与乱序补录 -----

    def record_conclusion(self, batch_id, clip_id, label, at=None):
        """登记一条人工结论。

        只接受名单锁定时点已在名单内的片段；补录顺序不限。开班重算
        另由 ``open_batch`` 在开班时点读取当时已收齐的结论，补录不会
        追溯改变任何已锁定名单或已冻结曲线。
        """
        batch = self._require_batch(batch_id)
        if batch.locked_list is None:
            raise ValueError(f'班级 {batch_id} 尚未签发名单，不能登记结论')
        if clip_id not in batch.locked_list.clip_ids:
            raise ValueError(f'片段 {clip_id} 不在 {batch_id} 的锁定名单内')
        if label not in (0, 1):
            raise ValueError('人工返听结论必须为 0 或 1')
        if clip_id in batch.conclusions:
            raise ValueError(f'片段 {clip_id} 的结论已登记，结论以首次登记为准')
        batch.conclusions[clip_id] = (int(label), at or _utc_now())
        return batch.conclusions[clip_id][1]

    def conclusions_at(self, batch_id):
        """按名单锁定时点之后、当前已登记的结论快照取数。"""
        batch = self._require_batch(batch_id)
        locked_at = batch.locked_list.locked_at if batch.locked_list else None
        return {
            clip_id: payload
            for clip_id, payload in batch.conclusions.items()
            if locked_at is None or payload[1] >= locked_at
        }

    # ----- 导出 -----

    def export_batch(self, batch_id):
        """导出锁定名单，入选概率沿用签发时点，不随补录或重试变化。"""
        batch = self._require_batch(batch_id)
        if batch.locked_list is None:
            raise ValueError(f'班级 {batch_id} 尚未签发名单')
        rows = batch.locked_list.to_export_rows()
        for row in rows:
            payload = batch.conclusions.get(row['id'])
            row['needs_relisten'] = None if payload is None else payload[0]
        return rows

    # ----- 快照持久化 -----

    def to_snapshot(self):
        return json.dumps(
            {
                'min_training_samples': self.min_training_samples,
                'baseline': _curve_to_dict(self.baseline),
                'order': list(self._order),
                'batches': [_batch_to_dict(self._batches[batch_id]) for batch_id in self._order],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_snapshot(cls, payload):
        data = json.loads(payload) if isinstance(payload, str) else payload
        workbench = cls(
            min_training_samples=data['min_training_samples'],
            baseline=_curve_from_dict(data['baseline']),
        )
        workbench._order = list(data['order'])
        workbench._batches = {
            batch_data['batch_id']: _batch_from_dict(batch_data) for batch_data in data['batches']
        }
        return workbench

    def _require_batch(self, batch_id):
        try:
            return self._batches[batch_id]
        except KeyError:
            raise ValueError(f'未知班级 {batch_id}') from None


def _curve_to_dict(curve):
    return asdict(curve)


def _curve_from_dict(data):
    return WeightedIsotonicCurve(
        scores=tuple(data['scores']),
        probabilities=tuple(data['probabilities']),
        total_weight=float(data['total_weight']),
        source=data['source'],
    )


def _batch_to_dict(batch):
    locked = batch.locked_list
    return {
        'batch_id': batch.batch_id,
        'opened_at': batch.opened_at,
        'curve_source': batch.curve_source,
        'training_batches': list(batch.training_batches),
        'fitted_curve': _curve_to_dict(batch.fitted_curve) if batch.fitted_curve else None,
        'locked_list': asdict(locked) if locked else None,
        'conclusions': {
            clip_id: {'label': payload[0], 'recorded_at': payload[1]}
            for clip_id, payload in batch.conclusions.items()
        },
    }


def _batch_from_dict(data):
    locked_data = data['locked_list']
    locked = None
    if locked_data is not None:
        locked = LockedList(
            batch_id=locked_data['batch_id'],
            quota=locked_data['quota'],
            clips=_selected_clips(locked_data['clips']),
            ranked_ids=tuple(locked_data['ranked_ids']),
            locked_at=locked_data['locked_at'],
        )
    return Batch(
        batch_id=data['batch_id'],
        locked_list=locked,
        conclusions={
            clip_id: (payload['label'], payload['recorded_at'])
            for clip_id, payload in data['conclusions'].items()
        },
        fitted_curve=_curve_from_dict(data['fitted_curve']) if data['fitted_curve'] else None,
        curve_source=data['curve_source'],
        opened_at=data['opened_at'],
        training_batches=tuple(data['training_batches']),
    )


def _selected_clips(rows):
    from .relisten_sampling import SelectedClip

    return tuple(
        SelectedClip(
            clip_id=row['clip_id'],
            raw_score=row['raw_score'],
            probability=row['probability'],
            inclusion_probability=row['inclusion_probability'],
            channel=row['channel'],
        )
        for row in rows
    )
