"""返听名额的确定性锁定与抽查抽样。

每班的新片段按校准概率从高到低排序：高分片段优先占满返听名额，
另留一席在剩余片段中等概率抽查。名单与每个片段的入选概率在同一
时点锁定，重试签发不改变名单，导出沿用锁定的入选概率。
"""

from dataclasses import dataclass, field
from random import Random


@dataclass(frozen=True)
class Clip:
    """参与本班返听名额竞争的片段。"""

    clip_id: str
    raw_score: float
    probability: float


@dataclass(frozen=True)
class SelectedClip:
    """名单中的一个锁定片段。"""

    clip_id: str
    raw_score: float
    probability: float
    inclusion_probability: float
    channel: str  # guarantee=高分保底名额，audit=剩余片段抽查名额

    def to_row(self):
        return {
            'id': self.clip_id,
            'raw_score': self.raw_score,
            'calibrated_probability': self.probability,
            'inclusion_probability': self.inclusion_probability,
            'selection_channel': self.channel,
        }


@dataclass(frozen=True)
class LockedList:
    """一次签发时冻结的名单和入选概率。"""

    batch_id: str
    quota: int
    clips: tuple[SelectedClip, ...]
    ranked_ids: tuple[str, ...]
    locked_at: str

    @property
    def clip_ids(self):
        return frozenset(clip.clip_id for clip in self.clips)

    def to_export_rows(self):
        """导出时严格沿用锁定时点的入选概率，不做任何重算。"""
        return [clip.to_row() for clip in self.clips]


def _ranked_clips(clips):
    """校准概率降序；同概率按编号升序，保证排序完全可复算。"""
    return sorted(clips, key=lambda clip: (-clip.probability, clip.clip_id))


def lock_review_list(batch_id, clips, quota, rng, locked_at):
    """锁定本班返听名单与入选概率。

    名额数为 ``quota``。除最后一个抽查名额外，其余名额按校准概率
    从高到低保底锁定；最后一个名额在未保底的剩余片段中等概率抽取。
    保底片段入选概率为 1，被抽查片段入选概率为 ``1/剩余片段数``。
    名单、入选概率和签发时点一并冻结，后续重试只返回同一份名单。
    """
    if quota < 0:
        raise ValueError('返听名额不能为负数')
    unique = {clip.clip_id: clip for clip in clips}
    if len(unique) != len(clips):
        raise ValueError('片段编号重复，无法锁定名单')
    if quota > len(clips):
        raise ValueError(f'返听名额 {quota} 多于片段数 {len(clips)}')

    ranked = _ranked_clips(clips)
    ranked_ids = tuple(clip.clip_id for clip in ranked)
    if quota == 0 or not clips:
        return LockedList(batch_id, quota, (), ranked_ids, locked_at)

    guarantee_count = max(quota - 1, 0)
    guaranteed = ranked[:guarantee_count]
    remaining = ranked[guarantee_count:]

    selected = [
        SelectedClip(
            clip.clip_id, clip.raw_score, clip.probability,
            inclusion_probability=1.0, channel='guarantee',
        )
        for clip in guaranteed
    ]
    if remaining:
        audit_clip = rng.choice(remaining)
        audit_probability = 1.0 / len(remaining)
        selected.append(
            SelectedClip(
                audit_clip.clip_id, audit_clip.raw_score, audit_clip.probability,
                inclusion_probability=audit_probability, channel='audit',
            )
        )

    # 名单对外仍按校准概率排序，抽查命中但分数靠后的片段排在保底之后。
    selected.sort(key=lambda item: (-item.probability, item.clip_id))
    return LockedList(batch_id, quota, tuple(selected), ranked_ids, locked_at)
