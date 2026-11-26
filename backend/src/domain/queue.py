"""Turn pending clip scores into handoff actions for the listening desk."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingClip:
    record_id: str
    record_ref: str
    side: str
    start_seconds: float
    end_seconds: float
    scored_on: str
    raw_score: float


def build_queue(clips, model, threshold):
    ranked = []
    for clip in clips:
        probability = model.predict(clip.raw_score)
        outside = model.outside_support(clip.raw_score)
        if probability >= threshold:
            priority = 0
            action = '优先返听'
            reason = '概率达到返听阈值。定位该片段，对照原转录与去噪版，登记返听结论。'
            if outside:
                reason += '风险分超出训练范围，概率采用端点。'
        elif outside:
            priority = 1
            action = '边界核听'
            reason = '风险分超出训练范围。抽听该片段并登记结论，补充边界样本。'
        else:
            priority = 2
            action = '暂缓'
            reason = '概率低于阈值且风险分位于训练范围内。'
        row = {
            'queue_position': None,
            'id': clip.record_id,
            'record_ref': clip.record_ref,
            'side': clip.side,
            'start_seconds': clip.start_seconds,
            'end_seconds': clip.end_seconds,
            'scored_on': clip.scored_on,
            'raw_score': clip.raw_score,
            'probability': probability,
            'outside_support': '是' if outside else '否',
            'action': action,
            'reason': reason,
        }
        ranked.append((priority, -probability, clip.scored_on, clip.record_id, row))

    queue = []
    action_count = 0
    for priority, _, _, _, row in sorted(ranked, key=lambda entry: entry[:4]):
        if priority < 2:
            action_count += 1
            row['queue_position'] = action_count
        queue.append(row)
    return queue, action_count
