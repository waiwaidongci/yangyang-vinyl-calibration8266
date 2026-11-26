"""Chinese labels and stable column layouts shared by the UI and CSV exports."""

from .records import FIELDS


RELIABILITY_FIELDS = {
    'series': '评分口径', 'bin_from': '概率下限（含）', 'bin_to': '概率上限（末箱含 1）',
    'count': '留出片段数', 'positives': '确认返听数', 'mean_prediction': '平均预测',
    'event_rate': '实际返听率', 'absolute_gap': '绝对差',
}
POOL_FIELDS = {
    'score_from': '训练最低分', 'score_to': '训练最高分', 'distinct_scores': '不同分值数',
    'count': '训练片段数', 'positives': '确认返听数', 'probability': '校准概率',
}
HOLDOUT_FIELDS = {
    'id': FIELDS['id'], 'scored_on': FIELDS['scored_on'], 'confirmed_on': FIELDS['confirmed_on'],
    'raw_score': FIELDS['raw_score'], 'probability': '校准概率',
    'needs_relisten': FIELDS['needs_relisten'], 'outside_support': '超出训练分值范围',
}
QUEUE_FIELDS = {
    'queue_position': '返听顺序', 'id': FIELDS['id'], 'record_ref': FIELDS['record_ref'],
    'side': FIELDS['side'], 'start_seconds': FIELDS['start_seconds'], 'end_seconds': FIELDS['end_seconds'],
    'scored_on': FIELDS['scored_on'], 'raw_score': FIELDS['raw_score'], 'probability': '校准概率',
    'outside_support': '超出训练分值范围', 'action': '交班处置', 'reason': '安排依据与操作',
}


def result_tables(queue, reliability, model, holdout):
    pool_rows = [
        {
            'score_from': model.scores[pool.first], 'score_to': model.scores[pool.last],
            'distinct_scores': pool.last - pool.first + 1, 'count': pool.count,
            'positives': pool.positives, 'probability': pool.probability,
        }
        for pool in model.pools
    ]
    holdout_rows = [
        {
            'id': row.record_id, 'scored_on': row.scored_on, 'confirmed_on': row.confirmed_on,
            'raw_score': row.raw_score, 'probability': row.probability, 'needs_relisten': row.label,
            'outside_support': '是' if row.outside_support else '否',
        }
        for row in holdout
    ]
    return [
        _table('relisten_queue', '待判评分与返听交班队列', QUEUE_FIELDS, queue),
        _table('reliability', '留出集分箱可靠性', RELIABILITY_FIELDS, reliability),
        _table('calibration_pools', '训练保序分段', POOL_FIELDS, pool_rows),
        _table('holdout_predictions', '逐片段留出验证', HOLDOUT_FIELDS, holdout_rows),
    ]


def _table(key, title, fields, rows):
    return {'key': key, 'title': title, 'columns': list(fields), 'labels': fields.copy(), 'rows': rows}
