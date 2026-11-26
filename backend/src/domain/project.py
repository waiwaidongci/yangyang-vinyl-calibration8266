"""Domain entry points used by source import, demo seeding, and the analysis worker."""

from .calibration import fit_calibration
from .chronology import split_records
from .diagnostics import brier_score, reliability_rows, score_holdout
from .fixtures import demo_rows as demo_rows
from .queue import PendingClip, build_queue
from .records import COLUMNS, FIELDS, normalize_rows as normalize_rows
from .reporting import result_tables


SPEC = {
    'columns': COLUMNS,
    'fields': FIELDS,
    'settings': [
        {'key': 'holdout_fraction', 'label': '按日期留出比例', 'default': 0.25, 'min': 0.15, 'max': 0.40, 'step': 0.01},
        {'key': 'reliability_bins', 'label': '可靠性分箱数', 'default': 5, 'min': 3, 'max': 12, 'step': 1},
        {'key': 'relisten_threshold', 'label': '优先返听概率阈值', 'default': 0.65, 'min': 0.05, 'max': 0.95, 'step': 0.01},
    ],
    'modules': {'datasets': '去噪质控台账', 'runs': '保序校准运行', 'results': '可靠性与返听队列', 'reviews': '交班复核'},
    'limitation': '同一去噪版本及人工判定口径下使用。训练范围外的分数沿用端点概率，稀疏分箱需人工核对。',
    'dataset_name': '针澜唱片转录质控台账·2026 年 11 月',
    'recommendation': '按返听顺序定位唱片盘面和起止秒，对照原转录与去噪版，记录结论供下次校准。',
}


def analyze(rows, settings):
    settings = _settings(settings)
    split = split_records(normalize_rows(rows), settings['holdout_fraction'])
    model = fit_calibration((row['raw_score'], row['needs_relisten']) for row in split.train)

    holdout = score_holdout(split.holdout, model)
    raw = [(row.raw_score, row.label) for row in holdout]
    calibrated = [(row.probability, row.label) for row in holdout]
    raw_brier, calibrated_brier = brier_score(raw), brier_score(calibrated)
    bins = settings['reliability_bins']
    reliability = reliability_rows(raw, bins, '原始风险分') + reliability_rows(calibrated, bins, '保序校准')

    clips = (
        PendingClip(row['id'], row['record_ref'], row['side'], row['start_seconds'], row['end_seconds'],
                    row['scored_on'], row['raw_score'])
        for row in split.pending
    )
    queue, action_count = build_queue(clips, model, settings['relisten_threshold'])
    warnings = _warnings(split, holdout, reliability, queue, raw_brier, calibrated_brier)
    return {
        'summary': (
            f'以 {split.cutoff} 为首个留出打分日，使用此前已打分且已确认的 {len(split.train)} 条片段拟合。'
            f'留出 {len(holdout)} 条检验概率，沿用冻结模型为 {len(queue)} 条待判片段评分。'
            f'按 {settings["relisten_threshold"]:.1%} 阈值和训练分值范围安排本班核听 {action_count} 条。'
        ),
        'metrics': {
            '训练片段数': len(split.train), '时间留出片段数': len(holdout),
            '原始分 Brier': raw_brier, '校准后 Brier': calibrated_brier, '本班核听条数': action_count,
        },
        'tables': result_tables(queue, reliability, model, holdout),
        'warnings': warnings,
    }


def _settings(settings):
    if settings.keys() - {parameter['key'] for parameter in SPEC['settings']}:
        raise ValueError('存在本项目未定义的运行参数')
    checked = {}
    for parameter in SPEC['settings']:
        key, label = parameter['key'], parameter['label']
        value = settings.get(key, parameter['default'])
        if type(value) not in (int, float) or not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{label}须在 {parameter['min']} 至 {parameter['max']} 之间")
        if parameter['step'] == 1:
            if value != int(value):
                raise ValueError(f'{label}必须是整数')
            value = int(value)
        checked[key] = value
    return checked


def _warnings(split, holdout, reliability, queue, raw_brier, calibrated_brier):
    warnings = []
    if split.late_confirmations:
        warnings.append(f'切分日前打分的 {split.late_confirmations} 条片段在切分日或之后才确认，已排除出训练和留出验证。')
    sparse = sum(0 < row['count'] < 5 for row in reliability)
    if sparse:
        warnings.append(f'两种评分口径合计有 {sparse} 个非空分箱少于 5 条留出片段，实际返听率可能波动。')
    outside_holdout = sum(row.outside_support for row in holdout)
    outside_pending = sum(row['outside_support'] == '是' for row in queue)
    if outside_holdout or outside_pending:
        warnings.append(f'留出集 {outside_holdout} 条、待判 {outside_pending} 条超出训练分值范围，概率沿用最近端点。')
    if len({row.label for row in holdout}) == 1:
        warnings.append('留出集只有一种人工结论，建议补充更多班次再确认返听阈值。')
    if calibrated_brier > raw_brier:
        warnings.append('本次留出集校准后 Brier 高于原始分，复核时核对分箱与人工确认口径。')
    if not queue:
        warnings.append('本台账没有待判片段，返听队列为空。')
    return warnings
