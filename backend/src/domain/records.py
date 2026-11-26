"""CSV schema, typed normalization, and chronology constraints."""

from datetime import date
from math import isfinite
from re import fullmatch


FIELDS = {
    'id': '片段编号',
    'record_ref': '唱片档号',
    'side': '盘面',
    'start_seconds': '片段起点（秒）',
    'end_seconds': '片段终点（秒）',
    'scored_on': '质控打分日',
    'raw_score': '原始风险分',
    'confirmed_on': '人工确认日',
    'needs_relisten': '人工返听确认（1 是 / 0 否）',
}
COLUMNS = list(FIELDS)
MIN_TRAIN = 8
MIN_HOLDOUT = 4
MAX_SIDE_SECONDS = 3600


def normalize_rows(rows):
    normalized = []
    seen = set()
    for line, source in enumerate(rows, start=2):
        if not isinstance(source, dict) or source.keys() != FIELDS.keys():
            raise ValueError(f'第 {line} 行字段有误，应包含：' + '、'.join(COLUMNS))
        try:
            row = _normalize_record(source)
            if row['id'] in seen:
                raise ValueError(f"片段编号重复（{row['id']}）")
        except ValueError as error:
            raise ValueError(f'第 {line} 行：{error}') from error
        seen.add(row['id'])
        normalized.append(row)

    historical = [row for row in normalized if row['needs_relisten'] is not None]
    if len(historical) < MIN_TRAIN + MIN_HOLDOUT:
        raise ValueError(f'至少需要 {MIN_TRAIN + MIN_HOLDOUT} 条人工确认，才能按时间训练并留出校验')
    latest_scored_day = max(row['scored_on'] for row in historical)
    if len({row['scored_on'] for row in historical}) < 2:
        raise ValueError('人工确认片段至少要覆盖两个不同打分日')
    for row in normalized:
        if row['needs_relisten'] is None and row['scored_on'] <= latest_scored_day:
            raise ValueError(f"待判片段 {row['id']} 的打分日必须晚于历史确认片段的末次打分日")
    return normalized


def _normalize_record(source):
    record_id = _text(source['id'], '片段编号')
    record_ref = _text(source['record_ref'], '唱片档号')
    side = _text(source['side'], '盘面').upper()
    if side not in ('A', 'B'):
        raise ValueError('盘面必须是 A 或 B')
    start = _number(source['start_seconds'], '片段起点', MAX_SIDE_SECONDS)
    end = _number(source['end_seconds'], '片段终点', MAX_SIDE_SECONDS)
    if end <= start:
        raise ValueError('片段终点必须大于起点')
    scored = _day(source['scored_on'], '质控打分日')
    score = _number(source['raw_score'], '原始风险分', 1)

    confirmed = None if _blank(source['confirmed_on']) else _day(source['confirmed_on'], '人工确认日')
    label = _label(source['needs_relisten'])
    if (confirmed is None) != (label is None):
        raise ValueError('人工确认日和 0/1 返听结论必须同时填写或同时留空')
    if confirmed is not None and confirmed < scored:
        raise ValueError('人工确认日必须等于或晚于质控打分日')

    return {
        'id': record_id,
        'record_ref': record_ref,
        'side': side,
        'start_seconds': start,
        'end_seconds': end,
        'scored_on': scored,
        'raw_score': score,
        'confirmed_on': confirmed,
        'needs_relisten': label,
    }


def _blank(value):
    return value is None or isinstance(value, str) and not value.strip()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}不能为空')
    text = value.strip()
    if len(text) > 80 or any(not character.isprintable() for character in text):
        raise ValueError(f'{label}最多 80 个可打印字符')
    return text


def _number(value, label, upper):
    try:
        if type(value) not in (int, float, str):
            raise ValueError
        number = float(value)
        if not isfinite(number) or not 0 <= number <= upper:
            raise ValueError
    except (ValueError, OverflowError):
        raise ValueError(f'{label}须为 0 至 {upper} 的有限数字') from None
    return number


def _day(value, label):
    if isinstance(value, str) and fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value.strip()):
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    raise ValueError(f'{label}须为有效日期，格式 YYYY-MM-DD')


def _label(value):
    if _blank(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in ('0', '1'):
            return int(value)
    elif type(value) is int and value in (0, 1):
        return value
    raise ValueError('人工返听确认须填写 0 或 1，待判时留空')
