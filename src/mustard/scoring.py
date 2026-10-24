"""Strict scoring and explicit missing-score behavior."""
import re
from collections import defaultdict


SCORE = re.compile(r'^\s*Anomaly\s+Score\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$', re.I | re.M)


def parse_score(text):
    matches = SCORE.findall(text)
    if len(matches) != 1:
        raise ValueError('expected exactly one Anomaly Score line')
    score = float(matches[0])
    if not 0 <= score <= 1:
        raise ValueError('anomaly score must be in [0, 1]')
    return score


def strip_scores(text):
    return '\n'.join(line for line in text.splitlines() if 'anomaly score' not in line.lower()).strip()


def auc(pairs):
    # Mann-Whitney statistic, including ties, using average ranks.
    pairs = sorted(pairs, key=lambda pair: pair[1])
    positives = sum(label == 1 for label, _ in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][1] == pairs[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(label == 1 for label, _ in pairs[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(predictions, expected=None):
    import math
    groups = defaultdict(list)
    keys = set()
    for row in predictions:
        key = (row['object'], row.get('anomaly'), row['video'])
        if key in keys:
            raise ValueError(f'duplicate prediction: {key}')
        keys.add(key)
        if row['label'] not in (0, 1) or not isinstance(row['score'], (int, float)) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError(f'invalid prediction: {key}')
        groups[row['object']].append((row['label'], row['score']))
    if expected is not None:
        expected_rows = {(row['object'], row.get('anomaly'), row['video']): row['label']
                         for row in expected if row['split'] == 'test'}
        if keys != set(expected_rows):
            raise ValueError(f'prediction coverage differs from test manifest: {len(set(expected_rows) - keys)} missing, {len(keys - set(expected_rows))} unexpected')
        if any(row['label'] != expected_rows[(row['object'], row.get('anomaly'), row['video'])] for row in predictions):
            raise ValueError('prediction labels differ from test manifest')
    if not groups:
        raise ValueError('no predictions to evaluate')
    per_object = {obj: dict(count=len(rows), auc=auc(rows)) for obj, rows in sorted(groups.items())}
    eligible = [row['auc'] for row in per_object.values() if row['auc'] is not None]
    return dict(count=len(keys), per_object=per_object, eligible_objects=len(eligible),
                macro_auc=sum(eligible) / len(eligible) if eligible else None,
                pooled_auc=auc([pair for rows in groups.values() for pair in rows]))
