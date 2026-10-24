"""PAEval exports and judge prompts, with explicit video identities."""
import re
from collections import defaultdict

PAEVAL_SYSTEM = 'I am an expert in text comparison. I evaluate the similarity between texts by leveraging semantic information, including spatiotemporal relationships between concepts and various elements of events. I then derive a score between 0 and 1, with higher scores indicating greater similarity.'


def paeval_messages(text, labels):
    return [{'role': 'system', 'content': PAEVAL_SYSTEM},
            {'role': 'user', 'content': f'Here is an input text that needs to be scored: {text}. Compare this input text with each entry in the label text library: {labels}. Assign a similarity score, and output the highest score as the final result. Just output the final result and no other reply.'}]


def explanation(response, field='final'):
    title = 'Reasoning:' if field == 'reasoning' else 'Final Explanation:'
    lines = response.splitlines()
    if field == 'reasoning':
        matches = [line.split(title, 1)[1].strip() for line in lines if line.startswith(title)]
        if len(matches) != 1 or not matches[0]:
            raise ValueError('expected one nonempty Reasoning field')
        return matches[0]
    headers = list(re.finditer(r'^Final Explanation:[ \t]*', response, re.M))
    if not headers:
        raise ValueError('missing Final Explanation field')
    if len(headers) != 1:
        raise ValueError('expected exactly one Final Explanation field')
    value = response[headers[0].end():].strip()
    if not value:
        raise ValueError('empty Final Explanation field')
    return value


def prediction_explanation(row, field='final'):
    # A normalized baseline/single-modality record explicitly declares its
    # original explanation. This is not a fallback for a missing fusion field.
    if 'explanation' in row:
        if not row.get('explanation_source') or not isinstance(row['explanation'], str) or not row['explanation'].strip():
            raise ValueError('provided explanation requires nonempty text and an explicit source')
        return row['explanation'].strip()
    return explanation(row['response'], field)


def export_paeval(predictions, field='final'):
    descriptions, explanations, identities = {}, {}, {}
    keys = set()
    for row in sorted(predictions, key=lambda r: (r['object'], r['anomaly'], r['video'])):
        obj, anomaly, video = row['object'], row['anomaly'], row['video']
        key = (obj, anomaly, video)
        if key in keys:
            raise ValueError(f'duplicate prediction: {key}')
        keys.add(key)
        for mapping in (descriptions, explanations, identities):
            mapping.setdefault(obj, {}).setdefault(anomaly, [])
        descriptions[obj][anomaly].append(row['raw_description'])
        explanations[obj][anomaly].append(prediction_explanation(row, field))
        identities[obj][anomaly].append(video)
    return dict(descriptions=descriptions, explanations=explanations, video_ids=identities,
                explanation_field=field)


def similarity_score(text):
    value = text.strip()
    if not re.fullmatch(r'(?:0(?:\.\d+)?|1(?:\.0+)?)', value):
        raise ValueError('similarity judge must return one number in [0, 1]')
    return float(value)


def aggregate_paeval(rows):
    """Average videos within each object/task, then average objects as upstream."""
    groups = defaultdict(list)
    keys = set()
    for row in rows:
        key = (row['object'], row['task'], row['anomaly'], row['video'])
        if key in keys:
            raise ValueError(f'duplicate judge score: {key}')
        keys.add(key)
        score = row['score']
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f'invalid similarity score: {score}')
        groups[(row['object'], row['task'])].append(score)
    if not groups:
        raise ValueError('no judge scores')
    per_object = {}
    by_task = defaultdict(list)
    for (obj, task), values in sorted(groups.items()):
        average = sum(values) / len(values)
        per_object.setdefault(obj, {})[task] = dict(count=len(values), mean=average)
        by_task[task].append(average)
    return dict(per_object=per_object, macro={task: sum(values) / len(values) for task, values in by_task.items()})


def run_paeval(predictions, labels, judge, field='final', tasks=('explain',)):
    rows = []
    exported = export_paeval(predictions, field)
    task_fields = {'descrip': ('descriptions', 'description_label'),
                   'explain': ('explanations', 'explanation_label')}
    for task in tasks:
        source, label_key = task_fields[task]
        for obj, groups in exported[source].items():
            for anomaly, texts in groups.items():
                references = labels[obj][label_key][anomaly]
                if not references:
                    raise ValueError(f'empty judge reference library for {obj}/{anomaly}')
                for video, text in zip(exported['video_ids'][obj][anomaly], texts):
                    if not text.strip():
                        raise ValueError(f'empty {task} input for {obj}/{anomaly}/{video}')
                    answer = judge.chat(paeval_messages(text, references), similarity_score)
                    rows.append(dict(object=obj, anomaly=anomaly, video=video, task=task,
                                     score=similarity_score(answer), response=answer))
    return dict(judge_model=judge.model, explanation_field=field, scores=rows,
                summary=aggregate_paeval(rows))


def pairwise_verdict(text):
    match = re.search(r'\[\[([AB])\]\]\s*$', text)
    if not match or len(re.findall(r'\[\[([AB])\]\]', text)) != 1:
        raise ValueError('pairwise answer must contain exactly one verdict, ending with [[A]] or [[B]]')
    return match.group(1)


def run_pairwise(ours, baseline, labels, judge, seed=2024, field='final'):
    import random
    from .pairwise_prompts import create_prompt
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row['object'], row['anomaly'], row['video'])
            if key in result:
                raise ValueError(f'duplicate pairwise sample: {key}')
            result[key] = row
        return result
    left, right = keyed(ours), keyed(baseline)
    if set(left) != set(right) or not left:
        raise ValueError('pairwise predictions must contain the same nonempty set of video IDs')
    rng = random.Random(seed)
    scores = []
    for key in sorted(left):
        obj, anomaly, video = key
        references = labels[obj]['description_label'][anomaly]
        if not references:
            raise ValueError(f'empty pairwise description library: {obj}/{anomaly}')
        swapped = rng.random() < 0.5
        a = prediction_explanation(right[key] if swapped else left[key], field)
        b = prediction_explanation(left[key] if swapped else right[key], field)
        system, user = create_prompt(references[0], a, b)
        answer = judge.chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], pairwise_verdict)
        selected = pairwise_verdict(answer)
        winner = 'ours' if (selected == 'B' if swapped else selected == 'A') else 'baseline'
        scores.append(dict(object=obj, anomaly=anomaly, video=video, swapped=swapped,
                           verdict=selected, winner=winner, response=answer))
    wins = sum(row['winner'] == 'ours' for row in scores)
    return dict(judge_model=judge.model, seed=seed, explanation_field=field, count=len(scores),
                ours_win_rate=wins / len(scores), baseline_win_rate=1 - wins / len(scores), scores=scores)
