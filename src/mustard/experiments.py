"""Run one reproducible Phys-AD recipe and normalize external baselines."""
import json
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
from .pipeline import Pipeline
from .model import Ollama
from .scoring import evaluate
from .storage import write_json, digest_json


def run_experiment(records, config, work, host):
    client = Ollama(host, work / 'calls')
    pipeline = Pipeline(client, work, config)
    models = [pipeline.config['caption_model']]
    if pipeline.config['mode'] == 'mustard':
        models.append(pipeline.config['reasoning_model'])
    client.resolve_models(models)
    identity = dict(config=pipeline.config, manifest_sha256=digest_json(records), model_digests=client.identities)
    identity_file = work / 'experiment.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('work directory belongs to a different experiment; choose a new --work directory')
    work.mkdir(parents=True, exist_ok=True)
    write_json(identity_file, identity)
    # Capture exactly the sample set and config used, not only its source path.
    write_json(work / 'manifest.json', records)
    write_json(work / 'config.json', pipeline.config)
    runtime = dict(python=sys.version, platform=platform.platform(), model_endpoint=host, packages={})
    for package in ('numpy', 'scipy', 'opencv-python-headless', 'mustard-vad'):
        try:
            runtime['packages'][package] = version(package)
        except PackageNotFoundError:
            runtime['packages'][package] = None
    write_json(work / 'runtime.json', runtime)
    if pipeline.config['mode'] == 'qwen-video':
        predictions = pipeline.qwen(records)
    else:
        patterns = pipeline.train(records)
        write_json(work / 'patterns.json', patterns)
        predictions = pipeline.infer(records, patterns)
    predictions['manifest_sha256'] = digest_json(records)
    write_json(work / 'predictions.json', predictions)
    metrics = evaluate(predictions['predictions'], records)
    write_json(work / 'metrics.json', metrics)
    return dict(work=str(work), count=metrics['count'], mode=pipeline.config['mode'],
                model_digests=client.identities)


def import_baseline(rows, manifest, source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('baseline source must be nonempty')
    expected = {(row['object'], row['anomaly'], row['video']): row for row in manifest if row['split'] == 'test'}
    if len(expected) != sum(row['split'] == 'test' for row in manifest):
        raise ValueError('test manifest contains duplicate sample IDs')
    seen = set()
    normalized = []
    for row in rows:
        key = (row['object'], row['anomaly'], row['video'])
        if key in seen or key not in expected:
            raise ValueError(f'duplicate or unexpected baseline ID: {key}')
        seen.add(key)
        if 'label' in row and row['label'] != expected[key]['label']:
            raise ValueError(f'baseline label differs from manifest: {key}')
        text = row.get('explanation')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'baseline explanation is empty: {key}')
        value = dict(object=key[0], anomaly=key[1], video=key[2], label=expected[key]['label'],
                     explanation=text, explanation_source=source)
        if 'score' in row:
            if not isinstance(row['score'], (int, float)) or not 0 <= row['score'] <= 1:
                raise ValueError(f'invalid baseline score: {key}')
            value['score'] = row['score']
        normalized.append(value)
    if seen != set(expected):
        raise ValueError(f'baseline coverage is incomplete: {len(set(expected) - seen)} missing IDs')
    if not normalized:
        raise ValueError('no baseline samples')
    return dict(schema_version=1, source=source, predictions=sorted(normalized, key=lambda r: (r['object'], r['anomaly'], r['video'])))
