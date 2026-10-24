"""MUSTARD command-line entry points."""
import argparse
import json
from pathlib import Path

from .data import index_dataset, load_manifest
from .experiments import run_experiment, import_baseline
from .keyframes import prepare
from .judging import export_paeval, aggregate_paeval, run_paeval, run_pairwise
from .judge_client import Judge
from .model import Ollama
from .pipeline import Pipeline
from .scoring import evaluate
from .storage import write_json
from .validation import validate_inputs


def main():
    parser = argparse.ArgumentParser(description="MUSTARD paper pipeline")
    sub = parser.add_subparsers(dest='command', required=True)
    keyframes = sub.add_parser('keyframes', help='extract aligned raw and trajectory keyframes')
    keyframes.add_argument('--raw-video', required=True, type=Path)
    keyframes.add_argument('--trajectory-video', required=True, type=Path)
    keyframes.add_argument('--tracks', required=True, type=Path)
    keyframes.add_argument('--out', required=True, type=Path)
    keyframes.add_argument('--config', type=Path)
    index = sub.add_parser('index', help='index external Phys-AD-style inputs into a manifest')
    for name in ('raw', 'trajectory', 'tracks', 'plots'):
        index.add_argument(f'--{name}-root', required=True, type=Path)
    index.add_argument('--normal-names', nargs='+', default=['norm', 'normal', 'good', 'anomaly_free'])
    index.add_argument('--out', required=True, type=Path)
    for command in ('train', 'infer', 'experiment'):
        task = sub.add_parser(command, help={'train': 'extract normal patterns', 'infer': 'run test inference', 'experiment': 'run induction, inference, and AUROC for one recipe'}[command])
        task.add_argument('--manifest', required=True, type=Path)
        task.add_argument('--config', type=Path)
        task.add_argument('--work', required=True, type=Path)
        if command != 'experiment':
            task.add_argument('--out', required=True, type=Path)
        task.add_argument('--host', default='http://127.0.0.1:11434')
        task.add_argument('--objects', nargs='+')
        if command == 'infer':
            task.add_argument('--patterns', required=True, type=Path)
    evaluation = sub.add_parser('evaluate', help='compute video-level per-object, macro and pooled AUROC')
    evaluation.add_argument('--predictions', required=True, type=Path)
    evaluation.add_argument('--manifest', required=True, type=Path)
    evaluation.add_argument('--out', required=True, type=Path)
    export = sub.add_parser('export-paeval', help='export external predictions for the PAEval judge')
    export.add_argument('--predictions', required=True, type=Path)
    export.add_argument('--field', choices=['reasoning', 'final'], default='final')
    export.add_argument('--out', required=True, type=Path)
    aggregate = sub.add_parser('aggregate-paeval', help='aggregate externally judged PAEval scores')
    aggregate.add_argument('--scores', required=True, type=Path)
    aggregate.add_argument('--out', required=True, type=Path)
    for name in ('paeval', 'pairwise'):
        judge = sub.add_parser(name, help='run GPT-4o explanation evaluation')
        judge.add_argument('--predictions', required=True, type=Path)
        judge.add_argument('--labels', required=True, type=Path)
        judge.add_argument('--model', default='gpt-4o-2024-08-06')
        judge.add_argument('--base-url', default='https://api.openai.com/v1')
        judge.add_argument('--work', required=True, type=Path)
        judge.add_argument('--out', required=True, type=Path)
        judge.add_argument('--field', choices=['reasoning', 'final'], default='final')
        if name == 'pairwise':
            judge.add_argument('--baseline', required=True, type=Path)
            judge.add_argument('--seed', type=int, default=2024)
        else:
            judge.add_argument('--tasks', nargs='+', choices=['descrip', 'explain'], default=['explain'])
    adapter = sub.add_parser('import-baseline', help='normalize external explanation records using explicit video IDs')
    adapter.add_argument('--records', required=True, type=Path)
    adapter.add_argument('--manifest', required=True, type=Path)
    adapter.add_argument('--source', required=True)
    adapter.add_argument('--out', required=True, type=Path)
    validate = sub.add_parser('validate-inputs', help='check external video/track/plot alignment without a model')
    validate.add_argument('--manifest', required=True, type=Path)
    validate.add_argument('--decode', action='store_true', help='decode every frame instead of checking video metadata')
    validate.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text()) if getattr(args, 'config', None) else None
        if args.command == 'keyframes':
            result = prepare(args.raw_video, args.trajectory_video, args.tracks, args.out, config)
            print(json.dumps({'output': str(args.out), 'indices': result['indices']}))
            return
        if args.command == 'index':
            result = index_dataset(args.raw_root, args.trajectory_root, args.tracks_root, args.plots_root, args.normal_names)
        elif args.command in ('paeval', 'pairwise'):
            judge = Judge(args.work / 'judge_calls', model=args.model, base_url=args.base_url)
            predictions = json.loads(args.predictions.read_text())['predictions']
            labels = json.loads(args.labels.read_text())
            if args.command == 'paeval':
                result = run_paeval(predictions, labels, judge, args.field, args.tasks)
            else:
                baseline = json.loads(args.baseline.read_text())['predictions']
                result = run_pairwise(predictions, baseline, labels, judge, args.seed, args.field)
        elif args.command == 'validate-inputs':
            result = validate_inputs(load_manifest(args.manifest), args.decode)
            write_json(args.out, result)
            if not result['valid']:
                parser.exit(1, f'input validation failed; see {args.out}\n')
            print(f'wrote {args.out}')
            return
        elif args.command == 'import-baseline':
            result = import_baseline(json.loads(args.records.read_text()), json.loads(args.manifest.read_text()), args.source)
        elif args.command == 'export-paeval':
            result = export_paeval(json.loads(args.predictions.read_text())['predictions'], args.field)
        elif args.command == 'aggregate-paeval':
            scores = json.loads(args.scores.read_text())
            result = aggregate_paeval(scores['scores'] if isinstance(scores, dict) else scores)
        elif args.command == 'evaluate':
            result = evaluate(json.loads(args.predictions.read_text())['predictions'], json.loads(args.manifest.read_text()))
        else:
            records = load_manifest(args.manifest)
            if args.objects:
                unknown = set(args.objects) - {record['object'] for record in records}
                if unknown:
                    raise ValueError(f'objects not in manifest: {sorted(unknown)}')
                records = [record for record in records if record['object'] in args.objects]
            if args.command == 'experiment':
                result = run_experiment(records, config, args.work, args.host)
                print(json.dumps(result))
                return
            client = Ollama(args.host, args.work / 'calls')
            pipeline = Pipeline(client, args.work, config)
            client.resolve_models([pipeline.config['caption_model'], pipeline.config['reasoning_model']])
            if args.command == 'train':
                result = pipeline.train(records)
            else:
                result = pipeline.infer(records, json.loads(args.patterns.read_text()))
        write_json(args.out, result)
        print(f'wrote {args.out}')
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'error: {exc}\n')
