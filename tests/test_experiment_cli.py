"""Real file/CLI integration with synthetic videos and mocked HTTP responses."""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from mustard.cli import main
from mustard.pipeline import settings
from mustard.keyframes import select


class ExperimentCLITests(unittest.TestCase):
    def inputs(self, root):
        rows = []
        for split, label, video in [('train', 0, '0000'), ('train', 0, '0001'),
                                    ('test', 0, '0000'), ('test', 1, '0001')]:
            folder = root / 'inputs' / split / video
            folder.mkdir(parents=True)
            for name in ('raw', 'trajectory'):
                writer = cv2.VideoWriter(str(folder / f'{name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
                self.assertTrue(writer.isOpened())
                for index in range(60):
                    writer.write(np.full((24, 32, 3), index + (30 if name == 'trajectory' else 0), dtype=np.uint8))
                writer.release()
            np.save(folder / 'tracks.npy', np.zeros((60, 3, 2), dtype=np.float32))
            cv2.imwrite(str(folder / 'plot.png'), np.full((24, 32, 3), 100, dtype=np.uint8))
            rows.append(dict(object='ball', split=split, video=video, label=label,
                             anomaly=None if split == 'train' else 'norm' if label == 0 else 'leakage',
                             raw_video=str(folder / 'raw.avi'), trajectory_video=str(folder / 'trajectory.avi'),
                             tracks=str(folder / 'tracks.npy'), plot=str(folder / 'plot.png')))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(rows))
        return manifest, rows

    def test_all_paper_recipes_and_judges_via_cli(self):
        calls = []
        def http(request, **kwargs):
            url = request if isinstance(request, str) else request.full_url
            if url.endswith('/api/tags'):
                value = {'models': [{'name': 'qwen2.5vl:7b', 'digest': 'q'}, {'name': 'gemma3:12b', 'digest': 'g'}]}
            else:
                payload = json.loads(request.data)
                calls.append((url, payload))
                user = payload['messages'][-1]['content']
                if url.endswith('/chat/completions'):
                    text = '0.75' if 'needs to be scored' in user else 'Evidence supports A. [[A]]'
                    value = {'choices': [{'message': {'content': text}}], 'model': payload['model']}
                else:
                    if payload['model'].startswith('qwen'):
                        if len(payload['messages'][-1].get('images', [])) == 8:
                            text = 'anomaly score: 0.6\nReasoning: visible movement'
                        elif len(payload['messages']) == 2:
                            text = 'max: 1\nmin: 0\nsummary: a motion curve'
                        else:
                            text = 'An object moves in the image.'
                    elif 'Output exactly one line starting with <Frame' in user:
                        number = re.findall(r'Output exactly one line starting with <Frame (\d+)>:', user)[-1]
                        text = f'<Frame {number}>: normal movement'
                    elif "starting with 'max:'" in user:
                        text = 'max: 1\nmin: 0\nsummary: normal movement'
                    elif 'Remember to provide the anomaly score in the specified format.' in user:
                        text = 'Reasoning: normal motion comparison\nAnomaly Score: 0.6'
                    else:
                        text = 'Reasoning: combined evidence\nAnomaly Score: 0.6\nFinal Explanation: object movement'
                    value = {'model': payload['model'], 'message': {'content': text}}
            return io.BytesIO(json.dumps(value).encode())
        def command(*args):
            with patch('sys.argv', ['mustard', *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
                main()
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen', side_effect=http), patch.dict(os.environ, {'OPENAI_API_KEY': 'unit-test-only'}):
            root = Path(directory)
            manifest, rows = self.inputs(root)
            command('validate-inputs', '--manifest', manifest, '--decode', '--out', root / 'validation.json')
            self.assertTrue(json.loads((root / 'validation.json').read_text())['valid'])
            variants = [['raw'], ['trajectory'], ['raw', 'flow'], ['trajectory', 'flow'], ['raw', 'trajectory', 'flow']]
            full = None
            for i, modalities in enumerate(variants):
                cfg = root / f'config-{i}.json'
                cfg.write_text(json.dumps(settings({'n_reference': 2, 'modalities': modalities})))
                work = root / f'run-{i}'
                args = ('experiment', '--manifest', manifest, '--config', cfg, '--work', work)
                command(*args)
                predictions = json.loads((work / 'predictions.json').read_text())
                self.assertEqual(len(predictions['predictions']), 2)
                self.assertEqual(json.loads((work / 'metrics.json').read_text())['count'], 2)
                count = len(calls)
                command(*args)
                self.assertEqual(len(calls), count, 'resuming must reuse successful HTTP calls')
                if len(modalities) == 3:
                    full = work / 'predictions.json'
            # A different recipe must not overwrite a completed experiment.
            changed = root / 'changed-config.json'
            changed.write_text(json.dumps(settings({'n_reference': 2, 'modalities': ['raw']})))
            old_predictions = full.read_bytes()
            with self.assertRaises(SystemExit):
                command('experiment', '--manifest', manifest, '--config', changed, '--work', full.parent)
            self.assertEqual(full.read_bytes(), old_predictions)
            baseline = None
            for method in ('uniform', 'keyframe'):
                cfg = root / f'qwen-{method}.json'
                cfg.write_text(json.dumps(settings({'mode': 'qwen-video', 'modalities': ['raw'], 'keyframes': {'method': method}})))
                work = root / f'qwen-{method}'
                command('experiment', '--manifest', manifest, '--config', cfg, '--work', work)
                self.assertFalse((work / 'patterns.json').exists())
                self.assertEqual(len(json.loads((work / 'predictions.json').read_text())['predictions']), 2)
                baseline = work / 'predictions.json'
            labels = root / 'labels.json'
            labels.write_text(json.dumps({'ball': {field: {'norm': ['normal'], 'leakage': ['abnormal']}
                                                   for field in ('description_label', 'explanation_label')}}))
            command('paeval', '--predictions', full, '--labels', labels, '--work', root / 'judge', '--out', root / 'paeval.json')
            self.assertEqual(len(json.loads((root / 'paeval.json').read_text())['scores']), 2)
            command('pairwise', '--predictions', full, '--baseline', baseline, '--labels', labels,
                    '--work', root / 'judge', '--out', root / 'pairwise.json')
            self.assertEqual(json.loads((root / 'pairwise.json').read_text())['count'], 2)
            command('evaluate', '--predictions', full, '--manifest', manifest, '--out', root / 'metrics.json')
            self.assertEqual(json.loads((root / 'metrics.json').read_text())['macro_auc'], 0.5)
            external = root / 'external.json'
            external.write_text(json.dumps([dict(object=r['object'], anomaly=r['anomaly'], video=r['video'],
                                                  explanation='external baseline explanation') for r in rows if r['split'] == 'test']))
            command('import-baseline', '--records', external, '--manifest', manifest, '--source', 'external test baseline', '--out', root / 'imported.json')
            self.assertEqual(len(json.loads((root / 'imported.json').read_text())['predictions']), 2)

    def test_baseline_import_rejects_identity_and_label_errors(self):
        from mustard.experiments import import_baseline
        manifest = [dict(object='ball', anomaly='norm', video='0', split='test', label=0)]
        row = dict(object='ball', anomaly='norm', video='0', explanation='normal')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            import_baseline([], manifest, 'baseline')
        with self.assertRaisesRegex(ValueError, 'label differs'):
            import_baseline([{**row, 'label': 1}], manifest, 'baseline')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            import_baseline([row, row], manifest, 'baseline')
        self.assertEqual(import_baseline([row], manifest, 'baseline')['predictions'][0]['label'], 0)

    def test_uniform_indices_and_invalid_recipe(self):
        indices, _ = select(np.zeros((240, 2, 2)), {'method': 'uniform'})
        self.assertEqual(indices, [0, 34, 68, 102, 136, 170, 204, 238])
        with self.assertRaisesRegex(ValueError, 'unknown modality'):
            settings({'modalities': ['raw', 'unknown']})
        with self.assertRaisesRegex(ValueError, 'raw frames only'):
            settings({'mode': 'qwen-video'})


if __name__ == '__main__':
    unittest.main()
