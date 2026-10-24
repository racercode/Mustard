import json
import tempfile
import unittest
from pathlib import Path

from mustard.model import Ollama
from mustard.pipeline import Pipeline
from mustard.scoring import evaluate, parse_score


class PipelineTests(unittest.TestCase):
    def test_failed_call_not_cached_and_config_changes_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def request(payload):
                calls.append(payload)
                return {'message': {'content': 'bad output'}}
            model = Ollama('http://unused', folder, attempts=1, transport=request, identities={'m': 'digest-a'})
            with self.assertRaises(RuntimeError):
                model.chat('m', [], validate=parse_score)
            self.assertFalse(list(Path(folder).glob('*.json')))
            model.transport = lambda payload: {'message': {'content': 'Anomaly Score: 0.6'}}
            self.assertEqual(model.chat('m', [], validate=parse_score), 'Anomaly Score: 0.6')
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 1)
            model.chat('m', [], options={'temperature': 0.2}, validate=parse_score)
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 2)
            model.identities['m'] = 'digest-b'
            model.chat('m', [], validate=parse_score)
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 3)

    def test_normal_pattern_and_fusion_flow(self):
        class FakeClient:
            identities = {'qwen2.5vl:7b': 'q', 'gemma3:12b': 'g'}
            def __init__(self):
                self.fusion = None
            def chat(self, model, messages, options=None, validate=None):
                user = messages[-1]['content']
                if 'Output exactly one line starting with <Frame' in user:
                    import re
                    number = re.findall(r'Output exactly one line starting with <Frame (\d+)>:', user)[-1]
                    text = f'<Frame {number}>: common normal motion'
                elif "starting with 'max:'" in user:
                    text = 'max: 1\nmin: 0\nsummary: normal'
                else:
                    text = 'Reasoning: evidence\nAnomaly Score: 0.6\nFinal Explanation: motion is unusual.'
                    if 'Raw Frame Description' in user:
                        self.fusion = messages
                validate(text)
                return text
        class TestPipeline(Pipeline):
            def captions(self, sample, raw=True, kinds=None):
                return {'raw': ['appearance'] * 8, 'trajectory': ['motion'] * 8}
            def plot_caption(self, sample):
                return 'max: 1\nsummary: motion'
            def fingerprint(self, path):
                return 'hash-' + path
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            pipeline = TestPipeline(client, folder, {'n_reference': 2})
            def sample(video, split='train', label=0):
                return dict(object='ball', video=video, split=split, label=label, anomaly='norm' if label == 0 else 'leakage',
                            raw_video=video, trajectory_video=video, tracks=video, plot=video)
            records = [sample('a'), sample('b'), sample('c', 'test', 1)]
            bundle = pipeline.train(records)
            self.assertEqual({item['video'] for item in bundle['objects']['ball']['references']}, {'a', 'b'})
            self.assertEqual(len(bundle['objects']['ball']['dynamic'].splitlines()), 8)
            result = pipeline.infer(records, bundle)
            self.assertEqual(result['predictions'][0]['score'], 0.6)
            self.assertNotIn('Anomaly Score: 0.6', client.fusion[-1]['content'])
            with self.assertRaisesRegex(ValueError, 'normal training'):
                pipeline.train([sample('d', 'train', 1)])

    def test_auc_and_strict_parser(self):
        predictions = [dict(object='a', video='0', label=0, score=0.5),
                       dict(object='a', video='1', label=1, score=0.5),
                       dict(object='b', video='2', label=1, score=0.9)]
        result = evaluate(predictions)
        self.assertEqual(result['macro_auc'], 0.5)
        self.assertEqual(result['eligible_objects'], 1)
        self.assertIsNone(result['per_object']['b']['auc'])
        for text in ['Anomaly Score: 1.5', 'no score', 'Anomaly Score: 0.2\nAnomaly Score: 0.3']:
            with self.assertRaises(ValueError):
                parse_score(text)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate(predictions + predictions)


if __name__ == '__main__':
    unittest.main()
