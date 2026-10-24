import unittest

from mustard.judging import export_paeval, aggregate_paeval, similarity_score
from mustard.scoring import evaluate


class EvaluationTests(unittest.TestCase):
    def test_incomplete_predictions_are_rejected(self):
        expected = [dict(object='ball', anomaly='norm', video='0', split='test', label=0),
                    dict(object='ball', anomaly='bad', video='1', split='test', label=1)]
        predictions = [dict(object='ball', anomaly='norm', video='0', label=0, score=0.1)]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            evaluate(predictions, expected)

    def test_paeval_preserves_order_and_field(self):
        rows = [dict(object='ball', anomaly='norm', video=video, raw_description='appearance',
                     response='Reasoning: technical\nAnomaly Score: 0.2\nFinal Explanation: plain')
                for video in ['0002', '0001']]
        result = export_paeval(rows)
        self.assertEqual(result['video_ids']['ball']['norm'], ['0001', '0002'])
        self.assertEqual(result['explanations']['ball']['norm'], ['plain', 'plain'])
        self.assertEqual(export_paeval(rows, 'reasoning')['explanations']['ball']['norm'], ['technical', 'technical'])
        self.assertEqual(export_paeval(rows, 'final')['explanations']['ball']['norm'], ['plain', 'plain'])

    def test_missing_final_does_not_fall_back_to_reasoning(self):
        row = dict(object='ball', anomaly='norm', video='0', raw_description='appearance',
                   response='Reasoning: technical\nAnomaly Score: 0.2')
        with self.assertRaisesRegex(ValueError, 'missing Final Explanation'):
            export_paeval([row])
        self.assertEqual(export_paeval([row], 'reasoning')['explanations']['ball']['norm'], ['technical'])

    def test_paeval_object_average_and_invalid_score(self):
        rows = [dict(object='a', anomaly='norm', video='0', task='explain', score=0),
                dict(object='a', anomaly='norm', video='1', task='explain', score=0),
                dict(object='b', anomaly='norm', video='0', task='explain', score=1)]
        self.assertEqual(aggregate_paeval(rows)['macro']['explain'], 0.5)
        with self.assertRaises(ValueError):
            similarity_score('Similarity: 0.9')
        with self.assertRaises(ValueError):
            similarity_score('1.1')

class JudgeTests(unittest.TestCase):
    def test_pairwise_swap_maps_back_to_system_identity(self):
        from mustard.judging import run_pairwise
        class FakeJudge:
            model = 'gpt-4o-2024-08-06'
            def chat(self, messages, validate):
                text = 'Compared evidence. [[A]]'
                validate(text)
                return text
        rows = [dict(object='ball', anomaly='norm', video=str(i),
                     response='Reasoning: motion is normal\nAnomaly Score: 0.1\nFinal Explanation: normal motion') for i in range(8)]
        labels = {'ball': {'description_label': {'norm': ['normal ball']}}}
        result = run_pairwise(rows, rows, labels, FakeJudge())
        self.assertEqual(result['judge_model'], 'gpt-4o-2024-08-06')
        for row in result['scores']:
            self.assertEqual(row['winner'], 'baseline' if row['swapped'] else 'ours')
        with self.assertRaisesRegex(ValueError, 'same nonempty'):
            run_pairwise(rows, rows[:-1], labels, FakeJudge())

    def test_judge_success_cache_and_invalid_response(self):
        import tempfile
        from pathlib import Path
        from mustard.judge_client import Judge
        from mustard.judging import similarity_score
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def request(payload):
                calls.append(payload)
                return {'choices': [{'message': {'content': '0.75'}}]}
            judge = Judge(folder, attempts=1, transport=request)
            for _ in range(2):
                self.assertEqual(judge.chat([], similarity_score), '0.75')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['model'], 'gpt-4o-2024-08-06')
            judge.transport = lambda payload: {'choices': [{'message': {'content': 'invalid'}}]}
            with self.assertRaises(RuntimeError):
                judge.chat([{'role': 'user', 'content': 'new'}], similarity_score)
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
