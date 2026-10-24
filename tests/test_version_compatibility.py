"""Regression checks for assumptions exposed by the Send-Physics-AD audit."""
import tempfile
import unittest
from pathlib import Path

from mustard.data import index_dataset
from mustard.judging import pairwise_verdict, explanation
from mustard.pipeline import fusion_response


class VersionCompatibilityTests(unittest.TestCase):
    def test_both_physad_normal_directory_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for kind in ['norm', 'anomaly_free', 'bad']:
                p = root / 'raw/ball/test' / kind / '0000.mp4'
                p.parent.mkdir(parents=True)
                p.touch()
            records = index_dataset(root / 'raw', root / 'trajectory', root / 'tracks', root / 'plots')
            self.assertEqual({r['anomaly']: r['label'] for r in records},
                             {'norm': 0, 'anomaly_free': 0, 'bad': 1})

    def test_ambiguous_pairwise_verdict_is_rejected(self):
        with self.assertRaises(ValueError):
            pairwise_verdict('Possibly [[B]], final [[A]]')
        self.assertEqual(pairwise_verdict('Evidence favors A. [[A]]'), 'A')

    def test_fusion_requires_final_field_not_just_score(self):
        with self.assertRaisesRegex(ValueError, 'missing Final Explanation'):
            fusion_response('Reasoning: technical\nAnomaly Score: 0.2')
        with self.assertRaisesRegex(ValueError, 'missing Final Explanation'):
            explanation('Reasoning: mentions Final Explanation: inside another field')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            explanation('Final Explanation: first\nFinal Explanation: second')
        fusion_response('Reasoning: technical\nAnomaly Score: 0.2\nFinal Explanation: readable')


if __name__ == '__main__':
    unittest.main()
