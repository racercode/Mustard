import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mustard.keyframes import prepare, select


class KeyframeTests(unittest.TestCase):
    def video(self, path, frames, offset=0):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for i in range(frames):
            writer.write(np.full((24, 32, 3), (i + offset) % 255, dtype=np.uint8))
        writer.release()

    def test_aligned_extraction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.video(root / 'raw.avi', 60)
            self.video(root / 'trajectory.avi', 60, offset=50)
            np.save(root / 'tracks.npy', np.zeros((60, 2, 2), dtype=np.float32))
            output = root / 'output'
            result = prepare(root / 'raw.avi', root / 'trajectory.avi', root / 'tracks.npy', output)
            self.assertEqual(len(result['indices']), 8)
            self.assertEqual(result['indices'][0], 0)
            self.assertEqual(result['indices'][-1], 59)
            self.assertEqual(result['effective_smooth_win'], 9)
            self.assertEqual(len(list((output / 'raw').glob('*.png'))), 8)
            self.assertEqual(len(list((output / 'trajectory').glob('*.png'))), 8)
            for i in result['indices']:
                raw = cv2.imread(str(output / 'raw' / f'{i:06d}.png'))
                trajectory = cv2.imread(str(output / 'trajectory' / f'{i:06d}.png'))
                self.assertLess(abs(float(raw.mean()) - i), 3)
                self.assertLess(abs(float(trajectory.mean()) - (i + 50)), 3)
            with self.assertRaises(FileExistsError):
                prepare(root / 'raw.avi', root / 'trajectory.avi', root / 'tracks.npy', output)

    def test_misaligned_video_leaves_no_completed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.video(root / 'raw.avi', 60)
            self.video(root / 'trajectory.avi', 59)
            np.save(root / 'tracks.npy', np.zeros((60, 1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'inputs must be aligned'):
                prepare(root / 'raw.avi', root / 'trajectory.avi', root / 'tracks.npy', root / 'output')
            self.assertFalse((root / 'output').exists())
            self.assertFalse(list(root.glob('.output-*')))

    def test_invalid_tracks_and_unknown_config(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            select(np.full((60, 1, 2), np.nan))
        with self.assertRaisesRegex(ValueError, 'unknown selection'):
            select(np.zeros((60, 1, 2)), {'typo': True})
        with self.assertRaisesRegex(ValueError, 'fewer frames'):
            select(np.zeros((3, 1, 2)))


if __name__ == '__main__':
    unittest.main()
