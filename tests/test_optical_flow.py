import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from mustard.cli import main
from mustard.optical_flow import (
    VisibilityGate, motion_series, preprocess_dataset, sample_grid, smooth_series, track_video,
)


class OpticalFlowTests(unittest.TestCase):
    def video(self, path, frames=10):
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for index in range(frames):
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            cv2.circle(image, (5 + index, 12), 3, (255, 255, 255), -1)
            writer.write(image)
        writer.release()

    def test_grid_stays_inside_frame_and_motion_length_matches(self):
        grid = sample_grid(32, 24, 4, 3)
        self.assertEqual(grid.shape, (12, 2))
        self.assertGreaterEqual(float(grid.min()), 0)
        self.assertLessEqual(float(grid[:, 0].max()), 31)
        self.assertLessEqual(float(grid[:, 1].max()), 23)
        tracks = np.zeros((5, 2, 2), dtype=np.float32)
        tracks[:, :, 0] = np.arange(5)[:, None]
        self.assertEqual(motion_series(tracks, 'mean').tolist(), [1.0] * 5)
        self.assertEqual(len(smooth_series(np.arange(5), 4)), 5)

    def test_real_tvl1_tracking_is_aligned_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'motion.mp4'
            self.video(video, frames=4)
            tracks = track_video(video, columns=4, rows=3, max_dim=32)
            self.assertEqual(tracks.shape, (4, 12, 2))
            self.assertTrue(np.isfinite(tracks).all())
            self.assertGreaterEqual(float(tracks.min()), 0)
            self.assertLessEqual(float(tracks[..., 0].max()), 31)
            self.assertLessEqual(float(tracks[..., 1].max()), 23)

    def test_cli_dispatches_preprocessing_without_pipeline_outputs(self):
        arguments = ['mustard', 'preprocess-optical-flow', '--raw-root', '/raw',
                     '--tracks-root', '/tracks', '--trajectory-root', '/trajectory',
                     '--plots-root', '/plots']
        with patch('sys.argv', arguments), patch('mustard.cli.preprocess_dataset', return_value={'videos': 2}) as run:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main()
        self.assertEqual(json.loads(output.getvalue()), {'videos': 2})
        run.assert_called_once()

    def test_dataset_preprocessing_mirrors_indexer_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            train = raw / 'ball' / 'train' / '0000.mp4'
            test = raw / 'ball' / 'test' / 'norm' / '0001.mp4'
            self.video(train)
            self.video(test)

            def fake_tracks(video, columns, rows, max_dim):
                tracks = np.zeros((10, 4, 2), dtype=np.float32)
                tracks[:, :, 0] = np.arange(10)[:, None]
                tracks[:, :, 1] = np.arange(4)
                return tracks

            with patch('mustard.optical_flow.track_video', side_effect=fake_tracks):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = preprocess_dataset(raw, root / 'tracks', root / 'trajectory', root / 'plots',
                                                point_count=2, pool_size=4, force_count=1)
            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual(result['videos'], 2)
            for relative in (Path('ball/train/0000'), Path('ball/test/norm/0001')):
                self.assertTrue((root / 'tracks' / relative).with_suffix('.npy').is_file())
                self.assertTrue((root / 'trajectory' / relative).with_suffix('.mp4').is_file())
                self.assertTrue((root / 'plots' / relative).with_name(relative.name + '_tracks.png').is_file())

    def test_visibility_gate_debounces_motion(self):
        gate = VisibilityGate(2.0, 1.0, warmup=2, cooldown=2, ema_alpha=0)
        self.assertEqual(
            [gate.update(value) for value in (0, 2, 2, 0, 0)],
            [False, False, True, True, False],
        )

    def test_preprocessing_settings_require_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            self.video(raw / 'ball' / 'train' / '0000.mp4', frames=3)

            def fake_tracks(video, columns, rows, max_dim):
                tracks = np.zeros((3, 4, 2), dtype=np.float32)
                tracks[:, :, 0] = np.arange(3)[:, None]
                return tracks

            with patch('mustard.optical_flow.track_video', side_effect=fake_tracks):
                preprocess_dataset(
                    raw, root / 'tracks', root / 'trajectory', root / 'plots',
                    point_count=2, pool_size=4, force_count=1,
                )
            settings = json.loads((root / 'tracks' / 'preprocessing.json').read_text())
            self.assertEqual(settings['options']['grid_columns'], 80)
            with patch('mustard.optical_flow.track_video') as track:
                preprocess_dataset(
                    raw, root / 'tracks', root / 'trajectory', root / 'plots',
                    point_count=2, pool_size=4, force_count=1,
                )
            track.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                preprocess_dataset(
                    raw, root / 'tracks', root / 'trajectory', root / 'plots',
                    columns=40, point_count=2, pool_size=4, force_count=1,
                )

    def test_preprocessing_rejects_colliding_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            self.video(raw / 'ball' / 'train' / '0000.mp4')
            with self.assertRaisesRegex(ValueError, 'roots must be distinct'):
                preprocess_dataset(raw, raw, root / 'trajectory', root / 'plots')


if __name__ == '__main__':
    unittest.main()
