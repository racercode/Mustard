"""Select once, then extract aligned raw and trajectory frames."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ._selection import select_keyframes_priority

DEFAULTS = dict(method="keyframe", frame_num=8, agg="mean", smooth_win=8, min_dist=4,
                num_peaks=3, num_valleys=2, shoulder_rel=0.5, min_prom_frac=0.02,
                priority=["start_end", "peaks", "valleys", "shoulders", "midpoints"])


def select(tracks, config=None):
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[2] != 2 or tracks.shape[1] == 0:
        raise ValueError("tracks must have shape (frames, points, 2), with at least one point")
    if not np.isfinite(tracks).all():
        raise ValueError("tracks contain non-finite coordinates")
    options = {**DEFAULTS, **(config or {})}
    if set(options) != set(DEFAULTS):
        raise ValueError(f"unknown selection options: {set(options) - set(DEFAULTS)}")
    for name in ("frame_num", "smooth_win", "min_dist", "num_peaks", "num_valleys"):
        if not isinstance(options[name], int) or isinstance(options[name], bool) or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if options['agg'] not in {"mean", "median", "sum"}:
        raise ValueError("agg must be mean, median, or sum")
    if not 0 < options['shoulder_rel'] < 1 or options['min_prom_frac'] < 0:
        raise ValueError("invalid shoulder_rel or min_prom_frac")
    priority = options['priority']
    allowed = {"start_end", "peaks", "valleys", "shoulders", "midpoints"}
    if not isinstance(priority, list) or not priority or any(p not in allowed for p in priority):
        raise ValueError("priority must be a list of supported selection stages")
    if tracks.shape[0] < options['frame_num']:
        raise ValueError("video has fewer frames than the requested keyframe budget")
    if options['method'] not in ('keyframe', 'uniform'):
        raise ValueError('method must be keyframe or uniform')
    if options['frame_num'] < 2:
        raise ValueError('at least two selected frames are required')
    if options['method'] == 'uniform':
        step = max((tracks.shape[0] - 1) // (options['frame_num'] - 1), 1)
        indices = np.asarray([i * step for i in range(options['frame_num'])])
    else:
        algorithm_options = {k: v for k, v in options.items() if k != 'method'}
        indices, _ = select_keyframes_priority(tracks, **algorithm_options)
    indices = [int(i) for i in indices]
    if len(indices) != options['frame_num'] or len(set(indices)) != len(indices):
        raise ValueError("selection cannot meet the keyframe budget; inspect video length and min_dist")
    return indices, options


def extract(video, indices, output, expected_frames):
    """Decode sequentially so frame indices refer to the decoded sequence."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    output.mkdir(parents=True)
    wanted = set(indices)
    count = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if count in wanted:
                if not cv2.imwrite(str(output / f'{count:06d}.png'), image):
                    raise OSError(f"failed to save frame {count} from {video}")
            count += 1
    finally:
        cap.release()
    if count != expected_frames:
        raise ValueError(f"{video}: decoded {count} frames, tracks contain {expected_frames}; inputs must be aligned")


def prepare(raw_video, trajectory_video, tracks_path, output, config=None):
    """Create a new complete sample directory; refuse to overwrite existing work."""
    raw_video, trajectory_video, tracks_path, output = map(Path, (raw_video, trajectory_video, tracks_path, output))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose a new directory")
    tracks = np.load(tracks_path, allow_pickle=False)
    indices, options = select(tracks, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        extract(raw_video, indices, tmp / 'raw', tracks.shape[0])
        extract(trajectory_video, indices, tmp / 'trajectory', tracks.shape[0])
        digest = hashlib.sha256()
        with tracks_path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        manifest = dict(indices=indices, total_frames=int(tracks.shape[0]), selection=options,
                        effective_smooth_win=(options['smooth_win'] + (options['smooth_win'] % 2 == 0)) if options['method'] == 'keyframe' else None,
                        inputs=dict(raw_video=str(raw_video.resolve()), trajectory_video=str(trajectory_video.resolve()),
                                    tracks=str(tracks_path.resolve()), tracks_sha256=digest.hexdigest()),
                        raw=[f'raw/{i:06d}.png' for i in indices],
                        trajectory=[f'trajectory/{i:06d}.png' for i in indices])
        (tmp / 'frames.json').write_text(json.dumps(manifest, indent=2) + '\n')
        tmp.rename(output)
        return manifest
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
