"""TV-L1 optical-flow preprocessing for MUSTARD input artifacts."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


_SETTINGS_FILE = "preprocessing.json"


def sample_grid(width, height, columns=80, rows=50):
    if width < 2 or height < 2:
        raise ValueError('video frames must be at least 2x2 pixels')
    if columns < 1 or rows < 1:
        raise ValueError('grid columns and rows must be positive')
    xs = np.linspace(0, width - 1, columns, dtype=np.float32)
    ys = np.linspace(0, height - 1, rows, dtype=np.float32)
    return np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)


def _tvl1():
    optflow = getattr(cv2, 'optflow', None)
    for name in ('DualTVL1OpticalFlow_create', 'createOptFlow_DualTVL1'):
        factory = getattr(optflow, name, None)
        if factory is not None:
            return factory()
    raise RuntimeError(
        'TV-L1 optical flow requires opencv-contrib-python-headless; '
        "install MUSTARD with `pip install -e '.[preprocessing]'`"
    )


def _video_info(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {video}')
    ok, frame = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not ok or frame is None:
        raise ValueError(f'cannot decode first frame: {video}')
    height, width = frame.shape[:2]
    return width, height, fps if fps > 0 else 24.0


def track_video(video, columns=80, rows=50, max_dim=640):
    """Track a full-frame grid and return float32 coordinates shaped (T,K,2)."""
    if max_dim < 2:
        raise ValueError('max_dim must be at least 2')
    width, height, _ = _video_info(video)
    scale = min(1.0, max_dim / max(width, height))
    work_width = max(2, int(width * scale + 0.5))
    work_height = max(2, int(height * scale + 0.5))
    seeds = sample_grid(width, height, columns, rows) * scale

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {video}')
    flow_engine = _tvl1()
    tracks = [seeds]
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f'cannot decode first frame: {video}')
        if scale < 1.0:
            frame = cv2.resize(frame, (work_width, work_height), interpolation=cv2.INTER_AREA)
        previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (work_width, work_height), interpolation=cv2.INTER_AREA)
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = flow_engine.calc(previous, current, None)
            points = tracks[-1]
            px = np.clip(points[:, 0], 0, work_width - 1)
            py = np.clip(points[:, 1], 0, work_height - 1)
            x0 = np.floor(px).astype(np.int32)
            y0 = np.floor(py).astype(np.int32)
            x1 = np.minimum(x0 + 1, work_width - 1)
            y1 = np.minimum(y0 + 1, work_height - 1)
            wx = (px - x0)[:, None]
            wy = (py - y0)[:, None]
            sampled = ((1 - wx) * ((1 - wy) * flow[y0, x0] + wy * flow[y1, x0])
                       + wx * ((1 - wy) * flow[y0, x1] + wy * flow[y1, x1]))
            next_points = points + sampled
            next_points[:, 0] = np.clip(next_points[:, 0], 0, work_width - 1)
            next_points[:, 1] = np.clip(next_points[:, 1], 0, work_height - 1)
            tracks.append(next_points.astype(np.float32))
            previous = current
    finally:
        cap.release()
    result = np.stack(tracks).astype(np.float32)
    result /= scale
    return result


def motion_series(tracks, aggregation='median'):
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[0] < 1 or tracks.shape[-1] != 2 or tracks.shape[1] < 1:
        raise ValueError('tracks must have shape (T,K,2)')
    if aggregation not in ('median', 'mean', 'sum'):
        raise ValueError('aggregation must be median, mean, or sum')
    steps = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    if aggregation == 'median':
        values = np.median(steps, axis=1)
    elif aggregation == 'mean':
        values = np.mean(steps, axis=1)
    else:
        values = np.sum(steps, axis=1)
    first = values[0] if values.size else 0.0
    return np.concatenate(([first], values)).astype(np.float32)


def smooth_series(values, window=7):
    if window < 1:
        raise ValueError('smooth window must be positive')
    window += window % 2 == 0
    if window == 1:
        return np.asarray(values, dtype=np.float32).copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode='valid').astype(np.float32)


class VisibilityGate:
    """Apply hysteresis and debounce to per-point motion visibility."""

    def __init__(self, on_threshold, off_threshold, warmup=2, cooldown=2, ema_alpha=0.25):
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.warmup = int(warmup)
        self.cooldown = int(cooldown)
        self.ema_alpha = float(ema_alpha)
        self.visible = False
        self.on_count = 0
        self.off_count = 0
        self.ema = None

    def update(self, step_length):
        value = 0.0 if step_length is None or not np.isfinite(step_length) else float(step_length)
        if self.ema_alpha > 0:
            self.ema = value if self.ema is None else (1 - self.ema_alpha) * self.ema + self.ema_alpha * value
            value = self.ema
        if not self.visible:
            self.on_count = self.on_count + 1 if value >= self.on_threshold else 0
            if self.on_count >= self.warmup:
                self.visible = True
                self.off_count = 0
        else:
            self.off_count = self.off_count + 1 if value < self.off_threshold else 0
            if self.off_count >= self.cooldown:
                self.visible = False
                self.on_count = 0
                self.ema = None
        return self.visible


def write_motion_plot(tracks, output, aggregation='median', smooth_window=7):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "motion plots require matplotlib; install MUSTARD with `pip install -e '.[preprocessing]'`"
        ) from exc
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = smooth_series(motion_series(tracks, aggregation), smooth_window)
    figure, axes = plt.subplots(figsize=(12, 4))
    axes.plot(np.arange(len(values)), values, linewidth=2.2)
    axes.set_xlabel('frame')
    axes.set_ylabel('overall motion')
    if values.size:
        for label, index, offset, align in (
                ('MAX', int(np.argmax(values)), 16, 'bottom'),
                ('MIN', int(np.argmin(values)), -18, 'top')):
            axes.annotate(f'{label} {values[index]:.3f}', xy=(index, values[index]),
                          xytext=(0, offset), textcoords='offset points', ha='center', va=align,
                          fontsize=18, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9),
                          arrowprops=dict(arrowstyle='-|>', linewidth=1.0))
    figure.subplots_adjust(right=0.98, left=0.08, top=0.92, bottom=0.12)
    with tempfile.NamedTemporaryFile(suffix='.png', dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        figure.savefig(temporary_path, dpi=160, bbox_inches='tight')
        os.replace(temporary_path, output)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def _farthest(points, count, seeds, width, height):
    if count <= 0 or len(points) == 0:
        return np.empty(0, dtype=np.int64)
    normalized = points.astype(np.float32) / np.asarray([width, height], dtype=np.float32)
    if len(seeds):
        seed_points = seeds.astype(np.float32) / np.asarray([width, height], dtype=np.float32)
        distances = np.min(np.sum((normalized[:, None] - seed_points[None]) ** 2, axis=2), axis=1)
    else:
        distances = np.sum((normalized - np.asarray([0.5, 0.5], dtype=np.float32)) ** 2, axis=1)
    selected = []
    for _ in range(min(count, len(points))):
        index = int(np.argmax(distances))
        if distances[index] < 0:
            break
        selected.append(index)
        distances = np.minimum(distances, np.sum((normalized - normalized[index]) ** 2, axis=1))
        distances[selected] = -1
    return np.asarray(selected, dtype=np.int64)


def select_overlay_points(tracks, width, height, count=50, pool_size=500, force_count=10):
    if count < 1 or pool_size < 1 or force_count < 0:
        raise ValueError('overlay point counts must be positive')
    speeds = np.linalg.norm(np.diff(tracks, axis=0), axis=2).mean(axis=0)
    pool = np.argsort(-speeds)[:min(pool_size, tracks.shape[1])]
    forced = pool[:min(force_count, count, len(pool))]
    candidates = pool[len(forced):]
    relative = _farthest(tracks[0, candidates], count - len(forced), tracks[0, forced], width, height)
    return np.concatenate((forced, candidates[relative])).astype(np.int64)


def render_trajectory(video, tracks, output, reference_fraction, point_count=50, pool_size=500,
                      force_count=10, trail_length=10, arrow_max_fraction=0.08,
                      min_visible_step_fraction=0.0005, visibility_hysteresis=0.25,
                      visibility_warmup=2, visibility_cooldown=2, visibility_ema=0.25):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = _video_info(video)
    selected = select_overlay_points(tracks, width, height, point_count, pool_size, force_count)
    reference_pixels = max(reference_fraction * width, 1e-6)
    arrow_limit = arrow_max_fraction * width
    arrow_scale = arrow_limit / reference_pixels
    off_threshold = min_visible_step_fraction * reference_pixels
    on_threshold = off_threshold * (1 + visibility_hysteresis)
    cap = cv2.VideoCapture(str(video))
    with tempfile.NamedTemporaryFile(suffix='.mp4', dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        temporary_path.unlink(missing_ok=True)
        raise OSError(f'cannot create trajectory video: {output}')
    trails = [[] for _ in selected]
    gates = [VisibilityGate(on_threshold, off_threshold, visibility_warmup,
                            visibility_cooldown, visibility_ema) for _ in selected]
    frames = 0
    completed = False
    try:
        for frame_index in range(tracks.shape[0]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'{video}: decoded {frames} frames, tracks contain {tracks.shape[0]}')
            if frame_index > 0:
                previous = tracks[frame_index - 1, selected]
                current = tracks[frame_index, selected]
                for index, (start, end) in enumerate(zip(previous, current)):
                    step = end - start
                    length = float(np.linalg.norm(step))
                    if not gates[index].update(length):
                        continue
                    if trail_length:
                        trails[index].append(end.copy())
                        trails[index] = trails[index][-trail_length:]
                    if len(trails[index]) > 1:
                        cv2.polylines(frame, [np.rint(trails[index]).astype(np.int32)], False,
                                      (0, 140, 255), 2, cv2.LINE_AA)
                    direction = step / length if length > 1e-6 else np.zeros(2, dtype=np.float32)
                    tip = end + direction * min(length * arrow_scale, arrow_limit)
                    origin = tuple(np.rint(end).astype(int))
                    target = tuple(np.rint(tip).astype(int))
                    cv2.arrowedLine(frame, origin, target, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.45)
                    cv2.arrowedLine(frame, origin, target, (0, 140, 255), 3, cv2.LINE_AA, tipLength=0.45)
            writer.write(frame)
            frames += 1
        ok, _ = cap.read()
        if ok:
            raise ValueError(f'{video}: video contains more frames than its tracks')
        completed = True
    finally:
        cap.release()
        writer.release()
        if not completed:
            temporary_path.unlink(missing_ok=True)
    os.replace(temporary_path, output)


def _dataset_videos(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'raw dataset root does not exist: {root}')
    videos = []
    for obj in sorted(path for path in root.iterdir() if path.is_dir()):
        videos.extend(sorted((obj / 'train').glob('*.mp4')))
        videos.extend(sorted((obj / 'test').glob('*/*.mp4')))
    if not videos:
        raise ValueError('no videos found; expected {object}/train/*.mp4 or {object}/test/{type}/*.mp4')
    return videos


def _save_tracks(path, tracks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.npy', dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.save(temporary_path, tracks.astype(np.float32), allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_tracks(path):
    tracks = np.load(path, allow_pickle=False, mmap_mode="r")
    if (tracks.ndim != 3 or tracks.shape[0] < 2 or tracks.shape[1] < 1
            or tracks.shape[-1] != 2 or not np.isfinite(tracks).all()):
        raise ValueError(f"invalid tracks: {path}")
    return tracks


def preprocess_dataset(raw_root, tracks_root, trajectory_root, plots_root, *, columns=80, rows=50,
                       max_dim=640, aggregation='median', smooth_window=7, point_count=50,
                       pool_size=500, force_count=10, trail_length=10, arrow_max_fraction=0.08,
                       reference_percentile=95, min_visible_step_fraction=0.0005,
                       visibility_hysteresis=0.25, visibility_warmup=2,
                       visibility_cooldown=2, visibility_ema=0.25, overwrite=False):
    """Generate only the optical-flow artifacts consumed by MUSTARD."""
    if columns < 1 or rows < 1 or max_dim < 2:
        raise ValueError('grid dimensions must be positive and max_dim must be at least 2')
    if smooth_window < 1 or point_count < 1 or pool_size < 1 or force_count < 0 or trail_length < 0:
        raise ValueError('smoothing and overlay counts are invalid')
    if arrow_max_fraction <= 0 or not 0 <= reference_percentile <= 100:
        raise ValueError('arrow fraction and reference percentile are invalid')
    if (min_visible_step_fraction < 0 or visibility_hysteresis < 0
            or visibility_warmup < 0 or visibility_cooldown < 0
            or not 0 <= visibility_ema <= 1):
        raise ValueError('trajectory visibility settings are invalid')

    raw_root = Path(raw_root).resolve()
    tracks_root, trajectory_root, plots_root = map(
        lambda output: Path(output).resolve(), (tracks_root, trajectory_root, plots_root))
    if len({raw_root, tracks_root, trajectory_root, plots_root}) != 4:
        raise ValueError('raw, tracks, trajectory, and plots roots must be distinct')
    videos = _dataset_videos(raw_root)

    options = dict(grid_columns=columns, grid_rows=rows, max_dim=max_dim, aggregation=aggregation,
                   smooth_window=smooth_window, overlay_points=point_count, top_pool=pool_size,
                   force_top=force_count, trail_length=trail_length,
                   arrow_max_fraction=arrow_max_fraction, reference_percentile=reference_percentile,
                   min_visible_step_fraction=min_visible_step_fraction,
                   visibility_hysteresis=visibility_hysteresis,
                   visibility_warmup=visibility_warmup, visibility_cooldown=visibility_cooldown,
                   visibility_ema=visibility_ema)
    settings = dict(schema_version=1, raw_root=str(raw_root), options=options)
    settings_path = tracks_root / _SETTINGS_FILE
    if settings_path.exists():
        try:
            previous_settings = json.loads(settings_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'invalid preprocessing settings: {settings_path}') from exc
        if previous_settings != settings and not overwrite:
            raise ValueError('preprocessing settings changed; use --overwrite to rebuild outputs')
    elif tracks_root.exists() and any(tracks_root.rglob('*.npy')) and not overwrite:
        raise ValueError('existing tracks have no preprocessing settings; use --overwrite to rebuild them')
    _write_json(settings_path, settings)

    items = []
    for video in videos:
        relative = video.relative_to(raw_root)
        tracks_path = tracks_root / relative.with_suffix('.npy')
        generated = overwrite or not tracks_path.exists()
        if generated:
            print(f'optical-flow {relative}', file=sys.stderr, flush=True)
            _save_tracks(tracks_path, track_video(video, columns, rows, max_dim))
        _load_tracks(tracks_path)
        items.append((video, relative, tracks_path, generated))

    references = {}
    for obj in sorted({relative.parts[0] for _, relative, _, _ in items}):
        fractions = []
        for video, relative, tracks_path, _ in items:
            if relative.parts[0] == obj and relative.parts[1] == 'train':
                tracks = _load_tracks(tracks_path)
                width, _, _ = _video_info(video)
                fractions.append(float(np.linalg.norm(np.diff(tracks, axis=0), axis=2).max()) / width)
        if not fractions:
            raise ValueError(f'{obj}: no usable training tracks for trajectory scaling')
        references[obj] = float(np.percentile(fractions, reference_percentile))

    for video, relative, tracks_path, generated in items:
        tracks = _load_tracks(tracks_path)
        trajectory_path = trajectory_root / relative
        plot_path = plots_root / relative.with_name(relative.stem + '_tracks.png')
        if generated or not trajectory_path.exists():
            print(f'trajectory {relative}', file=sys.stderr, flush=True)
            render_trajectory(
                video, tracks, trajectory_path, references[relative.parts[0]],
                point_count, pool_size, force_count, trail_length, arrow_max_fraction,
                min_visible_step_fraction, visibility_hysteresis, visibility_warmup,
                visibility_cooldown, visibility_ema)
        if generated or not plot_path.exists():
            print(f'motion-plot {relative}', file=sys.stderr, flush=True)
            write_motion_plot(tracks, plot_path, aggregation, smooth_window)

    return dict(videos=len(items), tracks_root=str(tracks_root), trajectory_root=str(trajectory_root),
                plots_root=str(plots_root), settings=str(settings_path),
                reference_fraction=references, options=options)
