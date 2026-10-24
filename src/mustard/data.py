"""Explicit dataset manifests; no implicit labels during inference."""
import json
from pathlib import Path


def index_dataset(raw_root, trajectory_root, tracks_root, plots_root, normal_names=('norm', 'normal', 'good', 'anomaly_free')):
    raw_root, trajectory_root, tracks_root, plots_root = map(Path, (raw_root, trajectory_root, tracks_root, plots_root))
    if not raw_root.is_dir():
        raise ValueError(f'raw dataset root does not exist: {raw_root}')
    records = []
    for obj in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        files = list((obj / 'train').glob('*.mp4')) + list((obj / 'test').glob('*/*.mp4'))
        for video in sorted(files):
            if 'demo' in video.name.lower():
                continue
            rel = video.relative_to(raw_root)
            split = rel.parts[1]
            anomaly = rel.parts[2] if split == 'test' else None
            tracks = tracks_root / rel.with_suffix('.npy')
            if not tracks.exists():
                tracks = tracks.with_name(tracks.stem + '_tracks.npy')
            plot = plots_root / rel.with_name(video.stem + '_tracks.png')
            records.append(dict(object=obj.name, split=split, video=video.stem, anomaly=anomaly,
                                label=0 if split == 'train' or anomaly in normal_names else 1,
                                raw_video=str(video.resolve()),
                                trajectory_video=str((trajectory_root / rel).resolve()),
                                tracks=str(tracks.resolve()), plot=str(plot.resolve())))
    if not records:
        raise ValueError('no videos found; expected {object}/train/*.mp4 or {object}/test/{type}/*.mp4')
    return records


def load_manifest(path):
    path = Path(path).resolve()
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('manifest must be a nonempty list of samples')
    keys = set()
    for record in records:
        for name in ('object', 'video', 'split'):
            value = record[name]
            if not isinstance(value, str) or value in ('', '.', '..') or '/' in value or '\\' in value:
                raise ValueError(f'invalid {name}: {value!r}')
        if record['split'] not in ('train', 'test'):
            raise ValueError('split must be train or test')
        if record.get('label') not in (0, 1):
            raise ValueError('every manifest sample needs an explicit binary label')
        if record['split'] == 'train' and record['label'] != 0:
            raise ValueError('training manifest contains an anomalous sample')
        key = (record['object'], record['split'], record.get('anomaly'), record['video'])
        if key in keys:
            raise ValueError(f'duplicate sample: {key}')
        keys.add(key)
        for name in ('raw_video', 'trajectory_video', 'tracks', 'plot'):
            value = Path(record[name])
            value = value if value.is_absolute() else path.parent / value
            record[name] = str(value.resolve())
            if not value.is_file():
                raise ValueError(f'missing {name}: {value}')
    return records
