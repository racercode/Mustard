"""Inspect external preprocessing without running a model."""
import cv2
import numpy as np


def validate_inputs(records, decode=False):
    errors = []
    samples = []
    for sample in records:
        identity = '/'.join(str(sample.get(key, '')) for key in ('object', 'split', 'anomaly', 'video'))
        try:
            tracks = np.load(sample['tracks'], mmap_mode='r', allow_pickle=False)
            if tracks.ndim != 3 or tracks.shape[2] != 2 or tracks.shape[0] < 8 or tracks.shape[1] < 1:
                raise ValueError('tracks must have shape (T>=8,K>=1,2)')
            if not np.issubdtype(tracks.dtype, np.number):
                raise ValueError('tracks must be numeric')
            with open(sample['plot'], 'rb') as file:
                if file.read(8) != b'\x89PNG\r\n\x1a\n':
                    raise ValueError('plot must be a PNG image')
            counts = {}
            for kind in ('raw_video', 'trajectory_video'):
                cap = cv2.VideoCapture(sample[kind])
                try:
                    if not cap.isOpened():
                        raise ValueError(f'cannot open {kind}')
                    if decode:
                        count = 0
                        while cap.read()[0]:
                            count += 1
                    else:
                        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
                if count != tracks.shape[0]:
                    raise ValueError(f'{kind} has {count} frames, tracks have {tracks.shape[0]}')
                counts[kind] = count
            samples.append(dict(sample=identity, track_shape=list(tracks.shape), frame_counts=counts))
        except (ValueError, OSError) as error:
            errors.append(dict(sample=identity, error=str(error)))
    return dict(valid=not errors, count=len(records), decoded=decode, samples=samples, errors=errors)
