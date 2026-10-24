"""Selection algorithm extracted from cswagger/OpticalFlow.

See docs/SOURCES.md. The SciPy algorithm is retained; the silent fallback is removed.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np

def _find_peaks_with_prominence(y: np.ndarray, distance: int = 1, prominence: float = 0.0):
    # Require the original SciPy path; never silently change the selection algorithm.
    from scipy.signal import find_peaks, peak_prominences
    peaks, props = find_peaks(y, distance=max(1, distance), prominence=prominence)
    prom = props.get("prominences")
    if prom is None and peaks.size:
        prom = peak_prominences(y, peaks)[0]
    return peaks, prom if prom is not None else np.array([], dtype=float)


def _find_valleys_with_prominence(y: np.ndarray, distance: int = 1, prominence: float = 0.0):
    v, prom = _find_peaks_with_prominence(-y, distance=distance, prominence=prominence)
    return v, prom

def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Odd-length box filter; reflect edges."""
    if win <= 1:
        return x.astype(np.float32, copy=True)
    win = int(win + (win % 2 == 0))  # make odd
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, ker, mode="valid").astype(np.float32)

def motion_series_from_tracks(tracks: np.ndarray, agg: str = "median") -> np.ndarray:
    """
    tracks: (T, K, 2)
    returns m: (T,) motion per frame (first frame duplicated so length matches T)
    """
    disp = np.diff(tracks, axis=0)                 # (T-1, K, 2)
    step = np.linalg.norm(disp, axis=2)            # (T-1, K)

    if agg == "sum":
        s = step.sum(axis=1)
    elif agg == "mean":
        s = step.mean(axis=1)
    else:  # 'median'
        s = np.median(step, axis=1)

    # Duplicate first value to get length T
    first = s[0] if s.size else 0.0
    return np.concatenate([[first], s]).astype(np.float32)

def _prom_thresh_from_frac(y: np.ndarray, frac: float) -> float:
    rng = float(y.max() - y.min())
    return max(0.0, float(frac) * rng)

def _enforce_min_dist(sel: List[int], cand: int, min_dist: int) -> bool:
    return all(abs(cand - j) >= min_dist for j in sel)

def _gap_midpoints_fill(sel_sorted: List[int], T: int, need_total: int, min_dist: int) -> List[int]:
    """Fill by largest gaps with a slight bias away from exact midpoints (for flat areas)."""
    sel = list(sel_sorted)
    while len(sel) < need_total:
        sel.sort()
        if len(sel) < 2:
            break
        gaps = [(sel[i], sel[i + 1], sel[i + 1] - sel[i]) for i in range(len(sel) - 1)]
        if not gaps:
            break
        L, R, _ = max(gaps, key=lambda g: g[2])
        # three candidates near the middle
        cands = [L + max(1, (R - L) // 3), L + max(1, 2 * (R - L) // 3), (L + R) // 2]
        for c in cands:
            c = int(c)
            if 0 <= c < T and _enforce_min_dist(sel, c, min_dist):
                sel.append(c)
                break
        else:
            break
    return sorted(sel)

def _nearest_valley(y: np.ndarray, i_peak: int, left: bool) -> Optional[float]:
    """Find the nearest valley value to the left or right of i_peak; None if not found."""
    if left:
        rng = range(i_peak - 1, 0, -1)
    else:
        rng = range(i_peak + 1, len(y) - 1)
    for i in rng:
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            return float(y[i])
    return None

def _shoulders_for_peak(y: np.ndarray, i_peak: int, rel: float = 0.6) -> Tuple[Optional[int], Optional[int]]:
    """
    Shoulders at a level between the local baseline and the peak.
    Baseline is min of nearest left/right valley (if exists), else side-wise min.
    """
    T = len(y)
    if i_peak <= 0 or i_peak >= T - 1:
        return None, None

    Lval = _nearest_valley(y, i_peak, left=True)
    Rval = _nearest_valley(y, i_peak, left=False)
    base_left = y[:i_peak].min() if i_peak > 0 else y.min()
    base_right = y[i_peak + 1 :].min() if i_peak + 1 < T else y.min()
    base = min(Lval if Lval is not None else base_left,
               Rval if Rval is not None else base_right)

    peak_val = float(y[i_peak])
    if peak_val <= base:
        return None, None

    lvl = base + float(rel) * (peak_val - base)

    # left crossing upward through lvl
    L = None
    for t in range(i_peak - 1, -1, -1):
        if y[t] <= lvl and y[min(t + 1, i_peak)] > lvl:
            L = t + 1
            break
    # right crossing downward through lvl
    R = None
    for t in range(i_peak + 1, T - 1):
        if y[t] >= lvl and y[t + 1] < lvl:
            R = t + 1
            break
    return L, R

def select_keyframes_priority(
    tracks: np.ndarray,
    frame_num: int,
    agg: str,
    smooth_win: int,
    min_dist: int,
    num_peaks: int,
    num_valleys: int,
    shoulder_rel: float,
    min_prom_frac: float,
    priority: List[str],
):
    T = tracks.shape[0]
    m_raw = motion_series_from_tracks(tracks, agg=agg)
    m_sm  = moving_average_1d(m_raw, smooth_win)

    # Candidate pools (ranked by prominence)
    prom_thresh = _prom_thresh_from_frac(m_sm, min_prom_frac)
    peaks, p_prom = _find_peaks_with_prominence(m_sm, distance=max(1, min_dist), prominence=prom_thresh)
    valleys, v_prom = _find_valleys_with_prominence(m_sm, distance=max(1, min_dist), prominence=prom_thresh)

    peaks_ranked   = peaks[np.argsort(-p_prom)] if p_prom.size else np.array([], int)
    valleys_ranked = valleys[np.argsort(-v_prom)] if v_prom.size else np.array([], int)

    peak_prom_map: Dict[int, float] = {int(i): float(p) for i, p in zip(peaks.tolist(), p_prom.tolist())}
    val_prom_map:  Dict[int, float] = {int(i): float(p) for i, p in zip(valleys.tolist(), v_prom.tolist())}

    chosen: List[int] = []
    chosen_set = set()
    kept_peaks: List[int] = []
    kept_vals: List[int] = []
    kept_shoulders: List[int] = []
    shoulders_all: List[int] = []

    def _try_add(idx: int) -> bool:
        if 0 <= idx < T and idx not in chosen_set and _enforce_min_dist(chosen, idx, min_dist):
            chosen.append(idx)
            chosen_set.add(idx)
            return True
        return False

    # Priority passes
    for token in [t.strip().lower() for t in priority]:
        if len(chosen) >= frame_num:
            break

        if token == "start_end":
            for i in (0, T - 1):
                _try_add(i)

        elif token == "peaks":
            for i in peaks_ranked.tolist():
                if len(kept_peaks) >= num_peaks:
                    break
                if _try_add(int(i)):
                    kept_peaks.append(int(i))

        elif token == "valleys":
            for i in valleys_ranked.tolist():
                if len(kept_vals) >= num_valleys:
                    break
                if _try_add(int(i)):
                    kept_vals.append(int(i))

        elif token == "shoulders":
            for p in kept_peaks:
                L, R = _shoulders_for_peak(m_sm, p, rel=shoulder_rel)
                for s in (L, R):
                    if s is None:
                        continue
                    s = int(s)
                    shoulders_all.append(s)
                    if _try_add(s):
                        kept_shoulders.append(s)

        elif token == "midpoints":
            if len(chosen) < frame_num:
                chosen[:] = _gap_midpoints_fill(sorted(chosen), T, frame_num, min_dist)

    chosen.sort()

    # -------------------- PRUNE if over budget --------------------
    if len(chosen) > frame_num:
        keep = set(chosen)

        # classify
        start_end = {i for i in (0, T - 1) if i in keep}
        peak_set  = {i for i in kept_peaks if i in keep}
        val_set   = {i for i in kept_vals if i in keep}
        sh_set    = {i for i in kept_shoulders if i in keep}
        mid_set   = keep - start_end - peak_set - val_set - sh_set

        # queues (lowest importance removed first)
        q_mid = [(0.0, i) for i in sorted(mid_set)]
        q_sh  = [(float(m_sm[i]), i) for i in sorted(sh_set)]
        q_val = [(float(val_prom_map.get(i, 0.0)), i) for i in sorted(val_set)]
        q_pk  = [(float(peak_prom_map.get(i, 0.0)), i) for i in sorted(peak_set)]

        for q in (q_mid, q_sh, q_val, q_pk):
            q.sort(key=lambda t: t[0])

        def drop(queue):
            nonlocal keep
            while len(keep) > frame_num and queue:
                _, idx = queue.pop(0)
                keep.discard(idx)

        drop(q_mid)
        drop(q_sh)
        drop(q_val)
        drop(q_pk)

        # last resort: allow trimming start/end if still over
        if len(keep) > frame_num:
            for i in sorted(start_end):
                if len(keep) <= frame_num:
                    break
                keep.discard(i)

        chosen = sorted(keep)

    # -------------------- FILL if under budget --------------------
    if len(chosen) < frame_num and len(chosen) >= 2:
        chosen = _gap_midpoints_fill(chosen, T, frame_num, min_dist)

    info = dict(
        motion_raw=m_raw.tolist(),
        motion_smooth=m_sm.tolist(),
        peaks_all=peaks_ranked.tolist(),
        valleys_all=valleys_ranked.tolist(),
        peaks=kept_peaks,
        valleys=kept_vals,
        shoulders=kept_shoulders,
        selected=chosen,
        shoulders_all=list(dict.fromkeys(shoulders_all)),
        shoulders_rejected=[i for i in shoulders_all if i not in set(kept_shoulders)],
        params=dict(
            frame_num=frame_num, agg=agg, smooth_win=smooth_win, min_dist=min_dist,
            num_peaks=num_peaks, num_valleys=num_valleys,
            shoulder_rel=shoulder_rel, min_prom_frac=min_prom_frac,
            priority=priority,
        ),
    )
    return np.asarray(chosen, dtype=int), info
