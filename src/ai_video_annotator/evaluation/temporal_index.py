"""Sparse temporal geometry for long recordings without quadratic overlap matrices."""

from math import isfinite

import numpy as np


class SuppressionIndex:
    def __init__(self, windows, threshold=0.3):
        if not windows or len({w.video_id for w in windows}) != 1:
            raise ValueError("Provide windows from one recording")
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid NMS threshold")
        self.windows = windows
        self.starts = np.array([w.interval.start_seconds for w in windows])
        ends = np.array([w.interval.end_seconds for w in windows])
        if np.any(np.diff(self.starts) < 0):
            raise ValueError("Windows must be start ordered")
        lengths = ends - self.starts
        maximum = lengths.max()
        self.neighbors = []
        for start, end, length in zip(self.starts, ends, lengths, strict=True):
            lo = np.searchsorted(self.starts, start - maximum, side="left")
            hi = np.searchsorted(self.starts, end, side="left")
            ids = np.arange(lo, hi, dtype=np.int32)
            overlap = np.maximum(
                0, np.minimum(ends[ids], end) - np.maximum(self.starts[ids], start)
            )
            self.neighbors.append(ids[overlap / (lengths[ids] + length - overlap) > threshold])

    def rank(self, scores, candidates=None, limit=None):
        scores = np.asarray(scores)
        if scores.shape != self.starts.shape or not np.isfinite(scores).all():
            raise ValueError("Provide one finite score per window")
        remaining = np.zeros(len(scores), bool)
        remaining[:] = candidates if candidates is not None else True
        order = np.lexsort((np.arange(len(scores)), self.starts, -scores))
        result = []
        for i in order:
            if remaining[i]:
                result.append(int(i))
                remaining[self.neighbors[i]] = False
                remaining[i] = False
                if limit is not None and len(result) == limit:
                    break
        return np.asarray(result, dtype=np.int32)


class EventIndex:
    def __init__(self, windows, truths):
        if len({t.behavior for t in truths}) > 1:
            raise ValueError("Evaluate one behavior at a time")
        self.truths = truths
        starts = np.array([t.interval.start_seconds for t in truths])
        ends = np.array([t.interval.end_seconds for t in truths])
        order = np.argsort(starts, kind="stable")
        sorted_starts = starts[order]
        maximum = (ends - starts).max() if len(truths) else 0
        self.rows = []
        for window in windows:
            start, end = window.interval.start_seconds, window.interval.end_seconds
            lo = np.searchsorted(sorted_starts, start - maximum, side="left")
            hi = np.searchsorted(sorted_starts, end, side="left")
            ids = np.sort(order[lo:hi])  # Original truth order breaks exact-IoU ties.
            ids = np.array([i for i in ids if truths[i].video_id == window.video_id], dtype=int)
            intersection = np.maximum(
                0, np.minimum(end, ends[ids]) - np.maximum(start, starts[ids])
            )
            iou = intersection / ((end - start) + (ends[ids] - starts[ids]) - intersection)
            self.rows.append((ids, iou))

    def match(self, ranked, threshold):
        if not isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Invalid event IoU")
        used = np.zeros(len(self.truths), bool)
        matches = []
        for row in ranked:
            ids, overlaps = self.rows[row]
            selected, best = None, threshold
            for index, score in zip(ids, overlaps, strict=True):
                if not used[index] and score >= best and (selected is None or score > best):
                    selected, best = int(index), score
            if selected is not None:
                used[selected] = True
            matches.append(selected)
        return np.array([i is not None for i in matches], bool), matches

    def coverage(self, candidates, threshold):
        if not self.truths:
            return None
        covered = np.zeros(len(self.truths), bool)
        for row in np.flatnonzero(candidates):
            ids, overlaps = self.rows[row]
            covered[ids[overlaps >= threshold]] = True
        return float(covered.mean())


def metrics_from_hits(hits, truth_count, budgets=(10, 20, 50)):
    cumulative = np.cumsum(hits)
    result = {
        "truth_count": truth_count,
        "prediction_count": len(hits),
        "average_precision": float(
            np.sum(cumulative / np.arange(1, len(hits) + 1) * hits) / truth_count
        )
        if truth_count
        else None,
        "budgets": {},
    }
    for k in (5, 10, 20):
        result[f"precision_at_{k}"] = int(hits[:k].sum()) / k
    for k in budgets:
        tp, reviewed = int(hits[:k].sum()), min(k, len(hits))
        fp, fn = reviewed - tp, truth_count - tp
        result["budgets"][str(k)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "reviewed": reviewed,
            "precision": tp / reviewed if reviewed else 0,
            "recall": tp / truth_count if truth_count else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_suggestions_per_hit": fp / tp if tp else None,
        }
        result[f"recall_at_{k}"] = tp / truth_count if truth_count else None
    return result
