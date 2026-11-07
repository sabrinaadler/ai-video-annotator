import numpy as np
import pytest
from scipy.io import savemat

from ai_video_annotator.datasets.flyvsfly import load_bouts, merge_intervals, official_split
from ai_video_annotator.types import TimeInterval


def test_one_based_inclusive_single_frame_and_actor_union(tmp_path):
    cells = np.empty((2, 2), dtype=object)
    cells[0, 0] = np.array([[1, 1, 0], [4, 6, 2]])
    cells[1, 0] = np.array([[5, 7, 0]])
    cells[0, 1] = np.empty((0, 3))
    cells[1, 1] = np.array([[9, 9, 0]])
    path = tmp_path / "actions.mat"
    savemat(path, {"behs": np.array(["lunge", "unknown"], dtype=object), "bouts": cells})
    events, provenance = load_bouts(path, "fly", 30, 10)
    assert events[0].interval == TimeInterval(0, 1 / 30)
    assert events[1].interval == TimeInterval(3 / 30, 7 / 30)
    assert events[2].behavior == "unknown"
    assert provenance[1]["raw_flag"] == 2
    with pytest.raises(ValueError, match="range"):
        load_bouts(path, "fly", 30, 5)


def test_actor_union_does_not_merge_disjoint_touching_bouts():
    assert merge_intervals([TimeInterval(0, 1), TimeInterval(1, 2)]) == (
        TimeInterval(0, 1),
        TimeInterval(1, 2),
    )


def test_official_split_respects_matlab_indices(tmp_path):
    savemat(
        tmp_path / "traintest.mat",
        {
            "movienames": np.array(["movie1", "movie2"], dtype=object),
            "trainids": np.array([1]),
            "testids": np.array([2]),
        },
    )
    assert official_split(tmp_path) == {"movie1": "train", "movie2": "test"}
