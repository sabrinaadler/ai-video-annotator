import pytest
import torch

from ai_video_annotator.device import select_device


def test_cpu_can_be_requested_explicitly() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_device("quantum")
