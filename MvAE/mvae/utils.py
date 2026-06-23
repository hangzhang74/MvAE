"""Utility functions for MvAE."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_hidden_dims(hidden_dims_text: str | tuple[int, ...] | list[int] | None) -> Tuple[int, ...]:
    """Convert hidden-dimension text such as ``"1024,512,256"`` to a tuple."""
    if hidden_dims_text is None:
        return ()
    if isinstance(hidden_dims_text, tuple):
        return tuple(int(x) for x in hidden_dims_text)
    if isinstance(hidden_dims_text, list):
        return tuple(int(x) for x in hidden_dims_text)

    hidden_dims_text = str(hidden_dims_text).strip()
    if hidden_dims_text == "":
        return ()
    return tuple(int(x.strip()) for x in hidden_dims_text.split(",") if x.strip() != "")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to a PyTorch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
