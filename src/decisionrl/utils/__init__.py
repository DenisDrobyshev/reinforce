"""Utility helpers: seeding, logging, normalization and torch tooling.

Everything here is torch-free except :mod:`decisionrl.utils.torch_utils`, whose
names are resolved lazily (PEP 562) so that importing this package — which
:mod:`decisionrl.core` does, for :class:`Logger` — does not pull in PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, List

from .dashboard import plot_dashboard
from .logger import HistoryLogger, Logger
from .render import record_gif
from .running_mean_std import RunningMeanStd
from .seeding import set_seed

_TORCH_UTILS = frozenset(
    {
        "explained_variance",
        "get_device",
        "hard_update",
        "maybe_compile",
        "polyak_update",
        "soft_update",
        "to_tensor",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _TORCH_UTILS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(".torch_utils", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from .torch_utils import (
        explained_variance,
        get_device,
        hard_update,
        maybe_compile,
        polyak_update,
        soft_update,
        to_tensor,
    )

__all__ = [
    "Logger",
    "HistoryLogger",
    "RunningMeanStd",
    "set_seed",
    "get_device",
    "to_tensor",
    "soft_update",
    "hard_update",
    "polyak_update",
    "explained_variance",
    "maybe_compile",
    "record_gif",
    "plot_dashboard",
]
