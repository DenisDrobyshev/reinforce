"""Shared plumbing for the package's lazy imports.

PyTorch is an optional dependency (see ``pyproject.toml``), so a name that needs it
can fail to resolve long after ``pip install decisionrl`` succeeded. Left alone that
surfaces as a bare ``ModuleNotFoundError: No module named 'torch'`` raised from
somewhere inside the package, which says nothing about what to do next. Route the
lazy imports through here instead, so the failure names the attribute the caller
asked for and the command that fixes it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["import_module"]

_TORCH_MISSING = """{what} needs PyTorch, which is not installed.

decisionrl keeps torch optional: the environments, the classical baselines, the
solvers and the core API are pure NumPy and install without it. The half that
trains does need it:

    pip install "decisionrl[torch]"

For a CPU-only or a specific CUDA build, install torch yourself first --
https://pytorch.org/get-started/locally/ -- and the extra will be satisfied."""


def _is_torch(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


def import_module(name: str, package: str, what: str) -> ModuleType:
    """Import ``name`` relative to ``package``, reporting a missing torch clearly.

    ``what`` names the thing the caller was after (``"PPO"``,
    ``"decisionrl.algorithms"``), so the message points at the caller's own request
    rather than at an internal module they have never heard of.
    """
    try:
        return importlib.import_module(name, package)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        # Only torch gets the friendly treatment. Any other missing module is a real
        # failure whose own message is the useful one, and swallowing it would hide it.
        if not _is_torch(missing):
            raise
        raise ModuleNotFoundError(_TORCH_MISSING.format(what=what), name=missing) from exc
