"""The torch-free surface must stay torch-free.

``decisionrl.envs``, ``decisionrl.baselines`` and ``decisionrl.core`` are useful
to consumers that only simulate or evaluate — they should cost neither the
multi-gigabyte PyTorch install nor the seconds it takes to import. The top-level
package resolves its public names lazily (PEP 562) to keep that true, and these
tests pin the property down, since a single eager ``import torch`` anywhere in
the chain would silently undo it.

Each check runs in a fresh interpreter: ``sys.modules`` in this one is already
full of torch from the rest of the suite.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# The child must import the same decisionrl this process did, whether that is an
# installed wheel, an editable install or a source checkout on sys.path.
_CHILD_PATH = [p for p in sys.path if p]

# Makes torch unimportable, exactly as a missing install would: nothing else on
# sys.meta_path gets a chance to find it, and the failure is the same
# ModuleNotFoundError that absence produces.
_HIDE_TORCH = """
import sys


class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _NoTorch())
"""


def _run(script: str, *, hide_torch: bool = False) -> None:
    """Run ``script`` in a fresh interpreter, failing the test on a non-zero exit."""
    source = textwrap.dedent(script)
    if hide_torch:
        source = _HIDE_TORCH + source
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(_CHILD_PATH)},
    )
    if result.returncode != 0:
        pytest.fail(f"subprocess failed:\n{result.stdout}\n{result.stderr}")


def test_importing_envs_does_not_import_torch():
    """The headline guarantee: no torch in sys.modules after importing envs."""
    _run(
        """
        import sys

        import decisionrl.envs

        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, f"torch was imported by decisionrl.envs: {leaked}"
        """
    )


def test_torch_free_surface_imports_with_torch_uninstalled():
    """The same modules import when torch genuinely cannot be found."""
    _run(
        """
        import decisionrl
        import decisionrl.baselines
        import decisionrl.core
        import decisionrl.envs
        from decisionrl.core.env import Env

        # Not just importable - usable.
        env = decisionrl.envs.CartPole()
        obs, _ = env.reset(seed=0)
        assert isinstance(env, Env)
        assert env.observation_space.contains(obs)

        assert "torch" not in sys.modules
        """,
        hide_torch=True,
    )


def test_public_api_still_resolves():
    """Every name the package advertises is reachable, torch-backed ones included."""
    _run(
        """
        import decisionrl
        from decisionrl import PPO
        from decisionrl.algorithms import PPO as PPOFromSubmodule

        assert PPO is PPOFromSubmodule

        unresolved = [name for name in decisionrl.__all__ if not hasattr(decisionrl, name)]
        assert not unresolved, f"__all__ entries that do not resolve: {unresolved}"

        exported = {}
        exec("from decisionrl import *", exported)
        assert "PPO" in exported and "envs" in exported
        """
    )


def test_torch_is_imported_on_first_use_of_an_algorithm():
    """Laziness is deferral, not removal: touching PPO must still bring torch in."""
    _run(
        """
        import sys

        import decisionrl

        assert "torch" not in sys.modules
        _ = decisionrl.PPO
        assert "torch" in sys.modules
        """
    )


def test_unknown_attribute_raises_attribute_error():
    """__getattr__ must not turn typos into ImportError or infinite recursion."""
    import decisionrl

    with pytest.raises(AttributeError):
        _ = decisionrl.NoSuchAttribute
