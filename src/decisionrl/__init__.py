"""decisionrl - a dependency-light, correctness-first reinforcement learning foundation.

Quick start
-----------
>>> from decisionrl.algorithms import PPO
>>> from decisionrl.envs import CartPole
>>> agent = PPO(CartPole(), seed=0)
>>> agent.learn(total_steps=50_000)          # doctest: +SKIP
>>> from decisionrl.training import evaluate_policy
>>> mean, std = evaluate_policy(agent, CartPole())   # doctest: +SKIP

Every agent shares the same surface: ``predict`` / ``learn`` / ``save`` / ``load``.

Lazy imports
------------
Nothing is imported when this package is. Every public name is resolved on first
access through the PEP 562 module ``__getattr__`` below, so ``from decisionrl
import PPO`` behaves exactly as it always has while ``import decisionrl`` itself
stays free.

This matters because the deep-RL half of the library needs PyTorch and the rest
does not. :mod:`decisionrl.envs`, :mod:`decisionrl.baselines`,
:mod:`decisionrl.core`, :mod:`decisionrl.solvers` and :mod:`decisionrl.wrappers`
import without torch installed at all, so a consumer that only simulates or
evaluates environments pays neither the multi-gigabyte install nor the seconds of
import time. Touch anything that trains — :mod:`decisionrl.algorithms`,
:mod:`decisionrl.networks`, ``PPO`` — and torch is imported then.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, List

__version__ = "0.4.0"

# Importable as ``decisionrl.<name>`` and, for those in ``__all__``, re-exported
# as an attribute of this package. ``cli`` and ``__main__`` are entry points
# rather than API, so they are reachable only via a real import statement.
_SUBMODULES = frozenset(
    {
        "algorithms",
        "alphazero",
        "bandits",
        "baselines",
        "buffers",
        "config",
        "configs",
        "core",
        "dashboard",
        "data",
        "distributed",
        "envs",
        "evaluation",
        "evolution",
        "exploration",
        "imitation",
        "meta",
        "multiagent",
        "networks",
        "registry",
        "rlhf",
        "serving",
        "solvers",
        "text",
        "tracking",
        "training",
        "tuning",
        "utils",
        "wrappers",
        "zoo",
    }
)

# Public name -> the submodule that defines it. Keep in sync with ``__all__``.
_ATTRIBUTES = {
    # core
    "Env": "core",
    "Wrapper": "core",
    "Space": "core",
    "Box": "core",
    "Discrete": "core",
    "Dict": "core",
    "Transition": "core",
    # algorithms
    "QLearning": "algorithms",
    "SARSA": "algorithms",
    "ExpectedSARSA": "algorithms",
    "DynaQ": "algorithms",
    "DQN": "algorithms",
    "C51": "algorithms",
    "QRDQN": "algorithms",
    "Rainbow": "algorithms",
    "REINFORCE": "algorithms",
    "A2C": "algorithms",
    "PPO": "algorithms",
    "TRPO": "algorithms",
    "GRPO": "algorithms",
    "IMPALA": "algorithms",
    "RecurrentPPO": "algorithms",
    "DDPG": "algorithms",
    "TD3": "algorithms",
    "SAC": "algorithms",
    "SACDiscrete": "algorithms",
    "TD3BC": "algorithms",
    "IQL": "algorithms",
    "CQL": "algorithms",
    "DecisionTransformer": "algorithms",
    "DiffusionPolicy": "algorithms",
    "HERDQN": "algorithms",
    "MBPO": "algorithms",
    "Dreamer": "algorithms",
    "DreamerRSSM": "algorithms",
    "NeuroevolutionAgent": "evolution",
    # offline data
    "TransitionDataset": "data",
    "collect_dataset": "data",
    "TrajectoryDataset": "data",
    "collect_trajectories": "data",
    "DistributedActorLearner": "distributed",
    # RLHF
    "RewardModel": "rlhf",
    "PreferenceDataset": "rlhf",
    "collect_segments": "rlhf",
    "synthetic_preferences": "rlhf",
    "train_reward_model": "rlhf",
    "RewardModelWrapper": "rlhf",
    "DPO": "rlhf",
    # imitation learning
    "BC": "imitation",
    "DAgger": "imitation",
    "GAIL": "imitation",
    "GAILDiscriminator": "imitation",
    "collect_expert_dataset": "imitation",
    # meta-RL (RL^2)
    "RL2Env": "meta",
    "make_meta_bandit": "meta",
    # reliable evaluation statistics
    "iqm": "evaluation",
    "bootstrap_ci": "evaluation",
    "aggregate_metrics": "evaluation",
    "performance_profile": "evaluation",
    "probability_of_improvement": "evaluation",
    "run_seeds": "evaluation",
    # helpers
    "evaluate_policy": "training",
    "set_seed": "utils",
    "make_agent": "registry",
    "make_env": "registry",
    "make_vec_env": "registry",
    "list_algorithms": "registry",
    "list_environments": "registry",
    "optuna_search": "tuning",
    # model zoo
    "list_pretrained": "zoo",
    "load_pretrained": "zoo",
    "save_to_zoo": "zoo",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562).

    The result is cached in module globals, so the lookup cost — and the import
    of the underlying submodule — is paid at most once.
    """
    if name in _SUBMODULES:
        value: Any = importlib.import_module(f".{name}", __name__)
    elif name in _ATTRIBUTES:
        value = getattr(importlib.import_module(f".{_ATTRIBUTES[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(__all__) | _SUBMODULES | {"__version__"})


if TYPE_CHECKING:
    # Static equivalents of the lazy imports above, so type checkers and IDEs
    # resolve the same names the runtime serves. Never executed.
    from . import (
        algorithms,
        alphazero,
        baselines,
        buffers,
        config,
        dashboard,
        envs,
        evolution,
        exploration,
        networks,
        solvers,
        text,
        tracking,
        training,
        utils,
        wrappers,
    )
    from .algorithms import (
        A2C,
        C51,
        CQL,
        DDPG,
        DQN,
        GRPO,
        HERDQN,
        IMPALA,
        IQL,
        MBPO,
        PPO,
        QRDQN,
        REINFORCE,
        SAC,
        SARSA,
        TD3,
        TD3BC,
        TRPO,
        DecisionTransformer,
        DiffusionPolicy,
        Dreamer,
        DreamerRSSM,
        DynaQ,
        ExpectedSARSA,
        QLearning,
        Rainbow,
        RecurrentPPO,
        SACDiscrete,
    )
    from .core import Box, Dict, Discrete, Env, Space, Transition, Wrapper
    from .data import (
        TrajectoryDataset,
        TransitionDataset,
        collect_dataset,
        collect_trajectories,
    )
    from .distributed import DistributedActorLearner
    from .evaluation import (
        aggregate_metrics,
        bootstrap_ci,
        iqm,
        performance_profile,
        probability_of_improvement,
        run_seeds,
    )
    from .evolution import NeuroevolutionAgent
    from .imitation import BC, GAIL, DAgger, GAILDiscriminator, collect_expert_dataset
    from .meta import RL2Env, make_meta_bandit
    from .registry import list_algorithms, list_environments, make_agent, make_env, make_vec_env
    from .rlhf import (
        DPO,
        PreferenceDataset,
        RewardModel,
        RewardModelWrapper,
        collect_segments,
        synthetic_preferences,
        train_reward_model,
    )
    from .training import evaluate_policy
    from .tuning import optuna_search
    from .utils import set_seed
    from .zoo import list_pretrained, load_pretrained, save_to_zoo

__all__ = [
    "__version__",
    # subpackages
    "algorithms",
    "baselines",
    "alphazero",
    "buffers",
    "config",
    "dashboard",
    "envs",
    "evolution",
    "exploration",
    "networks",
    "solvers",
    "text",
    "tracking",
    "training",
    "utils",
    "wrappers",
    # core
    "Env",
    "Wrapper",
    "Space",
    "Box",
    "Discrete",
    "Dict",
    "Transition",
    # algorithms
    "QLearning",
    "SARSA",
    "ExpectedSARSA",
    "DynaQ",
    "DQN",
    "C51",
    "QRDQN",
    "Rainbow",
    "REINFORCE",
    "A2C",
    "PPO",
    "TRPO",
    "GRPO",
    "IMPALA",
    "RecurrentPPO",
    "DDPG",
    "TD3",
    "SAC",
    "SACDiscrete",
    "TD3BC",
    "IQL",
    "CQL",
    "DecisionTransformer",
    "DiffusionPolicy",
    "HERDQN",
    "MBPO",
    "Dreamer",
    "DreamerRSSM",
    "NeuroevolutionAgent",
    # offline data
    "TransitionDataset",
    "collect_dataset",
    "TrajectoryDataset",
    "collect_trajectories",
    "DistributedActorLearner",
    # RLHF
    "RewardModel",
    "PreferenceDataset",
    "collect_segments",
    "synthetic_preferences",
    "train_reward_model",
    "RewardModelWrapper",
    "DPO",
    # imitation learning
    "BC",
    "DAgger",
    "GAIL",
    "GAILDiscriminator",
    "collect_expert_dataset",
    # meta-RL (RL^2)
    "RL2Env",
    "make_meta_bandit",
    # reliable evaluation statistics
    "iqm",
    "bootstrap_ci",
    "aggregate_metrics",
    "performance_profile",
    "probability_of_improvement",
    "run_seeds",
    # helpers
    "evaluate_policy",
    "set_seed",
    "make_agent",
    "make_env",
    "make_vec_env",
    "list_algorithms",
    "list_environments",
    "optuna_search",
    # model zoo
    "list_pretrained",
    "load_pretrained",
    "save_to_zoo",
]
