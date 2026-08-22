"""Batteries-included environments (no external dependencies required).

Also exposes optional Gymnasium interop via :func:`make_gym` / :class:`GymAdapter`.
"""

from .acrobot import Acrobot
from .bandit import BernoulliBandit, MultiArmedBandit
from .bitflipping import BitFlipping
from .cartpole import CartPole
from .contextual_bandit import ContextualBandit
from .dataset_demand_inventory import DatasetDemandInventory
from .energy import EnergyMicrogrid
from .grid_world import GridWorld
from .gym import (
    GymAdapter,
    convert_space,
    make_atari,
    make_gym,
    make_gym_vec,
    make_minigrid,
    register_envs,
    to_gymnasium,
)
from .inventory import InventoryManagement
from .joint_pricing_inventory import JointPricingInventory
from .lunar_lander import LunarLander
from .mountain_car import MountainCar, MountainCarContinuous
from .navigation import Navigation2D
from .nonstationary_inventory import NonstationaryInventory
from .pendulum import Pendulum
from .point_mass import PointMass
from .portfolio import PortfolioAllocation
from .pricing import DynamicPricing
from .queueing import QueueAdmissionControl
from .reacher import ReacherArm
from .supply_chain import SupplyChain
from .thermostat import Thermostat

#: The applied subset: operational decisions with a cost function and a classical
#: operations-research baseline beside them in :mod:`decisionrl.baselines`. Named here
#: rather than counted by hand, because the size of this set is the library's positioning
#: and it is quoted in CITATION.cff and the packaging description.
APPLIED_ENVIRONMENTS = (
    "DatasetDemandInventory",
    "InventoryManagement",
    "Thermostat",
    "DynamicPricing",
    "QueueAdmissionControl",
    "EnergyMicrogrid",
    "SupplyChain",
    "NonstationaryInventory",
    "JointPricingInventory",
)

__all__ = [
    "APPLIED_ENVIRONMENTS",
    # classic / toy
    "GridWorld",
    "BitFlipping",
    "MultiArmedBandit",
    "BernoulliBandit",
    "ContextualBandit",
    "CartPole",
    "Pendulum",
    "PointMass",
    "MountainCar",
    "MountainCarContinuous",
    "Acrobot",
    # complex / varied scenarios
    "ReacherArm",
    "Navigation2D",
    "LunarLander",
    "PortfolioAllocation",
    # applied (operational / business decisions)
    "DatasetDemandInventory",
    "InventoryManagement",
    "Thermostat",
    "DynamicPricing",
    "QueueAdmissionControl",
    "EnergyMicrogrid",
    "SupplyChain",
    "NonstationaryInventory",
    "JointPricingInventory",
    # gymnasium interop
    "GymAdapter",
    "make_gym",
    "make_gym_vec",
    "make_atari",
    "make_minigrid",
    "convert_space",
    "to_gymnasium",
    "register_envs",
]
