from gymnasium.envs.registration import register

try:
    from ._planner import Planner
except ImportError:
    from warnings import warn

    warn("Unable to import C++ planner, using python native planner")
    from .py_planner import Planner

from .run import _main
from .sim import RobotChasingTargetEnv

register(
    id="ChasingTargets-v0",
    entry_point="chasing_targets_gym:RobotChasingTargetEnv",
)
