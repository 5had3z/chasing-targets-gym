from gymnasium.envs.registration import register

from .sim import RobotChasingTargetEnv
from .run import _main
from ._planner import Planner

register(
    id="ChasingTargets-v0",
    entry_point="chasing_targets_gym:RobotChasingTargetEnv",
)
