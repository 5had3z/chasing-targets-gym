from gymnasium.envs.registration import register

from ._planner import Planner
from .run import _main
from .sim import RobotChasingTargetEnv

register(
    id="ChasingTargets-v0",
    entry_point="chasing_targets_gym:RobotChasingTargetEnv",
)
