__version__ = "0.0.7"

from gymnasium.envs.registration import register

from .sim import RobotChasingTargetEnv
from .run import _main

register(
    id="ChasingTargets-v0",
    entry_point="chasing_targets_gym:RobotChasingTargetEnv",
)
