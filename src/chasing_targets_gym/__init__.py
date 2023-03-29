__version__ = "0.0.4"

from gymnasium.envs.registration import register

from .sim import RobotChasingTargetEnv

register(
    id="ChasingTargets-v0",
    entry_point="chasing_targets_gym:RobotChasingTargetEnv",
)
