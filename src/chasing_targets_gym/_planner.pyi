import numpy

def inplace_move_targets(
    targets: numpy.ndarray, dt: float, limits: numpy.ndarray, n_steps: int
):
    """Move targets n iterations"""

class Planner:
    """Same as python planner but c++ impl for 70x speeeed"""

    def __init__(
        self,
        agent_radius: float,
        dt: float,
        max_velocity: float,
    ): ...
    def __call__(
        self, observation: dict[str, numpy.ndarray]
    ) -> dict[str, numpy.ndarray]: ...
