from typing import Dict

import numpy as np


def cartesian_product(*arrays):
    """Cartesian product of numpy arrays"""
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class Planner:
    """
    Basic planner from gym environment copied and refined from
    https://www.doc.ic.ac.uk/~ajd/Robotics/RoboticsResources/planningmultirobot.py
    """

    plan_ahead_steps = 10
    forward_weight = 12
    obstacle_weight = 6666
    max_acceleration = 0.4

    def __init__(self, agent_radius: float, dt: float, max_velocity: float) -> None:
        self.radius = agent_radius
        self.safe_dist = agent_radius
        self.max_velocity = max_velocity
        self.dt = dt
        self.tau = dt * self.plan_ahead_steps

    @property
    def width(self) -> float:
        return self.radius * 2.0

    def predictPosition(self, vL: np.ndarray, vR: np.ndarray, robot: np.ndarray):
        """
        Function to predict new robot position based on current pose and velocity controls
        Returns xnew, ynew, thetanew
        Also returns path. This is just used for graphics, and returns some complicated stuff
        used to draw the possible paths during planning. Don't worry about the details of that.
        """
        _, _, theta = robot
        # First cover general motion case
        R = self.radius * (vR + vL) / (vR - vL + np.finfo(vR.dtype).eps)
        dt = (vR - vL) / self.width
        dx = R * (np.sin(dt + theta) - np.sin(theta))
        dy = -R * (np.cos(dt + theta) - np.cos(theta))

        # Then cover straight motion case
        mask = np.isclose(vL, vR)
        dx[mask] = (vL * np.cos(theta))[mask]
        dy[mask] = (vL * np.sin(theta))[mask]

        return robot + self.tau * np.stack([dx, dy, dt], axis=-1)

    def calculateClosestObstacleDistance(self, robot, obstacle):
        """
        Function to calculate the closest obstacle at
        a position (x, y). Used during planning.
        """
        pairwise_distance = np.linalg.norm(robot[:, None] - obstacle[None], 2, axis=-1)
        return np.min(pairwise_distance, axis=1) - self.width

    def chooseAction(
        self,
        vL: float,
        vR: float,
        robot: np.ndarray,
        target: np.ndarray,
        obstacle: np.ndarray,
    ):
        """
        Planning
        We want to find the best benefit where we have a positive
        component for closeness to target, and a negative component
        for closeness to obstacles, for each of a choice of possible actions
        """
        # Range of possible motions: each of vL and vR could go up or down a bit
        dv = self.max_acceleration * self.dt
        actions = cartesian_product(
            np.array((vL - dv, vL, vL + dv)), np.array((vR - dv, vR, vR + dv))
        )

        # Predict new position in TAU seconds
        new_robot_pos = self.predictPosition(actions[:, 0], actions[:, 1], robot)

        # What is the distance to the closest obstacle from this possible position?
        distanceToObstacle = self.calculateClosestObstacleDistance(
            new_robot_pos[:, :2], obstacle
        )

        # Calculate how much close we've moved to target location
        previousTargetDistance = np.linalg.norm(robot[:2] - target, 2)
        newTargetDistance = np.linalg.norm(new_robot_pos[:, :2] - target, 2, axis=1)
        distanceForward = previousTargetDistance - newTargetDistance

        # Positive benefit
        distanceBenefit = self.forward_weight * distanceForward

        # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
        obstacleCost = (
            self.obstacle_weight
            * (self.safe_dist - distanceToObstacle)
            * (distanceToObstacle < self.safe_dist)
        )

        # Total benefit function to optimise
        benefit = distanceBenefit - obstacleCost

        # Invalid actions are -inf cost
        mask = np.abs(actions[:, 0]) > self.max_velocity
        mask |= np.abs(actions[:, 1]) > self.max_velocity
        benefit[mask] = -np.finfo(benefit.dtype).max

        best_idx = np.argmax(benefit)
        vLchosen, vRchosen = actions[best_idx]

        return {"vL": vLchosen, "vR": vRchosen}

    def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Determine the best action depending on the state observation
        """
        n_robot = obs["vL"].shape[0]
        actions = {k: np.empty((n_robot, 1), dtype=np.float64) for k in ["vL", "vR"]}

        for r_idx in range(n_robot):
            notr_idx = [i for i in range(0, n_robot) if i != r_idx]
            tgt_future = obs["future_target"][obs["robot_target_idx"][r_idx], :2]
            action = self.chooseAction(
                obs["vL"][r_idx, 0],
                obs["vR"][r_idx, 0],
                obs["current_robot"][r_idx, :3],
                tgt_future,
                obs["future_robot"][notr_idx, :2],
            )
            for k in ["vL", "vR"]:
                actions[k][r_idx] = action[k]

        return actions
