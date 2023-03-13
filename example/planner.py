import math
import itertools
from typing import Dict, Any

import numpy as np


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
        self.agent_radius = agent_radius
        self.safe_dist = agent_radius
        self.max_velocity = max_velocity
        self.dt = dt
        self.tau = dt * self.plan_ahead_steps

    def predictPosition(self, vL: float, vR: float, robot: np.ndarray, deltat: float):
        """
        Function to predict new robot position based on current pose and velocity controls
        Uses time deltat in future
        Returns xnew, ynew, thetanew
        Also returns path. This is just used for graphics, and returns some complicated stuff
        used to draw the possible paths during planning. Don't worry about the details of that.
        """
        x, y, theta = robot
        W = 2 * self.agent_radius

        if np.isclose(vL, vR):  # Straight line motion
            xnew = x + vL * deltat * math.cos(theta)
            ynew = y + vL * deltat * math.sin(theta)
            thetanew = theta
        elif np.isclose(vL, -vR):  # Pure rotation motion
            xnew = x
            ynew = y
            thetanew = theta + ((vR - vL) * deltat / W)
        else:  # Rotation and arc angle of general circular motion
            R = W / 2.0 * (vR + vL) / (vR - vL)
            deltatheta = (vR - vL) * deltat / W
            xnew = x + R * (math.sin(deltatheta + theta) - math.sin(theta))
            ynew = y - R * (math.cos(deltatheta + theta) - math.cos(theta))
            thetanew = theta + deltatheta

        return np.array([xnew, ynew, thetanew])

    def calculateClosestObstacleDistance(self, robot, obstacle):
        """
        Function to calculate the closest obstacle at
        a position (x, y). Used during planning.
        """
        return (
            np.min(np.linalg.norm(robot - obstacle, 2, axis=1)) - 2 * self.agent_radius
        )

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
        bestBenefit = -100000
        vLchosen = vL
        vRchosen = vR

        # Range of possible motions: each of vL and vR could go up or down a bit
        dv = self.max_acceleration * self.dt
        for vLpossible, vRpossible in itertools.product(
            (vL - dv, vL, vL + dv), (vR - dv * self.dt, vR, vR + dv)
        ):
            # We can only choose an action if it's within velocity limits
            if not (
                (-self.max_velocity <= vLpossible <= self.max_velocity)
                and (-self.max_velocity <= vRpossible <= self.max_velocity)
            ):
                continue

            # Predict new position in TAU seconds
            new_robot_pos = self.predictPosition(
                vLpossible, vRpossible, robot, self.tau
            )

            # What is the distance to the closest obstacle from this possible position?
            distanceToObstacle = self.calculateClosestObstacleDistance(
                new_robot_pos[:-1], obstacle
            )

            # Calculate how much close we've moved to target location
            previousTargetDistance = np.linalg.norm(robot[:-1] - target, 2)
            newTargetDistance = np.linalg.norm(new_robot_pos[:-1] - target, 2)
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
            if benefit > bestBenefit:
                vLchosen = vLpossible
                vRchosen = vRpossible
                bestBenefit = benefit

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
