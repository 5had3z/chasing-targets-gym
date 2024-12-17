"""Example Planning Algorithm"""

import numpy as np
from typing_extensions import deprecated


def cartesian_product(*arrays):
    """Cartesian product of numpy arrays"""
    la = len(arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i] = a
    return arr.reshape(la, -1)


@deprecated("C++ Native planner is 70x faster, use chasing_targets_gym.Planner")
class Planner:
    """
    Basic planner from gym environment copied and refined from
    https://www.doc.ic.ac.uk/~ajd/Robotics/RoboticsResources/planningmultirobot.py
    """

    plan_ahead_steps = 10
    forward_weight = 12
    obstacle_weight = 6666
    max_acceleration = 0.4

    def __init__(
        self,
        agent_radius: float,
        dt: float,
        max_velocity: float,
        use_batched: bool = True,
    ) -> None:
        self.radius = agent_radius
        self.max_velocity = max_velocity
        dv = self.max_acceleration * dt
        self.dv = np.array([-dv, 0, dv], dtype=np.float32)
        self.tau = dt * self.plan_ahead_steps
        self.use_batched = use_batched

    @property
    def width(self) -> float:
        return self.radius * 2.0

    def predict_position(self, vL: np.ndarray, vR: np.ndarray, robot: np.ndarray):
        """
        Function to predict new robot position based on current pose and velocity controls
        Returns xnew, ynew, thetanew
        Also returns path. This is just used for graphics, and returns some complicated stuff
        used to draw the possible paths during planning. Don't worry about the details of that.
        """
        theta = robot[2, None]
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        # First cover general motion case
        R = self.radius * (vR + vL) / (vR - vL + np.finfo(vR.dtype).eps)
        new_th = (vR - vL) / self.width + theta
        dx = R * (np.sin(new_th) - sin_th)
        dy = -R * (np.cos(new_th) - cos_th)

        # Then cover straight motion case
        mask = np.abs(vL - vR) < 1e-3
        # assert (mask == np.isclose(vL, vR)).all()
        dx[mask] = (vL * cos_th)[mask]
        dy[mask] = (vL * sin_th)[mask]

        return robot[:2, None] + self.tau * np.stack((dx, dy), axis=0)

    def closest_obstacle_distance(self, robot, obstacle):
        """
        Calculates the closest obstacle at a position (x, y). Used during planning.
        """
        pairwise_distance = np.linalg.norm(
            robot[:, None] - obstacle[:, :, None], 2, axis=0
        )
        return np.min(pairwise_distance, axis=0)

    def choose_action(
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
        actions = cartesian_product(vL + self.dv, vR + self.dv)
        # Remove invalid actions
        actions = actions[:, np.all(np.abs(actions) < self.max_velocity, axis=0)]

        # Predict new position in TAU seconds
        new_robot_pos = self.predict_position(actions[0], actions[1], robot)

        # Calculate how much close we've moved to target location
        previousTargetDistance = np.linalg.norm(robot[:2] - target, 2)
        newTargetDistance = np.linalg.norm(new_robot_pos - target[:, None], 2, axis=0)
        distanceForward = previousTargetDistance - newTargetDistance

        # Positive benefit
        distanceBenefit = self.forward_weight * distanceForward

        # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
        distanceToObstacle = self.closest_obstacle_distance(new_robot_pos, obstacle)
        obstacleCost = (
            self.obstacle_weight
            * (4 * self.radius - distanceToObstacle)
            * (distanceToObstacle < 4 * self.radius)
        )

        # Total benefit function to optimise
        benefit = distanceBenefit - obstacleCost

        # Select the best action's values
        vLchosen, vRchosen = actions[:, np.argmax(benefit)]

        return {"vL": vLchosen, "vR": vRchosen}

    def run_iterative(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run old iterative algorithm"""
        n_robot = obs["vL"].shape[0]
        actions = {k: np.empty(n_robot, dtype=np.float32) for k in ["vL", "vR"]}
        tgt_future = obs["future_target"][:2, obs["robot_target_idx"]]
        for r_idx in range(n_robot):
            action = self.choose_action(
                obs["vL"][r_idx],
                obs["vR"][r_idx],
                obs["current_robot"][:3, r_idx],
                tgt_future[:, r_idx],
                np.delete(obs["future_robot"][:2], r_idx, axis=1),
            )
            for k in ["vL", "vR"]:
                actions[k][r_idx] = action[k]

        return actions

    def choose_action_batched(
        self,
        vL: np.ndarray,
        vR: np.ndarray,
        robot: np.ndarray,
        target: np.ndarray,
        obstacle: np.ndarray,
    ):
        """Run algorithm batched, trading memory for speed"""
        actions = np.stack(
            [cartesian_product(l + self.dv, r + self.dv) for l, r in zip(vL, vR)],
            axis=-1,
        )

        # Predict new position in TAU seconds
        new_robot_pos = self.predict_position(actions[0], actions[1], robot)

        # Calculate how much close we've moved to target location
        previousTargetDistance = np.linalg.norm(robot[:2] - target, 2, axis=0)
        newTargetDistance = np.linalg.norm(new_robot_pos - target[:, None], 2, axis=0)
        distanceForward = previousTargetDistance[None] - newTargetDistance

        # Positive benefit
        distanceBenefit = self.forward_weight * distanceForward

        # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
        distanceToObstacle = self.closest_obstacle_distance(new_robot_pos, obstacle)
        obstacleCost = (
            self.obstacle_weight
            * (4 * self.radius - distanceToObstacle)
            * (distanceToObstacle < 4 * self.radius)
        )

        # Total benefit function to optimise
        benefit = distanceBenefit - obstacleCost
        invalid = np.any(np.abs(actions) > self.max_velocity, axis=0)
        benefit[invalid] = -np.inf

        # Select the best action's values, not sure why np.take is misbehaving, loop instead
        vLchosen, vRchosen = [], []
        selects = np.argmax(benefit, axis=0)
        for i, select in enumerate(selects):
            vLchosen.append(actions[0, select, i])
            vRchosen.append(actions[1, select, i])

        return {"vL": np.array(vLchosen), "vR": np.array(vRchosen)}

    def run_batched(self, obs: dict[str, np.ndarray]):
        """Run entire algorithm batched"""
        n_robot = obs["vL"].shape[0]
        obstacles = np.stack(
            [np.delete(obs["future_robot"][:2], i, axis=1) for i in range(n_robot)],
            axis=-1,
        )
        tgt_future = obs["future_target"][:2, obs["robot_target_idx"]]
        actions = self.choose_action_batched(
            obs["vL"],
            obs["vR"],
            obs["current_robot"][:3],
            tgt_future,
            obstacles,
        )
        return actions

    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Determine the best action depending on the state observation
        """
        obs = {
            k: v.astype(np.float32) if v.dtype.kind == "f" else v
            for k, v in obs.items()
        }

        if self.use_batched:
            return self.run_batched(obs)
        return self.run_iterative(obs)
