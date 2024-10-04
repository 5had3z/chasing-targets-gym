"""Example Planning Algorithm"""

import numpy as np
from typing_extensions import deprecated


def cartesian_product(*arrays):
    """Cartesian product of numpy arrays"""
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


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
        self.safe_dist = agent_radius
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
        theta = robot[..., 2, None]
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

        if robot.ndim == 2:  # Add extra dim when batched for broadcast
            robot = robot[:, None]
        return robot[..., :2] + self.tau * np.stack((dx, dy), axis=-1)

    def closest_obstacle_distance(self, robot, obstacle):
        """
        Calculates the closest obstacle at a position (x, y). Used during planning.
        """
        pairwise_distance = np.linalg.norm(
            robot[..., None, :] - obstacle[..., None, :, :], 2, axis=-1
        )
        return np.min(pairwise_distance, axis=-1) - self.width

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
        actions = actions[np.all(np.abs(actions) < self.max_velocity, axis=-1)]

        # Predict new position in TAU seconds
        new_robot_pos = self.predict_position(actions[:, 0], actions[:, 1], robot)

        # Calculate how much close we've moved to target location
        previousTargetDistance = np.linalg.norm(robot[:2] - target, 2)
        newTargetDistance = np.linalg.norm(new_robot_pos - target, 2, axis=1)
        distanceForward = previousTargetDistance - newTargetDistance

        # Positive benefit
        distanceBenefit = self.forward_weight * distanceForward

        # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
        distanceToObstacle = self.closest_obstacle_distance(new_robot_pos, obstacle)
        obstacleCost = (
            self.obstacle_weight
            * (self.safe_dist - distanceToObstacle)
            * (distanceToObstacle < self.safe_dist)
        )

        # Total benefit function to optimise
        benefit = distanceBenefit - obstacleCost

        # Select the best action's values
        vLchosen, vRchosen = actions[np.argmax(benefit)]

        return {"vL": vLchosen, "vR": vRchosen}

    def run_iterative(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run old iterative algorithm"""
        n_robot = obs["vL"].shape[0]
        actions = {k: np.empty(n_robot, dtype=np.float32) for k in ["vL", "vR"]}
        tgt_future = obs["future_target"][obs["robot_target_idx"], :2]
        for r_idx in range(n_robot):
            action = self.choose_action(
                obs["vL"][r_idx],
                obs["vR"][r_idx],
                obs["current_robot"][r_idx, :3],
                tgt_future[r_idx],
                np.delete(obs["future_robot"][:, :2], r_idx, axis=0),
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
            [cartesian_product(l + self.dv, r + self.dv) for l, r in zip(vL, vR)]
        )

        # Predict new position in TAU seconds
        new_robot_pos = self.predict_position(actions[..., 0], actions[..., 1], robot)

        # Calculate how much close we've moved to target location
        previousTargetDistance = np.linalg.norm(robot[..., :2] - target, 2, axis=-1)
        newTargetDistance = np.linalg.norm(new_robot_pos - target[:, None], 2, axis=-1)
        distanceForward = previousTargetDistance[:, None] - newTargetDistance

        # Positive benefit
        distanceBenefit = self.forward_weight * distanceForward

        # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
        distanceToObstacle = self.closest_obstacle_distance(new_robot_pos, obstacle)
        obstacleCost = (
            self.obstacle_weight
            * (self.safe_dist - distanceToObstacle)
            * (distanceToObstacle < self.safe_dist)
        )

        # Total benefit function to optimise
        benefit = distanceBenefit - obstacleCost
        invalid = np.any(np.abs(actions) > self.max_velocity, axis=-1)
        benefit[invalid] = -np.inf

        # Select the best action's values, not sure why np.take is misbehaving, loop instead
        vLchosen, vRchosen = [], []
        selects = np.argmax(benefit, axis=-1)
        for action, select in zip(actions, selects):
            vLchosen.append(action[select, 0])
            vRchosen.append(action[select, 1])

        return {"vL": np.array(vLchosen), "vR": np.array(vRchosen)}

    def run_batched(self, obs: dict[str, np.ndarray]):
        """Run entire algorithm batched"""
        n_robot = obs["vL"].shape[0]
        obstacles = np.stack(
            [np.delete(obs["future_robot"][:, :2], i, axis=0) for i in range(n_robot)]
        )
        actions = {k: np.empty((n_robot, 1), dtype=np.float32) for k in ["vL", "vR"]}
        tgt_future = obs["future_target"][obs["robot_target_idx"], :2]
        actions = self.choose_action_batched(
            obs["vL"],
            obs["vR"],
            obs["current_robot"][:, :3],
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
