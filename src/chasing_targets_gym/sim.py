from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from . import render_utils as ru
from .robots import Robots


class RobotChasingTargetEnv(gym.Env):
    """A multi-robot planning environment for gym.

    This environment simulates the movements of multiple robots in an environment with multiple
    targets to chase. The robots are controlled by the actions given by an agent. The observations
    received by the agent includes information about the position, velocity and orientation of each
    robot, the future position of the target and the future positio of the obstacles. The goal of
    the agent is to navigate the robots to the target while avoiding collisions with the obstacles.

    Args:
        n_robots (int): The number of robots in the environment.
        n_targets (int): The number of targets in the environment.
        render_mode (str): The render mode of the environment, either "rgb_array" or "human".
        barrier_radius (float): The radius of each barrier.
        robot_radius (float): The radius of each robot.
        wheel_blob (float): The size of the wheel blob.
        max_velocity (float): The maximum velocity of each robot.
        max_acceleration (float): The maximum acceleration of each robot.
        target_velocity_std (float): Standard deviation used for the normal distribution used for\
generating target particle velocities.
        dt (float): The time step of the simulation.
        steps_ahead_to_plan (int): The number of steps ahead the robots should plan for.
        reach_target_reward (float): The reward given when a robot reaches the target.
        collision_penalty (float): The penalty given when a robot collides with a barrier or\
another robot.
        reset_when_target_reached (bool): A flag indicating whether the environment should reset\
when a robot reaches the target.
        recording_path (Path | None) : Optional path to write a video of the simulation to,\
if none no video (default: None).
        sandbox_dimensions (Tuple[float, float, float, float] | None): the extents of the sandbox\
dimensions, default value is (-4., -3., 4., 3.)
    """

    metadata = {"render_modes": ["rgb_array", "human", "video"], "render_fps": 30}

    _f_dtype = np.float32

    def __init__(
        self,
        n_robots: int = 20,
        n_targets: int = 5,
        render_mode: str | None = None,
        robot_radius: float = 0.1,
        wheel_blob: float = 0.04,
        max_velocity: float = 0.5,
        max_acceleration: float = 0.4,
        target_velocity_std: float = 0.2,
        dt: float = 0.1,
        steps_ahead_to_plan: int = 10,
        reach_target_reward: float = 1000.0,
        collision_penalty: float = -500.0,
        reset_when_target_reached: bool = False,
        recording_path: Path | None = None,
        sandbox_dimensions: tuple[float, float, float, float] | None = None,
    ):
        self.robots = Robots(n_robots, robot_radius, dt, max_acceleration)
        self.targets = np.empty((n_targets, 4), dtype=self._f_dtype)
        self.target_idxs = np.empty(n_robots, dtype=np.int64)
        self.dt = dt
        self.steps_ahead_to_plan = steps_ahead_to_plan
        self.reset_when_target_reached = reset_when_target_reached
        self.wheel_blob = wheel_blob
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.target_velocity_std = target_velocity_std

        self.collision_markers: list[ru.DecayingMarker] = []
        self.reward_markers: list[ru.DecayingMarker] = []

        self.reach_target_reward = reach_target_reward
        self.collision_penalty = collision_penalty

        self.field_limits = (
            (-4.0, -3.0, 4.0, 3.0) if sandbox_dimensions is None else sandbox_dimensions
        )

        self.recorder = (
            None
            if recording_path is None
            else ru.PyGameRecorder(recording_path, ru.size, self.metadata["render_fps"])
        )

        self.render_mode = render_mode
        if self.recorder is not None and self.render_mode is None:
            self.render_mode = "video"

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.action_space = spaces.Dict(
            {
                "vR": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    dtype=self._f_dtype,
                    shape=(n_robots,),
                ),
                "vL": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    dtype=self._f_dtype,
                    shape=(n_robots,),
                ),
            }
        )

        min_limit = list(self.field_limits[:2])
        max_limit = list(self.field_limits[2:])

        target_min = np.array(min_limit + [-max_velocity] * 2, dtype=self._f_dtype)
        target_min = np.repeat(target_min[None], n_targets, axis=0)
        target_max = np.array(max_limit + [max_velocity] * 2, dtype=self._f_dtype)
        target_max = np.repeat(target_max[None], n_targets, axis=0)

        robot_min = np.array(
            min_limit
            + [-np.pi]
            + [-max_velocity] * 2
            + [-self.max_velocity / robot_radius],
            dtype=self._f_dtype,
        )
        robot_min = np.repeat(robot_min[None], n_robots, axis=0)
        robot_max = np.array(
            max_limit
            + [np.pi]
            + [max_velocity] * 2
            + [self.max_velocity / robot_radius],
            dtype=self._f_dtype,
        )
        robot_max = np.repeat(robot_max[None], n_robots, axis=0)

        self.observation_space = spaces.Dict(
            {
                "vR": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    dtype=self._f_dtype,
                    shape=(n_robots,),
                ),
                "vL": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    dtype=self._f_dtype,
                    shape=(n_robots,),
                ),
                "current_robot": spaces.Box(low=robot_min, high=robot_max),
                "future_robot": spaces.Box(low=robot_min, high=robot_max),
                "current_target": spaces.Box(low=target_min, high=target_max),
                "future_target": spaces.Box(low=target_min, high=target_max),
                "robot_target_idx": spaces.MultiDiscrete(nvec=[n_targets] * n_robots),
            }
        )

        self._info = {
            "n_robots": self.n_robots,
            "n_targets": self.n_targets,
            "max_acceleration": self.max_acceleration,
            "max_velocity": self.max_velocity,
            "robot_radius": self.robot_radius,
            "dt": self.dt,
            "tau": self.tau,
        }

        self.window: pygame.surface.Surface | None = None
        self.clock = pygame.time.Clock()

    @property
    def n_targets(self) -> int:
        return self.targets.shape[0]

    @property
    def n_robots(self) -> int:
        return len(self.robots)

    @property
    def robot_radius(self) -> float:
        return self.robots.radius

    @property
    def robot_width(self) -> float:
        return 2 * self.robot_radius

    @property
    def tau(self) -> float:
        return self.dt * self.steps_ahead_to_plan

    def _get_obs(self) -> dict[str, np.ndarray]:
        targets = self.targets.copy()
        for _ in range(self.steps_ahead_to_plan):
            self._move_targets(targets)

        robot_est = self.robots.forecast(self.tau)
        for i in [0, 1]:
            np.clip(
                robot_est[:, i],
                self.field_limits[i],
                self.field_limits[i + 2],
                robot_est[:, i],
            )

        obs = {
            "vR": self.robots.vR,
            "vL": self.robots.vL,
            "current_robot": self.robots.state[:, :6],
            "future_robot": robot_est,
            "current_target": self.targets,
            "future_target": targets,
            "robot_target_idx": self.target_idxs,
        }
        # assert self.observation_space.contains(obs)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        # Setup random target states
        self.targets[:, 0] = self.np_random.uniform(
            self.field_limits[0], self.field_limits[2], self.n_targets
        ).astype(self._f_dtype)
        self.targets[:, 1] = self.np_random.uniform(
            self.field_limits[1], self.field_limits[3], self.n_targets
        ).astype(self._f_dtype)
        self.targets[:, 2] = (
            self.np_random.normal(0.0, self.target_velocity_std, self.n_targets)
            .clip(-self.max_velocity, self.max_velocity)
            .astype(self._f_dtype)
        )
        self.targets[:, 3] = (
            self.np_random.normal(0.0, self.target_velocity_std, self.n_targets)
            .clip(-self.max_velocity, self.max_velocity)
            .astype(self._f_dtype)
        )

        # Setup robots at random poses
        self.robots.reset()
        self.robots.x = self.np_random.uniform(
            self.field_limits[0], self.field_limits[2], self.n_robots
        ).astype(self._f_dtype)
        self.robots.y = self.np_random.uniform(
            self.field_limits[1], self.field_limits[3], self.n_robots
        ).astype(self._f_dtype)
        self.robots.theta = self.np_random.uniform(-np.pi, np.pi, self.n_robots).astype(
            self._f_dtype
        )

        self.target_idxs = self.np_random.integers(0, self.n_targets, self.n_robots)

        # Reset display markers
        self.collision_markers.clear()
        self.reward_markers.clear()

        return self._get_obs(), self._info

    def _move_targets(self, targets: np.ndarray) -> None:
        targets[:, :2] += targets[:, 2:] * self.dt

        # Flip velocity when hitting boundary and clip particle to boundary limit.
        mask = targets[:, 0] < self.field_limits[0]
        mask |= targets[:, 0] > self.field_limits[2]
        targets[mask, 2] *= -1
        np.clip(
            targets[:, 0], self.field_limits[0], self.field_limits[2], targets[:, 0]
        )

        mask = targets[:, 1] < self.field_limits[1]
        mask |= targets[:, 1] > self.field_limits[3]
        targets[mask, 3] *= -1
        np.clip(
            targets[:, 1], self.field_limits[1], self.field_limits[3], targets[:, 1]
        )

    def step(self, action: dict[str, np.ndarray]):
        assert self.action_space.contains(action)
        self._move_targets(self.targets)
        self.robots.step(action)
        # Robots can scrape against the border
        np.clip(
            self.robots.x, self.field_limits[0], self.field_limits[2], self.robots.x
        )
        np.clip(
            self.robots.y, self.field_limits[1], self.field_limits[3], self.robots.y
        )

        all_positions = np.concatenate(
            [self.robots.state[:, :2], self.targets[:, :2]], axis=0
        )
        distances: np.ndarray = np.linalg.norm(
            self.robots.state[:, None, :2] - all_positions[None], 2, axis=-1
        )
        collisions = distances < self.robot_width

        reward = (
            0.5
            * self.collision_penalty
            * ((collisions[:, : self.n_robots].sum() - self.n_robots))
        )

        offset_tgt = self.target_idxs + self.n_robots
        for ridx in range(self.n_robots):
            if collisions[ridx, offset_tgt[ridx]]:
                reward += self.reach_target_reward
                self.target_idxs[ridx] = self.np_random.integers(0, self.n_targets)
                self.robots.history[ridx].clear()

        if self.render_mode == "human":
            for ridx in range(self.n_robots):
                robot_position = self.robots.state[ridx, :2]
                collision_coords = all_positions[collisions[ridx]]
                for idx, coord in zip(np.argwhere(collisions[ridx]), collision_coords):
                    mean_coord = (coord + robot_position) * 0.5
                    if idx == ridx:
                        pass
                    elif idx < self.n_robots:
                        self.collision_markers.append(ru.DecayingMarker(mean_coord))
                    else:
                        self.reward_markers.append(ru.DecayingMarker(mean_coord))

        return self._get_obs(), reward, False, False, self._info

    def _draw_targets(self, screen: pygame.Surface):
        for target in self.targets:
            pygame.draw.circle(
                screen,
                ru.lightblue,
                ru.to_display(*target[:2]),
                int(ru.k * self.robot_radius),
                0,
            )

    def _draw_event_markers(self, screen: pygame.Surface):
        for collision in self.collision_markers:
            pygame.draw.circle(
                screen,
                ru.red,
                ru.to_display(*collision.position),
                int(ru.k * self.robot_radius) // 2,
                0,
            )
        self.collision_markers = [m for m in self.collision_markers if not m.expired()]
        for reward in self.reward_markers:
            pygame.draw.circle(
                screen,
                ru.green,
                ru.to_display(*reward.position),
                int(ru.k * self.robot_radius) // 2,
                0,
            )
        self.reward_markers = [m for m in self.reward_markers if not m.expired()]

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(ru.size)

        canvas = pygame.Surface(ru.size)
        canvas.fill(ru.black)

        self._draw_targets(canvas)
        self.robots.draw(canvas, self.wheel_blob)
        self._draw_event_markers(canvas)

        if self.recorder is not None:
            self.recorder(canvas)

        if self.window is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        if self.recorder is not None:
            self.recorder.close()
