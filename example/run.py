import pstats
from cProfile import Profile

import numpy as np
from gymnasium import make

import chasing_targets_gym

try:
    from .planner import Planner
except ImportError:
    from planner import Planner

ROBOT_RADIUS = 0.1


def update_step(robots: np.ndarray, target_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Returns action for robot using pure pursuit algorithm"""
    X, Y = 0, 1
    lr_control = np.full([robots.shape[0], 2], 0.5, dtype=np.float64)
    alpha = np.arctan2(target_pos[Y] - robots[:, Y], target_pos[X] - robots[:, X])
    alpha -= robots[:, -1]
    delta = np.arctan2(2.0 * ROBOT_RADIUS * np.sin(alpha) * 3, 1.0)
    lr_control[delta > 0, 0] -= delta[delta > 0]
    lr_control[delta <= 0, 1] += delta[delta <= 0]
    # lr_control = np.clip(lr_control, -0.5, 0.5) # shouldn't be necessary

    return {"vL": lr_control[:, 0], "vR": lr_control[:, 1]}


def pure_pursuit(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    actions = []
    for robot, tgtid in zip(obs["current_robot"], obs["robot_target_idx"]):
        actions.append(update_step(robot[None], obs["future_target"][tgtid]))
    action = {}
    for key_ in actions[0]:
        action[key_] = np.stack([a[key_] for a in actions], axis=0)
    return action


def run_env():
    """Runs simulation of target chasers"""
    max_vel = 0.5
    env = make(
        "ChasingTargets-v0",
        render_mode="human",
        n_robots=10,
        n_targets=3,
        robot_radius=ROBOT_RADIUS,
        max_velocity=max_vel,
        barrier_velocity_range=max_vel,
        max_episode_steps=300,
        # recording_path=Path.cwd() / "test.mkv",
    )

    observation, info = env.reset(seed=2)

    planner = Planner(ROBOT_RADIUS, info["dt"], max_velocity=max_vel)
    # planner = pure_pursuit

    enable_profile = False
    if enable_profile:
        prof = Profile()
        prof.enable()

    done = False
    while not done:
        action = planner(observation)
        observation, _, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

    if enable_profile:
        prof.disable()
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(10)

    env.close()


if __name__ == "__main__":
    run_env()
