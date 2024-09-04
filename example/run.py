#!/usr/bin/env python3
from typing import Annotated

import numpy as np
import typer
from gymnasium import Env, make

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


def run_sim(env: Env, planner):
    observation, _ = env.reset(seed=2)

    done = False
    while not done:
        action = planner(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if env.render_mode == "human":
            env.render()
        done = terminated or truncated


app = typer.Typer()


@app.command()
def main(profile: Annotated[bool, typer.Option()] = False):
    """
    Runs simulation of target chasers.
    Profile sim with scalene turned off so it only starts with sim:
    `scalene --cpu --off example/run.py --profile`
    """
    max_vel = 0.5
    env = make(
        "ChasingTargets-v0",
        render_mode="human" if not profile else None,
        n_robots=10,
        n_targets=3,
        robot_radius=ROBOT_RADIUS,
        max_velocity=max_vel,
        barrier_velocity_range=max_vel,
        max_episode_steps=1000,
        # recording_path=Path.cwd() / "test.mkv",
    )

    planner = Planner(ROBOT_RADIUS, env.get_wrapper_attr("dt"), max_velocity=max_vel)
    # planner = pure_pursuit

    if profile:
        from scalene import scalene_profiler

        scalene_profiler.start()

    for _ in range(10 if profile else 1):
        run_sim(env, planner)

    if profile:
        scalene_profiler.stop()

    env.close()


if __name__ == "__main__":
    app()
