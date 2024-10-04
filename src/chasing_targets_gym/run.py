#!/usr/bin/env python3
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from ._planner import Planner
from .sim import RobotChasingTargetEnv

ROBOT_RADIUS = 0.1
MAX_VEL = 0.5


def update_step(robots: np.ndarray, target_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Returns action for robot using pure pursuit algorithm"""
    X, Y = 0, 1
    lr_control = np.full([robots.shape[0], 2], 0.5, dtype=np.float32)
    alpha = np.arctan2(target_pos[Y] - robots[:, Y], target_pos[X] - robots[:, X])
    alpha -= robots[:, -1]
    delta = np.arctan2(2.0 * ROBOT_RADIUS * np.sin(alpha) * 3, 1.0)
    lr_control[delta > 0, 0] -= delta[delta > 0]
    lr_control[delta <= 0, 1] += delta[delta <= 0]
    return {"vL": lr_control[:, 0], "vR": lr_control[:, 1]}


def pure_pursuit(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Simply go directly to the predicted target position, no collision avoidance"""
    actions = []
    for robot, tgtid in zip(obs["current_robot"], obs["robot_target_idx"]):
        actions.append(update_step(robot[None], obs["future_target"][tgtid]))
    action = {}
    for key_ in actions[0]:
        action[key_] = np.stack([a[key_] for a in actions], axis=0)
    return action


def run_sim(env: RobotChasingTargetEnv, planner, max_step: int, seed: int):
    """Run stimulation until terminated/truncated or max_step"""
    observation, _ = env.reset(seed=seed)

    steps = 0
    done = False
    while not done:
        action = planner(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if env.render_mode == "human":
            env.render()
        done = terminated or truncated
        steps += 1
        if steps > max_step:
            break


app = typer.Typer()


@app.command()
def main(
    profile: Annotated[bool, typer.Option(help="Enable Scalene Profiling")] = False,
    max_step: Annotated[int, typer.Option(help="Max step before termination")] = 500,
    record: Annotated[Optional[str], typer.Option(help="Filename to record")] = None,
    use_pure_pursuit: Annotated[bool, typer.Option(help="Greedy algorithm")] = False,
    seed: Annotated[int, typer.Option(help="Seed for simulation")] = 0,
):
    """
    Runs simulation of target chasers. When profiling is disabled, it will visualize the game.
    Use --cpu and --off when profiling: `scalene --cpu --off run.py --profile`
    """
    env = RobotChasingTargetEnv(
        render_mode="human" if not profile else None,
        n_robots=15,
        n_targets=4,
        robot_radius=ROBOT_RADIUS,
        max_velocity=MAX_VEL,
        target_velocity_std=MAX_VEL,
        recording_path=Path.cwd() / record if record is not None else None,
    )

    if use_pure_pursuit:
        planner = pure_pursuit
    else:
        planner = Planner(
            env.get_wrapper_attr("robot_radius"),
            env.get_wrapper_attr("dt"),
            env.get_wrapper_attr("max_velocity"),
        )

    if profile:
        assert record is None, "Can't record and profile"
        from scalene import scalene_profiler

        scalene_profiler.start()

    for _ in range(10 if profile else 1):
        run_sim(env, planner, max_step, seed)

    if profile:
        scalene_profiler.stop()

    env.close()


def _main():
    app()


if __name__ == "__main__":
    _main()
