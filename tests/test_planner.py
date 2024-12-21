from chasing_targets_gym import _planner
from chasing_targets_gym import py_planner
from gymnasium import make
import pytest
import numpy as np


@pytest.mark.parametrize("n_robots", [10, 5])
@pytest.mark.parametrize("n_targets", [3, 6, 9])
@pytest.mark.parametrize("seed", [123, 456, 789])
def test_cpp_py_planners(n_robots: int, n_targets: int, seed: int):
    env = make(
        "ChasingTargets-v0",
        n_robots=n_robots,
        n_targets=n_targets,
        robot_radius=0.1,
        max_velocity=0.5,
        target_velocity_std=0.5,
        max_episode_steps=60,
        enforce_spaces=True,
    )
    pylanner = py_planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
        use_batched=False,
    )
    bpylanner = py_planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
        use_batched=True,
    )
    cpplanner = _planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
    )

    observation, _ = env.reset(seed=seed)
    done = False
    while not done:
        p_action = pylanner(observation)
        b_action = bpylanner(observation)
        c_action = cpplanner(observation)
        ok = all(np.allclose(b_action[k], p_action[k]) for k in c_action)
        ok &= all(np.allclose(p_action[k], c_action[k]) for k in c_action)
        if not ok:
            raise AssertionError("Not ok!")
        observation, _, terminated, truncated, _ = env.step(c_action)
        done = terminated or truncated


@pytest.mark.parametrize("seed", [0, 15])
def test_restarting(seed: int):
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=8,
        robot_radius=0.1,
        max_velocity=0.5,
        target_velocity_std=0.5,
        max_episode_steps=90,
        enforce_spaces=True,
    )
    cpplanner = _planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
    )

    for i in range(10):
        observation, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            c_action = cpplanner(observation)
            observation, _, terminated, truncated, _ = env.step(c_action)
            done = terminated or truncated
