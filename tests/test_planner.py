from chasing_targets_gym import _planner
from chasing_targets_gym import py_planner
from gymnasium import make
import numpy as np


def test_cpp_py_planners():
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        target_velocity_std=0.5,
        max_episode_steps=30,
    )
    pylanner = py_planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
    )
    cpplanner = _planner.Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
    )

    observation, _ = env.reset(seed=0)
    done = False
    while not done:
        p_action = pylanner(observation)
        c_action = cpplanner(observation)
        assert all(np.allclose(p_action[k], c_action[k]) for k in p_action)
        observation, _, terminated, truncated, _ = env.step(p_action)
        done = terminated or truncated
