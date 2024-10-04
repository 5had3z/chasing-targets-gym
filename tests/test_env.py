from pathlib import Path

import numpy as np
import pytest
from gymnasium import Env, make

import chasing_targets_gym

_DEFAULT_MAX_VEL = 0.5


@pytest.fixture
def default_env():
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=_DEFAULT_MAX_VEL,
        target_velocity_std=0.5,
        max_episode_steps=30,
    )
    return env


def test_init(default_env: Env):
    """Test basic simulation can be built and stepped"""
    n_robots = default_env.get_wrapper_attr("n_robots")
    default_env.reset()
    default_env.step(
        {
            "vL": np.full(n_robots, 0.0, dtype=np.float32),
            "vR": np.full(n_robots, 0.0, dtype=np.float32),
        }
    )
    default_env.close()


def test_observation_space(default_env: Env):
    """Run simulation and check observation space is always valid"""
    obs, _ = default_env.reset(seed=2)

    done = False
    while not done:
        assert default_env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = default_env.step(
            default_env.action_space.sample()
        )
        done = terminated or truncated
    default_env.close()


def test_action_space(default_env: Env):
    """Test to ensure that sim catches invalid actions"""
    n_robot = default_env.get_wrapper_attr("n_robots")
    default_env.reset()

    # Within limits
    default_env.step(
        {
            "vL": np.full(n_robot, _DEFAULT_MAX_VEL, dtype=np.float32),
            "vR": np.full(n_robot, -_DEFAULT_MAX_VEL, dtype=np.float32),
        }
    )

    # Too big
    with pytest.raises(AssertionError):
        default_env.step(
            {
                "vL": np.full(n_robot, _DEFAULT_MAX_VEL + 0.1, dtype=np.float32),
                "vR": np.full(n_robot, 0.0, dtype=np.float32),
            }
        )

    # Too small
    with pytest.raises(AssertionError):
        default_env.step(
            {
                "vL": np.full(n_robot, 0.0, dtype=np.float32),
                "vR": np.full(n_robot, -_DEFAULT_MAX_VEL - 0.1, dtype=np.float32),
            }
        )

    default_env.close()


def test_video_writer(tmp_path: Path):
    """Test writing a video of the simulation"""
    vid_path = tmp_path / "test.mkv"
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        target_velocity_std=0.5,
        max_episode_steps=30,
        recording_path=vid_path,
    )
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(
            {
                "vL": np.full(10, 0.0, dtype=np.float32),
                "vR": np.full(10, 0.0, dtype=np.float32),
            }
        )
        env.render()
        done = terminated or truncated
    env.close()

    assert vid_path.exists(), "Video not written"
    assert (
        vid_path.stat().st_size > 1024
    ), f"Insufficent data written: {vid_path.stat().st_size} bytes"
