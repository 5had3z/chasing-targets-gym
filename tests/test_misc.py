from pathlib import Path

import pytest
from gymnasium import make
import numpy as np
import chasing_targets_gym


def test_init():
    """Test basic simulation can be built and stepped"""
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        barrier_velocity_range=0.5,
        max_episode_steps=30,
    )
    env.reset()
    env.step({"vL": np.full((10, 1), 0.0), "vR": np.full((10, 1), 0.0)})
    env.close()


def test_limits():
    """Test to ensure that sim catches invalid actions"""
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        barrier_velocity_range=0.5,
        max_episode_steps=30,
    )
    env.reset()
    # Within limits
    env.step({"vL": np.full((10, 1), 0.5), "vR": np.full((10, 1), -0.5)})

    # Too big
    with pytest.raises(AssertionError):
        env.step({"vL": np.full((10, 1), 0.6), "vR": np.full((10, 1), 0.0)})

    # Too small
    with pytest.raises(AssertionError):
        env.step({"vL": np.full((10, 1), 0.0), "vR": np.full((10, 1), -0.6)})

    env.close()


def test_video_writer(tmp_path: Path):
    """Test I can write a video of the simulation"""

    vid_path = tmp_path / "test.mkv"
    env = make(
        "ChasingTargets-v0",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        barrier_velocity_range=0.5,
        max_episode_steps=30,
        recording_path=vid_path,
    )
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(
            {"vL": np.full((10, 1), 0.0), "vR": np.full((10, 1), 0.0)}
        )
        env.render()
        done = terminated or truncated
    env.close()

    assert vid_path.exists(), "Video not written"
    assert (
        vid_path.stat().st_size > 1024
    ), f"Insufficent data written: {vid_path.stat().st_size} bytes"
