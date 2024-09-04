from pathlib import Path

import numpy as np
import pytest
from gymnasium import make

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
    env.step(
        {
            "vL": np.full((10, 1), 0.0, dtype=np.float32),
            "vR": np.full((10, 1), 0.0, dtype=np.float32),
        }
    )
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
    env.step(
        {
            "vL": np.full((10, 1), 0.5, dtype=np.float32),
            "vR": np.full((10, 1), -0.5, dtype=np.float32),
        }
    )

    # Too big
    with pytest.raises(AssertionError):
        env.step(
            {
                "vL": np.full((10, 1), 0.6, dtype=np.float32),
                "vR": np.full((10, 1), 0.0, dtype=np.float32),
            }
        )

    # Too small
    with pytest.raises(AssertionError):
        env.step(
            {
                "vL": np.full((10, 1), 0.0, dtype=np.float32),
                "vR": np.full((10, 1), -0.6, dtype=np.float32),
            }
        )

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
            {
                "vL": np.full((10, 1), 0.0, dtype=np.float32),
                "vR": np.full((10, 1), 0.0, dtype=np.float32),
            }
        )
        env.render()
        done = terminated or truncated
    env.close()

    assert vid_path.exists(), "Video not written"
    assert (
        vid_path.stat().st_size > 1024
    ), f"Insufficent data written: {vid_path.stat().st_size} bytes"
