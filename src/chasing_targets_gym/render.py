import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

try:
    import cv2
except ImportError:
    warn("Unable to import cv2, simulation video writer will be available")

import numpy as np
import pygame

from .robots import Robots

WIDTH = 1500
HEIGHT = 1000

SIZE = (WIDTH, HEIGHT)
BLACK = (20, 20, 40)
LIGHTBLUE = (0, 120, 255)
DARKBLUE = (0, 40, 160)
RED = (255, 100, 0)
GREEN = (46, 125, 50)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREY = (70, 70, 70)
K = 160


def to_display(x: float, y: float) -> tuple[int, int]:
    """Transform simulation coordinate to display coordinate"""
    disp_center = np.array([WIDTH / 2, HEIGHT / 2])
    center_tf = (disp_center + K * np.array([x, -y])).astype(np.int32)
    return tuple(center_tf)


@dataclass(slots=True)
class DecayingMarker:
    """Marker that decays after a period of time"""

    position: tuple[int, int]
    timestamp: float = -1.0
    decay: float = 2.0

    def __post_init__(self):
        self.timestamp = time.time()

    def expired(self) -> bool:
        """Marker is expired"""
        return time.time() - self.timestamp > self.decay


class PyGameRecorder:
    """Records pygame screen to video file"""

    def __init__(self, filename: Path, video_size: tuple[int, int], fps: float) -> None:
        assert filename.parent.exists(), "Video destination path does not exist"
        self.video_writer = cv2.VideoWriter(
            str(filename), cv2.VideoWriter_fourcc(*"MJPG"), fps, video_size
        )
        assert self.video_writer.isOpened(), "Error opening video writer"

        self.is_closed = False

    def __call__(self, screen: pygame.Surface):
        frame = pygame.surfarray.array3d(screen)
        frame = cv2.cvtColor(frame.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def close(self):
        self.is_closed = True
        self.video_writer.release()


def prepare_trajectory_render(
    x: float, y: float, theta: float, vL: float, vR: float, width: float, dt: float
):
    if np.allclose(vL, vR, atol=1e-3):
        return vL * dt
    if np.allclose(vL, -vR, atol=1e-3):
        return 0.0

    R = width / 2.0 * (vR + vL) / (vR - vL)
    dtheta = (vR - vL) * dt / width
    cx, cy = x - R * np.sin(theta), y + R * np.cos(theta)

    Rabs = abs(R)
    tlx, tly = to_display(cx - Rabs, cy + Rabs)
    Rx, Ry = int(K * (2 * Rabs)), int(K * (2 * Rabs))

    start_angle = theta - np.pi / 2.0 if R > 0 else theta + np.pi / 2.0
    stop_angle = start_angle + dtheta
    return ((tlx, tly), (Rx, Ry)), start_angle, stop_angle


def draw_robot(
    screen: pygame.Surface,
    wheel_blob: float,
    state: np.ndarray,
    radius: float,
    dt: float,
) -> None:
    """Draw individual robot"""
    _idxs = [Robots.lbl2idx[l] for l in ["x", "y", "t", "vL", "vR"]]
    x, y, theta, vL, vR = state[_idxs]
    xy = np.stack([x, y], axis=-1)
    pygame.draw.circle(screen, WHITE, to_display(*xy), int(K * radius), 3)

    diff = radius * np.array([-np.sin(theta), np.cos(theta)])
    wlxy = xy + diff
    pygame.draw.circle(screen, BLUE, to_display(*wlxy), int(K * wheel_blob))
    wlxy = xy - diff
    pygame.draw.circle(screen, BLUE, to_display(*wlxy), int(K * wheel_blob))

    path = prepare_trajectory_render(x, y, theta, vL, vR, 2 * radius, dt)

    if isinstance(path, float):
        line_start = to_display(*xy)
        line_end = to_display(x + path * np.cos(theta), y + path * np.sin(theta))
        pygame.draw.line(screen, (0, 200, 0), line_start, line_end, 1)
    else:
        start_angle = min(path[2:])
        stop_angle = max(path[2:])

        if start_angle < 0:
            start_angle += 2 * np.pi
            stop_angle += 2 * np.pi

        if path[0][0][0] > 0 and path[0][1][0] > 0 and path[0][1][1] > 1:
            pygame.draw.arc(screen, (0, 200, 0), path[0], start_angle, stop_angle, 1)


def draw_robots(
    screen: pygame.Surface,
    wheel_blob: float,
    robots: Robots,
    history: deque[np.ndarray] | None = None,
):
    if history is not None:
        for robots_pos in history:
            for pos in robots_pos:
                pygame.draw.circle(screen, GREY, to_display(*pos), 3, 0)

    for robot in robots.state.T:
        draw_robot(screen, wheel_blob, robot, robots.radius, robots.dt)
