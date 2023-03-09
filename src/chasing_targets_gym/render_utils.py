from dataclasses import dataclass
from pathlib import Path
import time
from typing import Tuple
from warnings import warn

try:
    import cv2
except ImportError:
    warn("Unable to import cv2, simulation video writer will be available")

import numpy as np
import pygame

WIDTH = 1500
HEIGHT = 1000

size = (WIDTH, HEIGHT)
black = (20, 20, 40)
lightblue = (0, 120, 255)
darkblue = (0, 40, 160)
red = (255, 100, 0)
green = (46, 125, 50)
white = (255, 255, 255)
blue = (0, 0, 255)
grey = (70, 70, 70)
k = 160


def to_display(x: float, y: float) -> Tuple[int, int]:
    """Transform simulation coordinate to display coordinate"""
    disp_center = np.array([WIDTH / 2, HEIGHT / 2])
    center_tf = (disp_center + k * np.array([x, -y])).astype(np.int32)
    return tuple(center_tf)


@dataclass
class DecayingMarker:
    position: Tuple[int, int]
    timestamp: float = -1.0
    decay: float = 2.0

    def __post_init__(self):
        self.timestamp = time.time()

    def expired(self) -> bool:
        return time.time() - self.timestamp > self.decay


class PyGameRecorder:
    def __init__(self, filename: Path, size: Tuple[int, int], fps: float) -> None:
        assert filename.parent.exists(), "Video destination path does not exist"
        self.video_writer = cv2.VideoWriter(
            str(filename), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
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
