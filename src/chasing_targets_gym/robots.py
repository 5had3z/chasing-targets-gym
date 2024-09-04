from collections import deque

import numpy as np
import pygame

from . import render_utils as ru


class Robots:
    """Class that contains a set of moving robots"""

    # So I don't have to remember indexes and prevent bugs
    _ax_lbl = ["x", "y", "t", "dx", "dy", "dt", "vL", "vR"]
    _l2i = {l: i for i, l in enumerate(_ax_lbl)}

    def __init__(self, n_robots: int, radius: float, dt: float, accel_limit: float):
        self.state = np.zeros((n_robots, 8), dtype=np.float32)
        self.accel_limit = accel_limit
        self.dt = dt
        self.radius = radius
        self.history = [deque() for _ in range(n_robots)]

    def __len__(self):
        return self.state.shape[0]

    def reset(self):
        self.state.fill(0)
        for h in self.history:
            h.clear()

    @property
    def width(self) -> float:
        return 2 * self.radius

    @property
    def x(self) -> np.ndarray:
        return self.state[:, Robots._l2i["x"]]

    @x.setter
    def x(self, value):
        self.state[:, Robots._l2i["x"]] = value

    @property
    def y(self) -> np.ndarray:
        return self.state[:, Robots._l2i["y"]]

    @y.setter
    def y(self, value):
        self.state[:, Robots._l2i["y"]] = value

    @property
    def theta(self) -> np.ndarray:
        return self.state[:, Robots._l2i["t"]]

    @theta.setter
    def theta(self, value):
        self.state[:, Robots._l2i["t"]] = value

    @property
    def vL(self) -> np.ndarray:
        return self.state[:, Robots._l2i["vL"]]

    @vL.setter
    def vL(self, value):
        self.state[:, Robots._l2i["vL"]] = value

    @property
    def vR(self) -> np.ndarray:
        return self.state[:, Robots._l2i["vR"]]

    @vR.setter
    def vR(self, value):
        self.state[:, Robots._l2i["vR"]] = value

    def step(self, action: dict[str, np.ndarray]) -> None:
        """Perform control action"""
        # Add state history
        for rhist, state in zip(self.history, self.state):
            rhist.append(tuple(state[:2]))
            if len(rhist) > 10:
                rhist.popleft()

        # Update intended control inputs
        max_dv = self.accel_limit * self.dt
        self.vL = np.clip(action["vL"][:, 0], self.vL - max_dv, self.vL + max_dv)
        self.vR = np.clip(action["vR"][:, 0], self.vR - max_dv, self.vR + max_dv)

        # Calculate rate of change
        dxdyxt = self._calculate_velocity()

        # Update state
        self.state[:, :3] += self.dt * dxdyxt
        self.state[:, 3:6] = dxdyxt

    def forecast(self, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        dxdydt = self._calculate_velocity()
        pred = self.state[:, :6].copy()
        pred[:, :3] += dxdydt * dt
        pred[:, 3:] = dxdydt
        return pred

    def _calculate_velocity(self) -> np.ndarray:
        theta = self.theta
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        vR = self.vR
        vL = self.vL

        # First cover general motion case
        R = (self.radius * (vR + vL)) / (vR - vL + np.finfo(vR.dtype).eps)
        dt = (vR - vL) / self.width
        dx = R * (np.sin(dt + theta) - sin_th)
        dy = -R * (np.cos(dt + theta) - cos_th)

        # Then cover straight motion case
        mask = np.abs(vL - vR) < 1e-3
        dx[mask] = (vL * cos_th)[mask]
        dy[mask] = (vL * sin_th)[mask]

        return np.stack([dx, dy, dt], axis=-1)

    def _prepare_trajectory_render(
        self, x: float, y: float, theta: float, vL: float, vR: float
    ):
        if np.allclose(vL, vR, atol=1e-3):
            return vL * self.dt
        if np.allclose(vL, -vR, atol=1e-3):
            return 0.0

        R = self.width / 2.0 * (vR + vL) / (vR - vL)
        dtheta = (vR - vL) * self.dt / self.width
        cx, cy = x - R * np.sin(theta), y + R * np.cos(theta)

        Rabs = abs(R)
        tlx, tly = ru.to_display(cx - Rabs, cy + Rabs)
        Rx, Ry = int(ru.k * (2 * Rabs)), int(ru.k * (2 * Rabs))

        start_angle = theta - np.pi / 2.0 if R > 0 else theta + np.pi / 2.0
        stop_angle = start_angle + dtheta
        return ((tlx, tly), (Rx, Ry)), start_angle, stop_angle

    def _draw(
        self, screen: pygame.Surface, wheel_blob: float, state: np.ndarray
    ) -> None:
        """Draw individual robot"""
        _idxs = [self._l2i[l] for l in ["x", "y", "t", "vL", "vR"]]
        x, y, theta, vL, vR = state[_idxs]
        xy = np.stack([x, y], axis=-1)
        pygame.draw.circle(
            screen, ru.white, ru.to_display(*xy), int(ru.k * self.radius), 3
        )

        diff = self.radius * np.array([-np.sin(theta), np.cos(theta)])
        wlxy = xy + diff
        pygame.draw.circle(
            screen, ru.blue, ru.to_display(*wlxy), int(ru.k * wheel_blob)
        )
        wlxy = xy - diff
        pygame.draw.circle(
            screen, ru.blue, ru.to_display(*wlxy), int(ru.k * wheel_blob)
        )

        path = self._prepare_trajectory_render(x, y, theta, vL, vR)

        if isinstance(path, float):
            line_start = ru.to_display(*xy)
            line_end = ru.to_display(x + path * np.cos(theta), y + path * np.sin(theta))
            pygame.draw.line(screen, (0, 200, 0), line_start, line_end, 1)

        else:
            start_angle = min(path[2:])
            stop_angle = max(path[2:])

            if start_angle < 0:
                start_angle += 2 * np.pi
                stop_angle += 2 * np.pi

            if path[0][0][0] > 0 and path[0][1][0] > 0 and path[0][1][1] > 1:
                pygame.draw.arc(
                    screen, (0, 200, 0), path[0], start_angle, stop_angle, 1
                )

    def draw(self, screen: pygame.Surface, wheel_blob: float):
        """Draw robots on screen"""
        for history in self.history:
            for pos in history:
                pygame.draw.circle(screen, ru.grey, ru.to_display(*pos), 3, 0)

        for robot in self.state:
            self._draw(screen, wheel_blob, robot)
