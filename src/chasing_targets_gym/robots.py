import numpy as np


def wrap_angle(angle):
    """Ensure angle is in range [-pi,pi]"""
    angle[angle > np.pi] -= 2 * np.pi
    angle[angle < -np.pi] += 2 * np.pi
    return angle


class Robots:
    """Class that contains a set of moving robots"""

    # So I don't have to remember indexes and prevent bugs
    ax_lbl = ["x", "y", "t", "dx", "dy", "dt", "vL", "vR"]
    lbl2idx = {l: i for i, l in enumerate(ax_lbl)}

    def __init__(self, n_robots: int, radius: float, dt: float, accel_limit: float):
        self.state = np.zeros((8, n_robots), dtype=np.float32)
        self.accel_limit = accel_limit
        self.dt = dt
        self.radius = radius

    def __len__(self):
        return self.state.shape[1]

    def reset(self):
        self.state.fill(0)

    @property
    def width(self) -> float:
        return 2 * self.radius

    @property
    def x(self) -> np.ndarray:
        return self.state[0]

    @x.setter
    def x(self, value):
        self.state[0] = value

    @property
    def y(self) -> np.ndarray:
        return self.state[1]

    @y.setter
    def y(self, value):
        self.state[1] = value

    @property
    def theta(self) -> np.ndarray:
        return self.state[2]

    @theta.setter
    def theta(self, value):
        self.state[2] = value

    @property
    def vL(self) -> np.ndarray:
        return self.state[-2]

    @vL.setter
    def vL(self, value):
        self.state[-2] = value

    @property
    def vR(self) -> np.ndarray:
        return self.state[-1]

    @vR.setter
    def vR(self, value):
        self.state[-1] = value

    def step(self, action: dict[str, np.ndarray]) -> None:
        """Perform control action"""
        # Update intended control inputs
        max_dv = self.accel_limit * self.dt
        self.vL = np.clip(action["vL"], self.vL - max_dv, self.vL + max_dv)
        self.vR = np.clip(action["vR"], self.vR - max_dv, self.vR + max_dv)

        # Calculate rate of change
        dxdyxt = self._calculate_velocity()

        # Update state
        self.state[:3] += self.dt * dxdyxt
        self.state[3:6] = dxdyxt
        self.state[2] = wrap_angle(self.state[2])

    def forecast(self, dt: float | None = None) -> np.ndarray:
        """Predict the future state given the current state

        Args:
            dt (float | None, optional): Timestep to linearly forecast. Defaults to None.

        Returns:
            np.ndarray: Predicted state dt into the future.
        """
        dt = self.dt if dt is None else dt
        dxdydt = self._calculate_velocity()
        pred = self.state[:6].copy()
        pred[:3] += dxdydt * dt
        pred[3:] = dxdydt
        pred[2] = wrap_angle(pred[2])
        return pred

    def _calculate_velocity(self) -> np.ndarray:
        theta = self.theta
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        vR = self.vR
        vL = self.vL

        dxdydt = np.empty([3, len(self)], dtype=np.float32)

        # First cover general motion case
        vDiff = vR - vL
        R = (self.radius * (vR + vL)) / (vDiff + np.finfo(vR.dtype).eps)
        np.multiply(vDiff, 1 / self.width, dxdydt[2])
        np.multiply(R, (np.sin(dxdydt[2] + theta) - sin_th), dxdydt[0])
        np.multiply(-R, (np.cos(dxdydt[2] + theta) - cos_th), dxdydt[1])

        # Then cover straight motion case
        mask = np.abs(vDiff) < 1e-3
        dxdydt[0, mask] = (vL * cos_th)[mask]
        dxdydt[1, mask] = (vL * sin_th)[mask]

        return dxdydt
