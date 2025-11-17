import numpy as np


class PID:
    def __init__(
        self,
        kp: float = 0.4,
        ki: float = 0.0,
        kd: float = 0.05,
        clamp: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.clamp = clamp
        self.ei = 0.0
        self.prev = 0.0

    def step(self, target: float, current: float, dt: float = 0.02) -> float:
        e = target - current
        self.ei += e * dt
        de = (e - self.prev) / dt if dt > 0 else 0.0
        u = self.kp * e + self.ki * self.ei + self.kd * de
        self.prev = e
        return float(np.clip(u, *self.clamp))
