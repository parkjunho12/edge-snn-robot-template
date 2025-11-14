import numpy as np


class PID:
    def __init__(self, kp=0.4, ki=0.0, kd=0.05, clamp=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.clamp = clamp
        self.ei = 0.0
        self.prev = 0.0

    def step(self, target, current, dt=0.02):
        e = target - current
        self.ei += e * dt
        de = (e - self.prev) / dt if dt > 0 else 0.0
        u = self.kp * e + self.ki * self.ei + self.kd * de
        self.prev = e
        return float(np.clip(u, *self.clamp))
