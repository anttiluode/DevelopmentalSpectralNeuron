from __future__ import annotations
from typing import Literal
import numpy as np
from .config import V0Config

HistoryId = Literal["H_A", "H_B"]


def epoch_schedule(history: HistoryId, config: V0Config) -> tuple[str, ...]:
    if history == "H_A":
        return tuple("LR"[i % 2] for i in range(config.epochs))
    if history == "H_B":
        half = config.epochs // 2
        return tuple(["L"] * half + ["R"] * (config.epochs - half))
    raise ValueError(f"unknown history {history!r}")


class DevelopmentalWorld:
    def __init__(self, config: V0Config):
        self.config = config

    def _gaussian(self, pos: np.ndarray, center: tuple[float, float]) -> tuple[float, np.ndarray]:
        p = np.asarray(pos, dtype=float)
        c = np.asarray(center, dtype=float)
        d = p - c
        s2 = self.config.target_sigma ** 2
        value = float(np.exp(-0.5 * float(d @ d) / s2))
        grad = -d / s2 * value
        return value, grad

    def _boundary_repulsion(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        p = np.asarray(pos, dtype=float)
        margin = 0.08
        value = 0.0
        grad = np.zeros(2, dtype=float)
        # Smooth quadratic soft wall, symmetric in x and y.
        for axis in range(2):
            if p[axis] < margin:
                q = (margin - p[axis]) / margin
                value -= q * q
                grad[axis] += 2.0 * q / margin
            elif p[axis] > 1.0 - margin:
                q = (p[axis] - (1.0 - margin)) / margin
                value -= q * q
                grad[axis] -= 2.0 * q / margin
        return value, grad

    def cue_and_gradient(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        lv, lg = self._gaussian(pos, self.config.left_target)
        rv, rg = self._gaussian(pos, self.config.right_target)
        bv, bg = self._boundary_repulsion(pos)
        return lv + rv + 0.08 * bv, lg + rg + 0.08 * bg

    def active_side(self, step: int, history: HistoryId) -> str:
        epoch = max(int(step), 0) // self.config.steps_per_epoch
        epoch = epoch % self.config.epochs
        return epoch_schedule(history, self.config)[epoch]

    def resource_and_gradient(self, pos: np.ndarray, step: int, history: HistoryId) -> tuple[float, np.ndarray]:
        center = self.config.left_target if self.active_side(step, history) == "L" else self.config.right_target
        return self._gaussian(pos, center)

    def stimulus(self, pos: np.ndarray, step: int, history: HistoryId) -> float:
        resource, _ = self.resource_and_gradient(pos, step, history)
        # Frequency follows the spatial side, not the history label itself.
        side = self.active_side(step, history)
        omega = 0.12 if side == "L" else 0.26
        return float(resource * np.sin(omega * step))
