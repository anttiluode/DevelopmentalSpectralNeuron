from __future__ import annotations
import numpy as np
from .config import V0Config


class StructuralField:
    def __init__(self, config: V0Config):
        self.config = config
        self.data = np.zeros((config.grid_size, config.grid_size), dtype=float)

    def copy(self) -> 'StructuralField':
        other = StructuralField(self.config)
        other.data = self.data.copy()
        return other

    def decay(self) -> None:
        self.data *= self.config.field_decay

    def _xy(self, pos: np.ndarray) -> tuple[float, float]:
        p = np.clip(np.asarray(pos, dtype=float), 0.0, 1.0)
        scale = self.config.grid_size - 1
        return float(p[0] * scale), float(p[1] * scale)

    def _sample_value(self, pos: np.ndarray) -> float:
        x, y = self._xy(pos)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, self.config.grid_size - 1), min(y0 + 1, self.config.grid_size - 1)
        tx, ty = x - x0, y - y0
        a = self.data[y0, x0] * (1 - tx) + self.data[y0, x1] * tx
        b = self.data[y1, x0] * (1 - tx) + self.data[y1, x1] * tx
        return float(a * (1 - ty) + b * ty)

    def sample_and_gradient(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        p = np.clip(np.asarray(pos, dtype=float), 0.0, 1.0)
        eps = 1.0 / max(2, self.config.grid_size - 1)
        value = self._sample_value(p)
        px1 = p.copy(); px1[0] = min(1.0, px1[0] + eps)
        px0 = p.copy(); px0[0] = max(0.0, px0[0] - eps)
        py1 = p.copy(); py1[1] = min(1.0, py1[1] + eps)
        py0 = p.copy(); py0[1] = max(0.0, py0[1] - eps)
        dx_denom = max(px1[0] - px0[0], 1e-12)
        dy_denom = max(py1[1] - py0[1], 1e-12)
        grad = np.array([
            (self._sample_value(px1) - self._sample_value(px0)) / dx_denom,
            (self._sample_value(py1) - self._sample_value(py0)) / dy_denom,
        ])
        return value, grad

    def deposit_segment(self, p0: np.ndarray, p1: np.ndarray, amount: float) -> None:
        if amount <= 0:
            return
        p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(dist * self.config.grid_size * 2)))
        radius = 2
        sigma2 = 1.2 ** 2
        for a in np.linspace(0.0, 1.0, n):
            p = (1 - a) * p0 + a * p1
            x, y = self._xy(p)
            ix, iy = int(round(x)), int(round(y))
            for yy in range(max(0, iy-radius), min(self.config.grid_size, iy+radius+1)):
                for xx in range(max(0, ix-radius), min(self.config.grid_size, ix+radius+1)):
                    d2 = (xx-x)**2 + (yy-y)**2
                    self.data[yy, xx] += float(amount) * np.exp(-0.5*d2/sigma2) / n
