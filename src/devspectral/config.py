from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class V0Config:
    grid_size: int = 64
    domain_min: float = 0.0
    domain_max: float = 1.0
    soma: tuple[float, float] = (0.50, 0.08)
    left_target: tuple[float, float] = (0.25, 0.78)
    right_target: tuple[float, float] = (0.75, 0.78)
    target_sigma: float = 0.11
    epochs: int = 12
    steps_per_epoch: int = 50
    initial_tips: int = 4
    max_active_tips: int = 48
    node_spacing: float = 0.025
    merge_radius: float = 0.035
    tip_step: float = 0.012
    field_decay: float = 0.997
    resonator_r: tuple[float, ...] = (0.86, 0.90, 0.94, 0.97)
    resonator_omega: tuple[float, ...] = (0.08, 0.12, 0.18, 0.26)
    resource_initial: float = 1.0
    resource_floor: float = 0.0
    resource_cap: float = 2.0
    maintenance_cost: float = 0.0015
    activity_cost: float = 0.0008
    harvest_gain: float = 0.012
    branch_threshold: float = 1.35
    min_productive_age: int = 35
    daughter_heading_delta: float = 0.35
    edge_initial_weight: float = 0.30
    edge_reinforcement: float = 0.010
    edge_decay: float = 0.9995
    prune_threshold: float = 0.06
    prune_grace: int = 120
    field_weight: float = 0.35
    cue_weight: float = 1.0
    resource_weight: float = 0.55
    curvature_weight: float = 0.15
    exploration_sd: float = 0.08
    diffusion_beta: float = 0.75
    diffusion_times: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    spectral_modes: int = 6
    lesion_x: float = 0.50
    lesion_y_min: float = 0.55
    regrowth_steps: int = 200

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


V0 = V0Config()
