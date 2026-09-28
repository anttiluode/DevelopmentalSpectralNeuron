from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .config import V0Config
from .field import StructuralField
from .graph import GraphState
from .world import DevelopmentalWorld, HistoryId


def resonator_update(q: np.ndarray, stimulus: float, r: float, omega: float, input_gain: float = 1.0) -> np.ndarray:
    c, s = math.cos(omega), math.sin(omega)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    drive = np.array([input_gain * stimulus, 0.0], dtype=float)
    return r * (rot @ np.asarray(q, dtype=float)) + (1.0 - r) * drive


@dataclass
class GrowthTip:
    tip_id: int
    node_id: int
    position: np.ndarray
    heading: float
    q: np.ndarray
    resource: float
    age: int = 0
    productive_age: int = 0
    parent_tip_id: int | None = None
    alive: bool = True
    r: float = 0.9
    omega: float = 0.12
    input_gain: float = 1.0


class DevelopmentSimulator:
    def __init__(self, config: V0Config, history: HistoryId, seed: int, control: str = 'full'):
        self.config = config
        self.history = history
        self.seed = int(seed)
        self.control = control
        self.rng = np.random.default_rng(self.seed)
        self.world = DevelopmentalWorld(config)
        self.graph = GraphState(np.array(config.soma, dtype=float))
        self.field = StructuralField(config)
        self.step_index = 0
        self.regrowth_mode = False
        self.tips: list[GrowthTip] = []
        offsets = np.linspace(-0.45, 0.45, config.initial_tips)
        for i, off in enumerate(offsets):
            self.tips.append(self._new_tip(i, self.graph.soma_node_id, np.array(config.soma), math.pi/2 + float(off), None))
        self._next_tip_id = len(self.tips)

    def _new_tip(self, tip_id: int, node_id: int, position: np.ndarray, heading: float, parent: int | None) -> GrowthTip:
        return GrowthTip(
            tip_id=tip_id, node_id=node_id, position=np.asarray(position, float).copy(), heading=float(heading),
            q=np.zeros(2, dtype=float), resource=self.config.resource_initial, parent_tip_id=parent,
            r=self.config.resonator_r[tip_id % len(self.config.resonator_r)],
            omega=self.config.resonator_omega[tip_id % len(self.config.resonator_omega)],
        )

    def _direction(self, tip: GrowthTip, resource_grad: np.ndarray, cue_grad: np.ndarray) -> np.ndarray:
        heading_vec = np.array([math.cos(tip.heading), math.sin(tip.heading)])
        _, field_grad = self.field.sample_and_gradient(tip.position)
        field_grad = np.asarray(field_grad, dtype=float)
        cue_grad = np.asarray(cue_grad, dtype=float)
        resource_grad = np.asarray(resource_grad, dtype=float)

        def bounded_gradient(g: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(g))
            return g / (1.0 + norm)

        # Preserve local signal magnitude for weak cues while bounding very steep fields.
        # This avoids turning numerically tiny far-field gradients into full-strength oracles.
        field_grad = bounded_gradient(field_grad)
        cue_grad = bounded_gradient(cue_grad)
        resource_grad = bounded_gradient(resource_grad)
        noise = self.rng.normal(0.0, self.config.exploration_sd, size=2)
        field_term = np.zeros(2) if self.control == 'no_stigmergy' else self.config.field_weight * field_grad
        vec = (
            self.config.cue_weight * cue_grad
            + self.config.resource_weight * resource_grad
            + field_term
            + self.config.curvature_weight * heading_vec
            + noise
        )
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-12:
            return heading_vec
        return vec / norm

    def _harvest(self, activity: float, opportunity: float) -> float:
        if self.control == 'no_metabolic_selection':
            return self.config.maintenance_cost + self.config.activity_cost * activity + 0.001
        return self.config.harvest_gain * activity * opportunity

    def _active_tips(self) -> list[GrowthTip]:
        return [t for t in self.tips if t.alive]

    def _maybe_branch(self, tip: GrowthTip, new_tips: list[GrowthTip]) -> None:
        if len(self._active_tips()) + len(new_tips) >= self.config.max_active_tips:
            return
        if tip.resource < self.config.branch_threshold or tip.productive_age < self.config.min_productive_age:
            return
        delta = self.config.daughter_heading_delta * (1 if self.rng.random() >= 0.5 else -1)
        child = self._new_tip(self._next_tip_id, tip.node_id, tip.position, tip.heading + delta, tip.tip_id)
        available = tip.resource
        tip.resource = 0.5 * available
        child.resource = 0.5 * available
        tip.productive_age = 0
        self._next_tip_id += 1
        new_tips.append(child)

    def _protected_edges(self) -> set[tuple[int, int]]:
        protected: set[tuple[int, int]] = set()
        for tip in self._active_tips():
            path = self.graph.path_to_soma(tip.node_id)
            if path:
                protected.update(path)
        return protected

    def _prune_edges(self) -> None:
        if self.control == 'no_pruning':
            return
        protected = self._protected_edges()
        to_remove = []
        for key, edge in self.graph.edges.items():
            if edge.weight < self.config.prune_threshold and edge.age >= self.config.prune_grace and key not in protected:
                to_remove.append(key)
        for u, v in to_remove:
            self.graph.remove_edge(u, v)

    def _structural_tick(self) -> None:
        for edge in self.graph.edges.values():
            edge.age += 1
            edge.weight *= self.config.edge_decay
            edge.traffic *= 0.98
        self._prune_edges()
        self.field.decay()

    def _record_growth_segment(self, tip: GrowthTip, p0: np.ndarray, p1: np.ndarray, activity: float, productive: bool) -> None:
        resource_fraction = (tip.resource - self.config.resource_floor) / max(1e-12, self.config.resource_cap - self.config.resource_floor)
        activity_gate = math.tanh(max(0.0, activity) * 4.0)
        amount = 0.003 + 0.035 * max(0.0, resource_fraction) * activity_gate
        self.field.deposit_segment(p0, p1, amount)

        anchor_node = tip.node_id
        anchor = self.graph.nodes[anchor_node]
        if float(np.linalg.norm(p1 - anchor)) >= self.config.node_spacing:
            new_node = self.graph.add_node(p1)
            self.graph.add_edge(anchor_node, new_node, self.config.edge_initial_weight, self.step_index)
            tip.node_id = new_node
            # The parent edge is already present; exclude it so it cannot mask a second nearby branch.
            nearby = self.graph.nearest_node_within(p1, self.config.merge_radius, exclude={new_node, anchor_node})
            if nearby is not None and nearby not in self.graph.neighbors(new_node):
                self.graph.add_edge(new_node, nearby, self.config.edge_initial_weight, self.step_index)
        if productive:
            # Reinforce edges immediately incident to the current structural anchor only.
            for nb in self.graph.neighbors(tip.node_id):
                key = self.graph._key(tip.node_id, nb)
                edge = self.graph.edges[key]
                edge.weight += self.config.edge_reinforcement
                edge.traffic += 1.0
                edge.last_use = self.step_index

    def step(self) -> None:
        self._structural_tick()
        new_tips: list[GrowthTip] = []
        for tip in list(self.tips):
            if not tip.alive:
                continue
            stimulus = self.world.stimulus(tip.position, self.step_index, self.history)
            tip.q = resonator_update(tip.q, stimulus, tip.r, tip.omega, tip.input_gain)
            activity = float(np.linalg.norm(tip.q))
            opportunity, resource_grad = self.world.resource_and_gradient(tip.position, self.step_index, self.history)
            _, cue_grad = self.world.cue_and_gradient(tip.position)
            harvest = self._harvest(activity, opportunity)
            resource = tip.resource - self.config.maintenance_cost - self.config.activity_cost * activity + harvest
            tip.resource = float(np.clip(resource, self.config.resource_floor, self.config.resource_cap))
            productive = harvest > self.config.maintenance_cost
            if productive:
                tip.productive_age += 1
            else:
                tip.productive_age = max(0, tip.productive_age - 1)
            if tip.resource <= self.config.resource_floor:
                tip.alive = False
                tip.age += 1
                continue
            direction = self._direction(tip, resource_grad, cue_grad)
            p0 = tip.position.copy()
            p1 = np.clip(p0 + self.config.tip_step * direction, 0.0, 1.0)
            tip.position = p1
            tip.heading = float(math.atan2(direction[1], direction[0]))
            self._record_growth_segment(tip, p0, p1, activity, productive)
            tip.age += 1
            self._maybe_branch(tip, new_tips)
        self.tips.extend(new_tips)
        self.step_index += 1

    def run(self, steps: int | None = None) -> None:
        count = self.config.total_steps if steps is None else int(steps)
        for _ in range(count):
            self.step()

    def validate_invariants(self) -> list[str]:
        errors: list[str] = []
        for node_id, pos in self.graph.nodes.items():
            if pos.shape != (2,) or not np.all(np.isfinite(pos)):
                errors.append(f'invalid node {node_id}')
        for key, edge in self.graph.edges.items():
            if edge.u not in self.graph.nodes or edge.v not in self.graph.nodes:
                errors.append(f'missing endpoint {key}')
            if edge.weight < 0 or not np.isfinite(edge.weight):
                errors.append(f'invalid weight {key}')
        for tip in self._active_tips():
            if tip.node_id not in self.graph.nodes:
                errors.append(f'orphan tip {tip.tip_id}')
            elif not self.graph.connected_to_soma(tip.node_id):
                errors.append(f'disconnected tip {tip.tip_id}')
            if not np.all(np.isfinite(tip.position)) or not np.all(np.isfinite(tip.q)):
                errors.append(f'nonfinite tip {tip.tip_id}')
            if not (self.config.resource_floor <= tip.resource <= self.config.resource_cap):
                errors.append(f'resource bounds {tip.tip_id}')
        return errors

    def snapshot(self) -> dict:
        return {
            'step': self.step_index,
            'history': self.history,
            'seed': self.seed,
            'control': self.control,
            'graph': self.graph.copy(),
            'field': self.field.copy(),
            'tips': [
                {
                    'tip_id': t.tip_id, 'node_id': t.node_id, 'position': t.position.copy(),
                    'heading': t.heading, 'q': t.q.copy(), 'resource': t.resource,
                    'age': t.age, 'productive_age': t.productive_age,
                    'parent_tip_id': t.parent_tip_id, 'alive': t.alive, 'r': t.r, 'omega': t.omega,
                }
                for t in self.tips
            ],
        }
