from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EdgeState:
    u: int
    v: int
    weight: float
    age: int = 0
    traffic: float = 0.0
    last_use: int = 0


class GraphState:
    def __init__(self, soma_position: np.ndarray):
        self.nodes: dict[int, np.ndarray] = {0: np.asarray(soma_position, dtype=float).copy()}
        self.edges: dict[tuple[int, int], EdgeState] = {}
        self.soma_node_id = 0
        self._next_node_id = 1

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        if u == v:
            raise ValueError('self edges are not supported')
        return (u, v) if u < v else (v, u)

    def add_node(self, position: np.ndarray) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[node_id] = np.asarray(position, dtype=float).copy()
        return node_id

    def add_edge(self, u: int, v: int, weight: float, step: int) -> None:
        if weight < 0:
            raise ValueError('edge weight must be nonnegative')
        if u not in self.nodes or v not in self.nodes:
            raise ValueError('edge endpoint missing')
        key = self._key(u, v)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = float(weight)
            edge.last_use = int(step)
        else:
            self.edges[key] = EdgeState(key[0], key[1], float(weight), last_use=int(step))

    def weight(self, u: int, v: int) -> float:
        return self.edges[self._key(u, v)].weight

    def neighbors(self, node_id: int) -> list[int]:
        out = []
        for (u, v), _ in self.edges.items():
            if u == node_id:
                out.append(v)
            elif v == node_id:
                out.append(u)
        return out

    def remove_edge(self, u: int, v: int) -> None:
        self.edges.pop(self._key(u, v), None)

    def nearest_node_within(self, position: np.ndarray, radius: float, exclude: set[int] | None = None) -> int | None:
        p = np.asarray(position, dtype=float)
        excluded = exclude or set()
        best = None
        best_d = float(radius)
        for node_id, q in self.nodes.items():
            if node_id in excluded:
                continue
            d = float(np.linalg.norm(q - p))
            if d <= best_d:
                best = node_id
                best_d = d
        return best

    def connected_to_soma(self, node_id: int) -> bool:
        if node_id == self.soma_node_id:
            return True
        if node_id not in self.nodes:
            return False
        seen = {self.soma_node_id}
        stack = [self.soma_node_id]
        while stack:
            u = stack.pop()
            for v in self.neighbors(u):
                if v == node_id:
                    return True
                if v not in seen:
                    seen.add(v); stack.append(v)
        return False

    def path_to_soma(self, node_id: int) -> list[tuple[int, int]] | None:
        if node_id == self.soma_node_id:
            return []
        parent = {self.soma_node_id: None}
        queue = [self.soma_node_id]
        for u in queue:
            for v in self.neighbors(u):
                if v in parent:
                    continue
                parent[v] = u
                if v == node_id:
                    cur = v; path = []
                    while parent[cur] is not None:
                        par = parent[cur]
                        path.append(self._key(par, cur))
                        cur = par
                    return path
                queue.append(v)
        return None

    def copy(self) -> 'GraphState':
        g = GraphState(self.nodes[self.soma_node_id])
        g.nodes = {k: v.copy() for k, v in self.nodes.items()}
        g.edges = {k: EdgeState(**vars(v)) for k, v in self.edges.items()}
        g.soma_node_id = self.soma_node_id
        g._next_node_id = self._next_node_id
        return g
