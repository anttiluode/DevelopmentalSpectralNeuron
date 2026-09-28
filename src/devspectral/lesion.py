from __future__ import annotations
from dataclasses import dataclass
from .config import V0Config
from .graph import GraphState
from .growth import DevelopmentSimulator


@dataclass
class LesionResult:
    graph: GraphState
    removed_edges: tuple[tuple[int,int], ...]


def _crosses_band(graph: GraphState, key: tuple[int,int], config: V0Config) -> bool:
    u,v=key
    p=graph.nodes[u]; q=graph.nodes[v]
    if p[1] < config.lesion_y_min or q[1] < config.lesion_y_min:
        return False
    x0=config.lesion_x
    return (p[0]-x0) * (q[0]-x0) < 0.0


def lesion_midline_band(graph: GraphState, config: V0Config) -> LesionResult:
    out=graph.copy()
    removed=[]
    for key in list(out.edges):
        if _crosses_band(out,key,config):
            removed.append(key)
            out.remove_edge(*key)
    return LesionResult(out,tuple(sorted(removed)))


def resume_regrowth(simulator: DevelopmentSimulator, steps: int=200):
    simulator.history='H_A'
    simulator.regrowth_mode=True
    simulator.run(steps=int(steps))
    return simulator.snapshot()
