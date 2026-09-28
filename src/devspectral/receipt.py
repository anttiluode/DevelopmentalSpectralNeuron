from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .graph import GraphState, EdgeState
from .spectral import Spectrum
from .probe import ProbeResult


def serialize_graph(graph: GraphState) -> dict[str, Any]:
    ids=sorted(graph.nodes)
    return {
        'soma_node_id': int(graph.soma_node_id),
        'node_ids': [int(n) for n in ids],
        'positions': [[float(x) for x in graph.nodes[n]] for n in ids],
        'edges': [
            [int(u),int(v),float(e.weight),int(e.age),float(e.traffic),int(e.last_use)]
            for (u,v),e in sorted(graph.edges.items())
        ],
    }


def deserialize_graph(data: dict[str, Any]) -> GraphState:
    ids=[int(x) for x in data['node_ids']]
    positions=[np.asarray(p,float) for p in data['positions']]
    soma=int(data['soma_node_id'])
    if soma not in ids: raise ValueError('soma missing')
    pos_by=dict(zip(ids,positions))
    g=GraphState(pos_by[soma])
    g.nodes={n:p.copy() for n,p in pos_by.items()}
    g.soma_node_id=soma
    g._next_node_id=(max(ids)+1) if ids else 0
    g.edges={}
    for row in data['edges']:
        u,v,w,age,traffic,last_use=row
        key=g._key(int(u),int(v))
        g.edges[key]=EdgeState(key[0],key[1],float(w),int(age),float(traffic),int(last_use))
    return g


@dataclass
class RunRecord:
    history: str
    seed: int
    control: str
    graph: GraphState
    spectrum: Spectrum
    probe: ProbeResult
    probe_signature: np.ndarray
    invariants: list[str]
    config_hash: str
    field_data: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'history': self.history,
            'seed': int(self.seed),
            'control': self.control,
            'config_hash': self.config_hash,
            'invariants': list(self.invariants),
            'graph': serialize_graph(self.graph),
            'spectrum': {
                'eigenvalues': [float(x) for x in self.spectrum.eigenvalues],
                'all_eigenvalues': [float(x) for x in self.spectrum.all_eigenvalues],
                'component_count': int(self.spectrum.component_count),
            },
            'probe': {
                'times': [float(x) for x in self.probe.times],
                'transfer': [float(x) for x in self.probe.transfer],
                'component_count': int(self.probe.component_count),
                'cross_component': bool(self.probe.cross_component),
                'arrival_time': None if self.probe.arrival_time is None else float(self.probe.arrival_time),
                'spatial_signature': [float(x) for x in self.probe_signature],
            },
        }


@dataclass
class SeparationResult:
    ratio: float
    threshold: float
    matched_cross: int
    cross_above_threshold: int
    required: int
    passed: bool
    median_within: float
    median_cross: float

    def to_dict(self): return {k:(bool(v) if isinstance(v,(np.bool_,bool)) else int(v) if isinstance(v,(np.integer,)) else float(v) if isinstance(v,(np.floating,)) else v) for k,v in vars(self).items()}


@dataclass
class ReadbackResult:
    accuracy: float
    correct: int
    total: int
    predictions: tuple[str, ...]
    truths: tuple[str, ...]
    passed: bool

    def to_dict(self):
        return {'accuracy':float(self.accuracy),'correct':int(self.correct),'total':int(self.total),'predictions':list(self.predictions),'truths':list(self.truths),'passed':bool(self.passed)}


@dataclass
class GateReceipt:
    config_hash: str
    config: dict[str, Any]
    seeds: tuple[int, ...]
    gate0: dict[str, Any]
    structural_separation: SeparationResult
    probe_separation: SeparationResult
    readback: ReadbackResult
    spectral_alignment: dict[str, Any]
    controls: dict[str, Any]
    lesion_regrowth: dict[str, Any]
    overall_status: str
    records: list[RunRecord] = field(default_factory=list)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'config_hash':self.config_hash,
            'config':self.config,
            'seeds':[int(x) for x in self.seeds],
            'git_commit':self.git_commit,
            'gate0':self.gate0,
            'structural_separation':self.structural_separation.to_dict(),
            'probe_separation':self.probe_separation.to_dict(),
            'readback':self.readback.to_dict(),
            'spectral_alignment':self.spectral_alignment,
            'controls':self.controls,
            'lesion_regrowth':self.lesion_regrowth,
            'overall_status':self.overall_status,
            'records':[r.to_dict() for r in self.records],
        }
