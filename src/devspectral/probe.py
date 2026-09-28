from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from .graph import GraphState
from .spectral import laplacian, low_modes


@dataclass(frozen=True)
class ProbeSpec:
    source_center: tuple[float,float]
    source_radius: float
    readout_center: tuple[float,float]
    readout_radius: float


@dataclass
class ProbeResult:
    times: np.ndarray
    states: np.ndarray
    transfer: np.ndarray
    source_indices: tuple[int,...]
    readout_indices: tuple[int,...]
    node_ids: tuple[int,...]
    component_count: int
    cross_component: bool
    arrival_time: float | None


def _indices_near(graph: GraphState, center: tuple[float,float], radius: float, ids: tuple[int,...]) -> tuple[int,...]:
    c=np.asarray(center,float)
    d=np.array([np.linalg.norm(graph.nodes[n]-c) for n in ids],float)
    idx=tuple(int(i) for i in np.flatnonzero(d<=radius))
    if idx:
        return idx
    return (int(np.argmin(d)),)


def _component_labels(graph: GraphState, ids: tuple[int,...]) -> np.ndarray:
    index={n:i for i,n in enumerate(ids)}
    labels=np.full(len(ids),-1,int); label=0
    for n in ids:
        i=index[n]
        if labels[i]>=0: continue
        stack=[n]; labels[i]=label
        while stack:
            u=stack.pop()
            for v in graph.neighbors(u):
                j=index[v]
                if labels[j]<0:
                    labels[j]=label; stack.append(v)
        label+=1
    return labels


def diffusion_trajectory(graph: GraphState, probe: ProbeSpec, times, beta: float) -> ProbeResult:
    for edge in graph.edges.values():
        if edge.weight < 0 or not np.isfinite(edge.weight):
            raise ValueError('invalid graph edge weight')
    ids=tuple(sorted(graph.nodes))
    if not ids:
        raise ValueError('graph has no nodes')
    L=laplacian(graph)
    if L.shape!=(len(ids),len(ids)) or not np.allclose(L,L.T,atol=1e-12):
        raise ValueError('invalid Laplacian')
    if not np.all(np.isfinite(L)):
        raise ValueError('nonfinite Laplacian')
    t=np.asarray(tuple(times),float)
    if t.ndim!=1 or np.any(t<0) or not np.all(np.isfinite(t)):
        raise ValueError('invalid times')
    src=_indices_near(graph,probe.source_center,probe.source_radius,ids)
    read=_indices_near(graph,probe.readout_center,probe.readout_radius,ids)
    x0=np.zeros(len(ids),float); x0[list(src)]=1.0/len(src)
    states=np.vstack([expm(-float(beta)*L*float(tt))@x0 for tt in t])
    transfer=states[:,list(read)].sum(axis=1)
    labels=_component_labels(graph,ids)
    cross=not any(labels[i]==labels[j] for i in src for j in read)
    threshold=0.01
    hits=np.flatnonzero(transfer>=threshold)
    arrival=float(t[hits[0]]) if len(hits) else None
    return ProbeResult(t,states,transfer,src,read,ids,int(labels.max()+1),cross,arrival)


def probe_distance(a: ProbeResult,b: ProbeResult) -> float:
    if a.transfer.shape!=b.transfer.shape or not np.allclose(a.times,b.times):
        raise ValueError('probe grids differ')
    return float(np.linalg.norm(a.transfer-b.transfer))
