from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from .graph import GraphState


@dataclass
class Spectrum:
    node_ids: tuple[int, ...]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    all_eigenvalues: np.ndarray
    component_count: int


def weighted_adjacency(graph: GraphState) -> np.ndarray:
    ids=tuple(sorted(graph.nodes))
    index={n:i for i,n in enumerate(ids)}
    A=np.zeros((len(ids),len(ids)),float)
    for (u,v),edge in graph.edges.items():
        if edge.weight < 0 or not np.isfinite(edge.weight):
            raise ValueError('invalid edge weight')
        i,j=index[u],index[v]
        A[i,j]=A[j,i]=edge.weight
    return A


def laplacian(graph: GraphState) -> np.ndarray:
    A=weighted_adjacency(graph)
    return np.diag(A.sum(axis=1))-A


def low_modes(graph: GraphState, k: int=6, tol: float=1e-10) -> Spectrum:
    ids=tuple(sorted(graph.nodes))
    L=laplacian(graph)
    if L.size == 0:
        return Spectrum(ids,np.zeros(0),np.zeros((0,0)),np.zeros(0),0)
    vals,vecs=eigh(L,check_finite=True)
    vals=np.where(np.abs(vals)<tol,0.0,vals)
    components=int(np.count_nonzero(vals <= tol))
    keep=np.flatnonzero(vals>tol)[:k]
    return Spectrum(ids,vals[keep].copy(),vecs[:,keep].copy(),vals.copy(),components)


def mode_alignment(a: np.ndarray,b: np.ndarray) -> float:
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel()
    if a.shape!=b.shape: raise ValueError('mode shapes differ')
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<1e-15 or nb<1e-15: return 0.0
    return abs(float(a@b)/(na*nb))


def subspace_distance(U: np.ndarray,V: np.ndarray) -> float:
    U=np.asarray(U,float); V=np.asarray(V,float)
    if U.ndim==1: U=U[:,None]
    if V.ndim==1: V=V[:,None]
    if U.shape[0]!=V.shape[0]: raise ValueError('subspace ambient dimensions differ')
    Qu,_=np.linalg.qr(U)
    Qv,_=np.linalg.qr(V)
    Pu=Qu@Qu.T; Pv=Qv@Qv.T
    return float(np.linalg.norm(Pu-Pv,ord='fro')/np.sqrt(2.0))


def spectral_signature(spectrum: Spectrum,k: int=6) -> np.ndarray:
    out=np.zeros(k,float)
    n=min(k,len(spectrum.eigenvalues))
    if n: out[:n]=spectrum.eigenvalues[:n]
    return out


def clustered_subspaces(spectrum: Spectrum, rel_gap: float=1e-7) -> list[np.ndarray]:
    vals=spectrum.eigenvalues
    if len(vals)==0: return []
    clusters=[]; start=0
    for i in range(len(vals)-1):
        scale=max(1.0,abs(float(vals[i])),abs(float(vals[i+1])))
        if abs(float(vals[i+1]-vals[i])) >= rel_gap*scale:
            clusters.append(spectrum.eigenvectors[:,start:i+1]); start=i+1
    clusters.append(spectrum.eigenvectors[:,start:])
    return clusters
