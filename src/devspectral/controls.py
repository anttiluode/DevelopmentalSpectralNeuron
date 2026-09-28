from __future__ import annotations
import math
import numpy as np
from .config import V0Config
from .graph import GraphState, EdgeState

DEVELOPMENTAL_CONTROLS = ('full','no_stigmergy','no_metabolic_selection','no_pruning')


def weight_shuffle(graph: GraphState, seed: int) -> GraphState:
    rng=np.random.default_rng(seed)
    out=graph.copy()
    keys=list(out.edges)
    weights=np.array([out.edges[k].weight for k in keys],float)
    rng.shuffle(weights)
    for k,w in zip(keys,weights): out.edges[k].weight=float(w)
    return out


def _edge_length(graph: GraphState, key: tuple[int,int]) -> float:
    u,v=key
    return float(np.linalg.norm(graph.nodes[u]-graph.nodes[v]))


def _bin_index(length: float) -> int:
    bins=np.linspace(0.0,math.sqrt(2.0),9)
    return int(np.clip(np.digitize([length],bins,right=False)[0]-1,0,7))


def geometry_null(graph: GraphState, seed: int) -> GraphState:
    """Degree-ish edge swaps preserving geometry budget and connectivity.

    Starts from the observed geometry and performs length-bin-preserving double-edge
    swaps. Node coordinates and edge-weight multiset are unchanged.
    """
    rng=np.random.default_rng(seed)
    base=graph.copy()
    if len(base.edges) < 2:
        raise RuntimeError('geometry null needs at least two edges')
    original_keys=list(base.edges)
    original_weights=[base.edges[k].weight for k in original_keys]
    out=base.copy()
    swaps=0
    target=max(1,len(original_keys)//3)
    for _ in range(200):
        keys=list(out.edges)
        if len(keys)<2: break
        i,j=rng.choice(len(keys),size=2,replace=False)
        e1,e2=keys[int(i)],keys[int(j)]
        a,b=e1; c,d=e2
        if len({a,b,c,d})<4: continue
        proposals=[((a,d),(c,b)),((a,c),(b,d))]
        rng.shuffle(proposals)
        accepted=False
        old_bins=sorted((_bin_index(_edge_length(out,e1)),_bin_index(_edge_length(out,e2))))
        for p1,p2 in proposals:
            k1=out._key(*p1); k2=out._key(*p2)
            if k1==k2 or k1 in out.edges or k2 in out.edges: continue
            new_bins=sorted((_bin_index(float(np.linalg.norm(out.nodes[k1[0]]-out.nodes[k1[1]]))),
                             _bin_index(float(np.linalg.norm(out.nodes[k2[0]]-out.nodes[k2[1]])))))
            if new_bins != old_bins: continue
            old1=out.edges[e1]; old2=out.edges[e2]
            del out.edges[e1]; del out.edges[e2]
            out.edges[k1]=EdgeState(k1[0],k1[1],old1.weight,old1.age,old1.traffic,old1.last_use)
            out.edges[k2]=EdgeState(k2[0],k2[1],old2.weight,old2.age,old2.traffic,old2.last_use)
            if all(out.connected_to_soma(n) for n in out.nodes):
                swaps += 1; accepted=True; break
            del out.edges[k1]; del out.edges[k2]
            out.edges[e1]=old1; out.edges[e2]=old2
        if accepted and swaps>=target and set(out.edges) != set(original_keys):
            break
    if swaps==0 or set(out.edges) == set(original_keys):
        raise RuntimeError('could not produce connected geometry null in 200 attempts')
    # Randomize the preserved weight multiset across the new topology.
    keys=list(out.edges); weights=np.array(original_weights,float); rng.shuffle(weights)
    for k,w in zip(keys,weights): out.edges[k].weight=float(w)
    return out


def fixed_lattice(config: V0Config, node_budget: int, edge_budget: int, seed: int) -> GraphState:
    if node_budget < 2 or edge_budget < node_budget-1:
        raise ValueError('connected lattice requires edge_budget >= node_budget-1')
    max_edges=node_budget*(node_budget-1)//2
    if edge_budget>max_edges: raise ValueError('too many edges requested')
    g=GraphState(np.array(config.soma,float))
    side=int(math.ceil(math.sqrt(node_budget-1)))
    candidates=[]
    for iy in range(side):
        for ix in range(side):
            if len(candidates)>=node_budget-1: break
            x=(ix+1)/(side+1); y=0.16 + 0.76*(iy+1)/(side+1)
            candidates.append((x,y))
        if len(candidates)>=node_budget-1: break
    ids=[g.add_node(np.array(p,float)) for p in candidates]
    all_ids=[g.soma_node_id]+ids
    # Prim-like nearest connection for a connected geometric backbone.
    connected=[g.soma_node_id]; remaining=set(ids)
    while remaining:
        best=None
        for u in connected:
            for v in remaining:
                d=float(np.linalg.norm(g.nodes[u]-g.nodes[v]))
                if best is None or d<best[0]: best=(d,u,v)
        _,u,v=best
        g.add_edge(u,v,config.edge_initial_weight,0)
        connected.append(v); remaining.remove(v)
    pairs=[]
    existing=set(g.edges)
    for i,u in enumerate(all_ids):
        for v in all_ids[i+1:]:
            k=g._key(u,v)
            if k in existing: continue
            pairs.append((float(np.linalg.norm(g.nodes[u]-g.nodes[v])),u,v))
    # Seed only resolves equal-distance ties while retaining geometric locality.
    rng=np.random.default_rng(seed)
    jitter=rng.uniform(0,1e-9,size=len(pairs)) if pairs else []
    pairs=[(*p,j) for p,j in zip(pairs,jitter)]
    pairs.sort(key=lambda x:(x[0],x[3]))
    for _,u,v,_ in pairs[:edge_budget-len(g.edges)]:
        g.add_edge(u,v,config.edge_initial_weight,0)
    return g
