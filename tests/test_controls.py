from dataclasses import replace
import numpy as np

from devspectral.config import V0
from devspectral.growth import DevelopmentSimulator
from devspectral.graph import GraphState
from devspectral.controls import geometry_null, weight_shuffle, fixed_lattice


def _length_hist(g):
    bins=np.linspace(0.0,np.sqrt(2.0),9)
    lengths=[]
    for (u,v) in g.edges:
        lengths.append(np.linalg.norm(g.nodes[u]-g.nodes[v]))
    return np.histogram(lengths,bins=bins)[0]


def rich_graph():
    g=GraphState(np.array([0.5,0.08]))
    pts=[(0.2,0.2),(0.4,0.2),(0.6,0.2),(0.8,0.2),(0.2,0.6),(0.4,0.6),(0.6,0.6),(0.8,0.6)]
    ids=[g.add_node(np.array(p,float)) for p in pts]
    edges=[(0,ids[0]),(ids[0],ids[1]),(ids[1],ids[2]),(ids[2],ids[3]),
           (ids[0],ids[4]),(ids[1],ids[5]),(ids[2],ids[6]),(ids[3],ids[7]),
           (ids[4],ids[5]),(ids[5],ids[6]),(ids[6],ids[7]),(ids[1],ids[4]),(ids[2],ids[7])]
    for k,(u,v) in enumerate(edges): g.add_edge(u,v,0.2+0.01*k,0)
    return g


def test_no_stigmergy_disables_following_but_still_deposits_field():
    full=DevelopmentSimulator(V0,'H_A',seed=10,control='full')
    off=DevelopmentSimulator(V0,'H_A',seed=10,control='no_stigmergy')
    # Prime identical nonzero trail fields away from the tips so the following term matters.
    for sim in (full,off):
        sim.field.deposit_segment(np.array([0.45,0.08]),np.array([0.45,0.3]),3.0)
    full.step(); off.step()
    assert full.field.data.sum() > 0 and off.field.data.sum() > 0
    assert not np.allclose(full.tips[0].position,off.tips[0].position)


def test_no_metabolic_selection_removes_opportunity_dependence_from_harvest():
    full=DevelopmentSimulator(V0,'H_A',seed=1,control='full')
    off=DevelopmentSimulator(V0,'H_A',seed=1,control='no_metabolic_selection')
    assert full._harvest(activity=1.0,opportunity=0.0) != full._harvest(activity=1.0,opportunity=1.0)
    assert off._harvest(activity=1.0,opportunity=0.0) == off._harvest(activity=1.0,opportunity=1.0)


def test_no_pruning_keeps_depleted_edge_while_full_prunes_it():
    cfg=replace(V0,prune_threshold=0.5,prune_grace=0)
    full=DevelopmentSimulator(cfg,'H_A',seed=1,control='full')
    off=DevelopmentSimulator(cfg,'H_A',seed=1,control='no_pruning')
    for sim in (full,off):
        n=sim.graph.add_node(np.array([0.8,0.1]))
        sim.graph.add_edge(0,n,0.01,0)
    full._prune_edges(); off._prune_edges()
    assert len(full.graph.edges)==0
    assert len(off.graph.edges)==1


def test_weight_shuffle_preserves_topology_and_weight_multiset():
    g=rich_graph(); h=weight_shuffle(g,seed=3)
    assert set(g.edges)==set(h.edges)
    assert sorted(round(e.weight,12) for e in g.edges.values()) == sorted(round(e.weight,12) for e in h.edges.values())
    assert all(e.weight>=0 for e in h.edges.values())


def test_geometry_null_preserves_geometry_budget_connectivity_and_length_histogram():
    g=rich_graph(); h=geometry_null(g,seed=4)
    assert set(g.nodes)==set(h.nodes)
    for n in g.nodes: np.testing.assert_allclose(g.nodes[n],h.nodes[n])
    assert len(g.edges)==len(h.edges)
    assert h.connected_to_soma(max(h.nodes))
    assert np.max(np.abs(_length_hist(g)-_length_hist(h))) <= 1
    assert all(e.weight>=0 for e in h.edges.values())
    assert set(h.edges) != set(g.edges)


def test_fixed_lattice_matches_requested_node_and_edge_budget_and_is_connected():
    g=fixed_lattice(V0,node_budget=16,edge_budget=22,seed=2)
    assert len(g.nodes)==16
    assert len(g.edges)==22
    assert all(g.connected_to_soma(n) for n in g.nodes)
