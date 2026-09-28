from dataclasses import replace
import numpy as np

from devspectral.config import V0
from devspectral.graph import GraphState
from devspectral.growth import DevelopmentSimulator
from devspectral.lesion import lesion_midline_band, resume_regrowth
from devspectral.probe import ProbeSpec, diffusion_trajectory
from devspectral.spectral import low_modes


def bridge_graph():
    g=GraphState(np.array([0.2,0.2]))
    l1=g.add_node(np.array([0.25,0.70])); l2=g.add_node(np.array([0.42,0.70]))
    r1=g.add_node(np.array([0.58,0.70])); r2=g.add_node(np.array([0.75,0.70]))
    low_l=g.add_node(np.array([0.42,0.30])); low_r=g.add_node(np.array([0.58,0.30]))
    for u,v,w in [(0,l1,1),(l1,l2,1),(l2,r1,0.4),(r1,r2,1),(l2,low_l,0.5),(low_l,low_r,0.5),(low_r,r1,0.5)]:
        g.add_edge(u,v,w,0)
    return g, (l2,r1), (low_l,low_r)


def test_lesion_uses_only_frozen_geometric_band():
    g, high, low=bridge_graph()
    g.edge_centrality={high:9999, low:-9999}
    g.eigenvectors='must not be read'
    result=lesion_midline_band(g,V0)
    assert tuple(sorted(high)) in result.removed_edges
    assert tuple(sorted(low)) not in result.removed_edges
    assert tuple(sorted(high)) not in result.graph.edges
    assert tuple(sorted(low)) in result.graph.edges
    # Original stays untouched.
    assert tuple(sorted(high)) in g.edges


def test_bridge_lesion_lowers_algebraic_connectivity_and_transfer():
    g, high, low=bridge_graph()
    # use a config lesion band that also removes the low bridge so graph disconnects
    cfg=replace(V0,lesion_y_min=0.0)
    before=low_modes(g,k=4).all_eigenvalues[1]
    p=ProbeSpec((0.25,0.70),0.08,(0.75,0.70),0.08)
    tr_before=diffusion_trajectory(g,p,(0,0.5,2.0),0.75).transfer[-1]
    lesion=lesion_midline_band(g,cfg)
    after=low_modes(lesion.graph,k=4).all_eigenvalues[1]
    tr_after=diffusion_trajectory(lesion.graph,p,(0,0.5,2.0),0.75).transfer[-1]
    assert after < before
    assert tr_after < tr_before
    assert abs(after) < 1e-10
    assert abs(tr_after) < 1e-12


def test_resume_regrowth_uses_balanced_schedule_and_same_local_simulator():
    cfg=replace(V0,regrowth_steps=4,steps_per_epoch=2,epochs=2)
    sim=DevelopmentSimulator(cfg,'H_B',seed=7)
    sim.run(steps=cfg.total_steps)
    start=sim.step_index
    resume_regrowth(sim,steps=4)
    assert sim.step_index == start+4
    assert sim.history == 'H_A'
    assert sim.regrowth_mode is True
