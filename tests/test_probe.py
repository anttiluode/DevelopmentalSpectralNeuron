import numpy as np
import pytest
from devspectral.graph import GraphState
from devspectral.probe import ProbeSpec, diffusion_trajectory, probe_distance


def line_graph(disconnected=False):
    g=GraphState(np.array([0.0,0.0]))
    a=g.add_node(np.array([0.33,0.0])); b=g.add_node(np.array([0.66,0.0])); c=g.add_node(np.array([1.0,0.0]))
    g.add_edge(0,a,1.0,0)
    if not disconnected:
        g.add_edge(a,b,1.0,0)
    g.add_edge(b,c,1.0,0)
    return g


def probe():
    return ProbeSpec(source_center=(0.0,0.0), source_radius=0.05, readout_center=(1.0,0.0), readout_radius=0.05)


def test_same_graph_probe_is_deterministic_and_conserves_mass():
    g=line_graph()
    times=(0.0,0.2,0.5,1.0)
    a=diffusion_trajectory(g,probe(),times,beta=0.75)
    b=diffusion_trajectory(g,probe(),times,beta=0.75)
    np.testing.assert_allclose(a.states,b.states)
    np.testing.assert_allclose(a.states.sum(axis=1),1.0,atol=1e-10)
    assert a.component_count==1
    assert a.transfer[-1] > a.transfer[0]
    assert probe_distance(a,b)==0.0


def test_disconnected_graph_reports_components_and_no_cross_component_transfer():
    r=diffusion_trajectory(line_graph(disconnected=True),probe(),(0.0,0.5,2.0),beta=0.75)
    assert r.component_count==2
    assert np.all(np.isfinite(r.states))
    assert np.allclose(r.transfer,0.0,atol=1e-12)
    assert r.cross_component is True


def test_negative_edge_weight_is_rejected_before_propagation():
    g=line_graph()
    edge=next(iter(g.edges.values()))
    edge.weight=-0.5
    with pytest.raises(ValueError):
        diffusion_trajectory(g,probe(),(0.0,1.0),beta=0.75)


def test_empty_source_or_readout_region_falls_back_to_nearest_node():
    g=line_graph()
    p=ProbeSpec(source_center=(0.1,0.2),source_radius=1e-6,readout_center=(0.9,0.2),readout_radius=1e-6)
    r=diffusion_trajectory(g,p,(0.0,1.0),beta=0.5)
    assert len(r.source_indices)==1
    assert len(r.readout_indices)==1
    assert np.isfinite(r.transfer).all()
