import numpy as np
from devspectral.graph import GraphState
from devspectral.spectral import weighted_adjacency, laplacian, low_modes, mode_alignment, subspace_distance, spectral_signature


def path_graph(n=4):
    g = GraphState(np.array([0.0,0.0]))
    ids=[g.soma_node_id]
    for i in range(1,n): ids.append(g.add_node(np.array([i/(n-1),0.0])))
    for a,b in zip(ids[:-1],ids[1:]): g.add_edge(a,b,1.0,0)
    return g


def cycle_graph(n=4):
    g=path_graph(n)
    ids=sorted(g.nodes)
    g.add_edge(ids[-1],ids[0],1.0,0)
    return g


def two_lobe_graph():
    g=GraphState(np.array([0.0,0.0]))
    a=g.add_node(np.array([0.2,0.0])); b=g.add_node(np.array([0.4,0.0]))
    c=g.add_node(np.array([0.6,0.0])); d=g.add_node(np.array([0.8,0.0]))
    for u,v,w in [(0,a,1),(a,b,1),(b,c,0.1),(c,d,1)]: g.add_edge(u,v,w,0)
    return g


def test_laplacian_is_symmetric_psd_and_connected_has_positive_lambda2():
    for g in [path_graph(), cycle_graph(), two_lobe_graph()]:
        A=weighted_adjacency(g); L=laplacian(g)
        np.testing.assert_allclose(A,A.T)
        np.testing.assert_allclose(L,L.T)
        vals=np.linalg.eigvalsh(L)
        assert vals[0] >= -1e-10
        assert abs(vals[0]) < 1e-10
        assert vals[1] > 0
        spec=low_modes(g,k=3)
        assert spec.component_count == 1
        assert np.all(np.diff(spec.eigenvalues) >= -1e-12)


def test_mode_alignment_is_sign_invariant():
    u=np.array([1.0,2.0,3.0]); u/=np.linalg.norm(u)
    assert abs(mode_alignment(u,u)-1.0)<1e-12
    assert abs(mode_alignment(u,-u)-1.0)<1e-12


def test_subspace_distance_ignores_rotation_inside_degenerate_space():
    U=np.eye(5)[:, :2]
    th=0.7
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    V=U@R
    assert subspace_distance(U,V) < 1e-12
    # Naive column comparison would see a difference.
    assert abs(float(U[:,0]@V[:,0])) < 0.9


def test_disconnected_graph_reports_components_and_skips_zero_modes():
    g=GraphState(np.array([0.0,0.0]))
    a=g.add_node(np.array([0.1,0])); b=g.add_node(np.array([0.9,0]))
    g.add_edge(0,a,1.0,0)
    spec=low_modes(g,k=3)
    assert spec.component_count == 2
    assert np.all(spec.eigenvalues > 1e-10)


def test_spectral_signature_is_fixed_length_and_finite():
    spec=low_modes(path_graph(3),k=6)
    sig=spectral_signature(spec,k=6)
    assert sig.shape==(6,)
    assert np.all(np.isfinite(sig))
    assert np.count_nonzero(sig)==2
