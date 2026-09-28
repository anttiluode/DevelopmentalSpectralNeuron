import inspect
import numpy as np

from devspectral.config import V0
from devspectral.graph import GraphState
from devspectral.growth import DevelopmentSimulator, GrowthTip, resonator_update
from devspectral.world import DevelopmentalWorld


def test_resonator_update_matches_rotation_equation():
    q = np.array([1.0, 0.0])
    out = resonator_update(q, stimulus=0.5, r=0.9, omega=0.2, input_gain=1.0)
    R = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    expected = 0.9 * (R @ q) + 0.1 * np.array([0.5, 0.0])
    np.testing.assert_allclose(out, expected)


def test_graph_is_undirected_and_nonnegative():
    g = GraphState(soma_position=np.array(V0.soma))
    n1 = g.add_node(np.array([0.5, 0.1]))
    n2 = g.add_node(np.array([0.5, 0.2]))
    g.add_edge(n1, n2, weight=0.3, step=0)
    assert g.weight(n1, n2) == g.weight(n2, n1) == 0.3
    assert set(g.neighbors(n1)) == {n2}
    assert set(g.neighbors(n2)) == {n1}
    try:
        g.add_edge(n1, n2, weight=-0.1, step=0)
    except ValueError:
        pass
    else:
        raise AssertionError('negative edge weights must be rejected')


def test_one_step_is_deterministic_and_resource_is_clipped():
    a = DevelopmentSimulator(V0, 'H_A', seed=3)
    b = DevelopmentSimulator(V0, 'H_A', seed=3)
    a.step(); b.step()
    assert len(a.tips) == len(b.tips) == V0.initial_tips
    for ta, tb in zip(a.tips, b.tips):
        np.testing.assert_allclose(ta.position, tb.position)
        np.testing.assert_allclose(ta.q, tb.q)
        assert V0.resource_floor <= ta.resource <= V0.resource_cap
        assert ta.resource == tb.resource


def test_tip_motion_never_exceeds_step_length():
    sim = DevelopmentSimulator(V0, 'H_A', seed=4)
    before = [t.position.copy() for t in sim.tips]
    sim.step()
    for p0, tip in zip(before, sim.tips):
        assert np.linalg.norm(tip.position - p0) <= V0.tip_step + 1e-12
        assert np.all((0.0 <= tip.position) & (tip.position <= 1.0))


def test_direction_selector_has_no_graph_or_spectral_argument():
    sig = inspect.signature(DevelopmentSimulator._direction)
    names = list(sig.parameters)
    assert names == ['self', 'tip', 'resource_grad', 'cue_grad']
    assert 'graph' not in names and 'spectrum' not in names


def test_snapshot_copies_state_not_aliases():
    sim = DevelopmentSimulator(V0, 'H_A', seed=2)
    snap = sim.snapshot()
    old = snap['tips'][0]['position'].copy()
    sim.step()
    np.testing.assert_allclose(snap['tips'][0]['position'], old)
