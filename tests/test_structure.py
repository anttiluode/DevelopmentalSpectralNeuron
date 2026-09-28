from dataclasses import replace
import numpy as np

from devspectral.config import V0
from devspectral.field import StructuralField
from devspectral.growth import DevelopmentSimulator, GrowthTip


def test_field_deposition_and_decay_are_local_and_monotone():
    field = StructuralField(V0)
    p0 = np.array([0.45, 0.4])
    p1 = np.array([0.55, 0.4])
    before = field.data.sum()
    field.deposit_segment(p0, p1, amount=1.0)
    after = field.data.sum()
    assert after > before
    value, grad = field.sample_and_gradient(np.array([0.5, 0.4]))
    assert value > 0
    assert grad.shape == (2,)
    field.decay()
    assert 0 < field.data.sum() < after


def test_growth_deposits_field_and_builds_explicit_graph():
    sim = DevelopmentSimulator(V0, 'H_A', seed=0)
    sim.run(steps=5)
    assert sim.field.data.sum() > 0
    assert len(sim.graph.nodes) > 1
    assert len(sim.graph.edges) > 0
    assert sim.validate_invariants() == []


def test_branching_is_local_and_respects_active_tip_cap():
    cfg = replace(
        V0,
        max_active_tips=6,
        branch_threshold=0.0,
        min_productive_age=0,
        maintenance_cost=0.0,
        activity_cost=0.0,
        harvest_gain=0.0,
    )
    sim = DevelopmentSimulator(cfg, 'H_A', seed=1)
    sim.run(steps=8)
    active = [t for t in sim.tips if t.alive]
    assert cfg.initial_tips <= len(active) <= cfg.max_active_tips
    assert any(t.parent_tip_id is not None for t in active)


def test_local_merge_helper_only_returns_nodes_within_radius():
    sim = DevelopmentSimulator(V0, 'H_A', seed=2)
    n1 = sim.graph.add_node(np.array([0.4, 0.4]))
    n2 = sim.graph.add_node(np.array([0.8, 0.8]))
    got = sim.graph.nearest_node_within(np.array([0.41, 0.4]), radius=0.02)
    assert got == n1
    assert sim.graph.nearest_node_within(np.array([0.6, 0.6]), radius=0.02) is None
    assert n2 != n1


def test_pruning_never_orphans_an_active_tip():
    cfg = replace(V0, prune_threshold=0.5, prune_grace=0)
    sim = DevelopmentSimulator(cfg, 'H_A', seed=3)
    g = sim.graph
    n1 = g.add_node(np.array([0.5, 0.12]))
    n2 = g.add_node(np.array([0.5, 0.16]))
    g.add_edge(g.soma_node_id, n1, weight=0.01, step=0)
    g.add_edge(n1, n2, weight=0.01, step=0)
    tip = sim.tips[0]
    tip.node_id = n2
    tip.position = g.nodes[n2].copy()
    # Other tips stay at soma, so this chain is the only dependency.
    sim._prune_edges()
    assert tip.alive
    assert g.connected_to_soma(tip.node_id)
    assert (min(g.soma_node_id, n1), max(g.soma_node_id, n1)) in g.edges
    assert (min(n1, n2), max(n1, n2)) in g.edges
    assert sim.validate_invariants() == []


def test_unused_low_weight_edge_can_be_pruned():
    cfg = replace(V0, prune_threshold=0.5, prune_grace=0)
    sim = DevelopmentSimulator(cfg, 'H_A', seed=4)
    g = sim.graph
    n1 = g.add_node(np.array([0.6, 0.12]))
    g.add_edge(g.soma_node_id, n1, weight=0.01, step=0)
    sim._prune_edges()
    assert len(g.edges) == 0
    assert sim.validate_invariants() == []


def test_local_merge_can_join_neighboring_branch_instead_of_being_masked_by_parent():
    sim = DevelopmentSimulator(V0, 'H_A', seed=20)
    tip = sim.tips[0]
    parent = sim.graph.add_node(np.array([0.50, 0.50]))
    tip.node_id = parent
    tip.position = np.array([0.50, 0.50])
    other = sim.graph.add_node(np.array([0.526, 0.530]))
    p0 = tip.position.copy()
    p1 = np.array([0.526, 0.50])  # parent is closer than `other`, both inside merge radius from the new node
    sim._record_growth_segment(tip, p0, p1, activity=1.0, productive=False)
    new_node = tip.node_id
    assert new_node not in (parent, other)
    assert other in sim.graph.neighbors(new_node)


def test_branching_conserves_parent_plus_child_resource():
    cfg = replace(V0, branch_threshold=0.0, min_productive_age=0, max_active_tips=5)
    sim = DevelopmentSimulator(cfg, 'H_A', seed=21)
    tip = sim.tips[0]
    tip.resource = 1.6
    before = tip.resource
    newborns = []
    sim._maybe_branch(tip, newborns)
    assert len(newborns) == 1
    assert abs(tip.resource + newborns[0].resource - before) < 1e-12


def test_vanishing_local_gradient_is_not_amplified_to_full_directional_signal():
    cfg = replace(V0, exploration_sd=0.0)
    a = DevelopmentSimulator(cfg, 'H_A', seed=22)
    b = DevelopmentSimulator(cfg, 'H_A', seed=22)
    tip_a, tip_b = a.tips[0], b.tips[0]
    zero = np.zeros(2)
    tiny = np.array([1e-9, 0.0])
    d0 = a._direction(tip_a, zero, zero)
    d1 = b._direction(tip_b, tiny, zero)
    assert np.linalg.norm(d1 - d0) < 1e-6
