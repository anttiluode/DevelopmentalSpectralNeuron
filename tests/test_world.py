import math
import numpy as np

from devspectral.config import V0, V0Config
from devspectral.world import DevelopmentalWorld, epoch_schedule


def test_v0_constants_are_frozen_and_exact():
    assert isinstance(V0, V0Config)
    assert V0.grid_size == 64
    assert V0.epochs == 12
    assert V0.steps_per_epoch == 50
    assert V0.total_steps == 600
    assert V0.soma == (0.50, 0.08)
    assert V0.left_target == (0.25, 0.78)
    assert V0.right_target == (0.75, 0.78)
    assert V0.target_sigma == 0.11
    try:
        V0.grid_size = 32
    except Exception:
        pass
    else:
        raise AssertionError('V0Config must be frozen')


def test_histories_have_same_exposure_counts_but_different_order():
    ha = epoch_schedule('H_A', V0)
    hb = epoch_schedule('H_B', V0)
    assert ha == ('L','R','L','R','L','R','L','R','L','R','L','R')
    assert hb == ('L','L','L','L','L','L','R','R','R','R','R','R')
    assert ha.count('L') == hb.count('L') == 6
    assert ha.count('R') == hb.count('R') == 6
    assert ha != hb


def test_static_cue_is_history_independent_and_symmetric():
    world = DevelopmentalWorld(V0)
    for p in [(0.5, 0.2), (0.25, 0.78), (0.75, 0.78), (0.05, 0.5)]:
        c1, g1 = world.cue_and_gradient(np.array(p, float))
        c2, g2 = world.cue_and_gradient(np.array(p, float))
        assert math.isfinite(c1)
        assert c1 == c2
        np.testing.assert_allclose(g1, g2)
    cl, _ = world.cue_and_gradient(np.array([0.25, 0.78]))
    cr, _ = world.cue_and_gradient(np.array([0.75, 0.78]))
    assert abs(cl - cr) < 1e-12


def test_resource_schedule_activates_only_scheduled_target():
    world = DevelopmentalWorld(V0)
    left = np.array(V0.left_target)
    right = np.array(V0.right_target)
    # First epoch is left for both histories.
    rl, _ = world.resource_and_gradient(left, 0, 'H_A')
    rr, _ = world.resource_and_gradient(right, 0, 'H_A')
    assert rl > rr
    # At step 50 H_A switches to right while H_B remains left.
    ha_l, _ = world.resource_and_gradient(left, 50, 'H_A')
    ha_r, _ = world.resource_and_gradient(right, 50, 'H_A')
    hb_l, _ = world.resource_and_gradient(left, 50, 'H_B')
    hb_r, _ = world.resource_and_gradient(right, 50, 'H_B')
    assert ha_r > ha_l
    assert hb_l > hb_r


def test_stimulus_is_deterministic_and_not_a_history_label():
    world = DevelopmentalWorld(V0)
    pos = np.array([0.25, 0.78])
    a = world.stimulus(pos, 17, 'H_A')
    b = world.stimulus(pos, 17, 'H_A')
    assert a == b
    # Same scheduled target at same step => same local stimulus across histories.
    assert a == world.stimulus(pos, 17, 'H_B')
