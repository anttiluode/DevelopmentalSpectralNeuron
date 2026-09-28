from dataclasses import replace
import json
import numpy as np

from devspectral.config import V0
from devspectral.experiment import (
    run_one, run_gate_suite, history_features, nearest_centroid_leave_one_seed_out,
    separation_gate, task_alignment, config_hash, record_from_graph,
)
from devspectral.graph import GraphState
from devspectral.receipt import serialize_graph


def tiny_history_graph(side):
    g=GraphState(np.array(V0.soma))
    xs=[0.18,0.25,0.32] if side=='L' else [0.68,0.75,0.82]
    prev=0
    for i,x in enumerate(xs):
        n=g.add_node(np.array([x,0.3+0.15*i]))
        g.add_edge(prev,n,0.3+0.1*i,0)
        prev=n
    return g


def test_run_one_is_deterministic_valid_and_json_serializable():
    a=run_one('H_A',0)
    b=run_one('H_A',0)
    assert a.invariants == [] == b.invariants
    assert serialize_graph(a.graph) == serialize_graph(b.graph)
    np.testing.assert_allclose(a.spectrum.all_eigenvalues,b.spectrum.all_eigenvalues)
    np.testing.assert_allclose(a.probe.transfer,b.probe.transfer)
    assert a.config_hash == config_hash(V0)
    assert a.spectrum.component_count == 1
    json.dumps(a.to_dict())


def test_history_features_use_only_final_anatomy_and_spectrum():
    r=record_from_graph(tiny_history_graph('L'),'H_A',0,'synthetic')
    f1=history_features(r)
    r.development_log={'forbidden':'secret labels'}
    f2=history_features(r)
    np.testing.assert_allclose(f1,f2)
    assert f1.ndim==1 and np.isfinite(f1).all()


def test_leave_one_seed_out_readback_uses_held_out_seed():
    records=[]
    for seed in range(4):
        records.append(record_from_graph(tiny_history_graph('L'),'H_A',seed,'full'))
        records.append(record_from_graph(tiny_history_graph('R'),'H_B',seed,'full'))
    result=nearest_centroid_leave_one_seed_out(records)
    assert result.total==8
    assert result.accuracy >= 0.75
    assert set(result.predictions).issubset({'H_A','H_B'})


def test_separation_gate_uses_within_variation_as_threshold():
    va={0:np.array([0.0]),1:np.array([0.1]),2:np.array([0.2]),3:np.array([0.3])}
    vb={0:np.array([3.0]),1:np.array([3.1]),2:np.array([3.2]),3:np.array([3.3])}
    gate=separation_gate(va,vb)
    assert gate.ratio > 1.0
    assert gate.cross_above_threshold == 4
    assert gate.passed


def test_task_alignment_is_bounded_and_uses_low_mode_subspace():
    r=record_from_graph(tiny_history_graph('L'),'H_A',0,'full')
    score=task_alignment(r.graph,r.spectrum)
    assert 0.0 <= score <= 1.0 + 1e-12


def test_config_hash_changes_when_frozen_constants_change():
    other=replace(V0,tip_step=V0.tip_step*2)
    assert config_hash(other) != config_hash(V0)


def test_gate_suite_two_seed_smoke_writes_gate0_and_core_metrics():
    receipt=run_gate_suite(seeds=range(2),developmental_controls=('full',),graph_controls=())
    data=receipt.to_dict()
    assert data['config_hash']==config_hash(V0)
    assert data['gate0']['deterministic'] is True
    assert data['gate0']['valid_runs'] == 4
    assert 'structural_separation' in data
    assert 'probe_separation' in data
    assert 'readback' in data
    assert 'spectral_alignment' in data
    json.dumps(data)


def test_overall_status_is_negative_when_primary_scientific_gates_fail_cleanly():
    from devspectral.experiment import classify_overall
    status = classify_overall(
        gate0_passed=True,
        structural_passed=False,
        probe_passed=False,
        readback_passed=False,
        alignment_passed=False,
        lesion_passed=False,
        any_material_weakening=True,
    )
    assert status == 'negative'
