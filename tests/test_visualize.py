import matplotlib
matplotlib.use('Agg')
import numpy as np

from devspectral.config import V0
from devspectral.graph import GraphState
from devspectral.experiment import record_from_graph
from devspectral.receipt import serialize_graph
from devspectral.visualize import render_snapshot


def tiny_record():
    g=GraphState(np.array(V0.soma))
    a=g.add_node(np.array([0.3,0.4])); b=g.add_node(np.array([0.7,0.7])); c=g.add_node(np.array([0.5,0.85]))
    g.add_edge(0,a,0.4,0); g.add_edge(a,b,0.2,0); g.add_edge(b,c,0.5,0); g.add_edge(a,c,0.3,0)
    g.edges[g._key(a,b)].traffic=3.0
    r=record_from_graph(g,'H_A',0,'full')
    yy,xx=np.mgrid[0:64,0:64]
    r.field_data=np.exp(-((xx-32)**2+(yy-38)**2)/(2*8**2))
    return r


def test_all_visualizer_layers_render_headlessly_without_mutating_record(tmp_path):
    r=tiny_record()
    before_graph=serialize_graph(r.graph)
    before_field=r.field_data.copy()
    for layer in ('field','graph','usage','mode1','mode2','mode3','probe','lesion'):
        out=tmp_path/f'{layer}.png'
        fig=render_snapshot(r,layer,output_path=out)
        assert fig is not None
        assert out.exists() and out.stat().st_size>0
    assert serialize_graph(r.graph)==before_graph
    np.testing.assert_allclose(r.field_data,before_field)


def test_unknown_layer_is_rejected():
    r=tiny_record()
    try:
        render_snapshot(r,'made-up')
    except ValueError as exc:
        assert 'layer' in str(exc).lower()
    else:
        raise AssertionError('unknown layer should fail')
