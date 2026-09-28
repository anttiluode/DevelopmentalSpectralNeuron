from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import V0
from .receipt import RunRecord

LAYERS=('field','graph','usage','mode1','mode2','mode3','probe','lesion')


def _draw_graph(ax, record: RunRecord, usage: bool=False):
    g=record.graph
    for (u,v),e in g.edges.items():
        p,q=g.nodes[u],g.nodes[v]
        lw=0.7 + (2.0*np.tanh(e.traffic/3.0) if usage else 2.0*np.tanh(e.weight))
        ax.plot([p[0],q[0]],[p[1],q[1]],linewidth=lw,alpha=0.8)
    pts=np.vstack([g.nodes[n] for n in sorted(g.nodes)])
    ax.scatter(pts[:,0],pts[:,1],s=10)
    soma=g.nodes[g.soma_node_id]
    ax.scatter([soma[0]],[soma[1]],s=50,marker='s')


def render_snapshot(record: RunRecord, layer: str, output_path=None):
    if layer not in LAYERS:
        raise ValueError(f'unknown layer {layer!r}; choose one of {LAYERS}')
    fig,ax=plt.subplots(figsize=(6,6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.set_title(layer)
    if layer=='field':
        data=record.field_data if record.field_data is not None else np.zeros((V0.grid_size,V0.grid_size))
        ax.imshow(np.asarray(data),origin='lower',extent=(0,1,0,1),aspect='auto')
    elif layer in ('graph','usage'):
        _draw_graph(ax,record,usage=(layer=='usage'))
    elif layer.startswith('mode'):
        idx=int(layer[-1])-1
        _draw_graph(ax,record,usage=False)
        ids=record.spectrum.node_ids
        pts=np.vstack([record.graph.nodes[n] for n in ids])
        if idx < record.spectrum.eigenvectors.shape[1]:
            vals=record.spectrum.eigenvectors[:,idx]
        else:
            vals=np.zeros(len(ids))
        ax.scatter(pts[:,0],pts[:,1],c=vals,s=28)
    elif layer=='probe':
        _draw_graph(ax,record,usage=False)
        ids=record.probe.node_ids
        pts=np.vstack([record.graph.nodes[n] for n in ids])
        vals=record.probe.states[-1]
        ax.scatter(pts[:,0],pts[:,1],c=vals,s=28)
    elif layer=='lesion':
        _draw_graph(ax,record,usage=False)
        ax.axvline(V0.lesion_x,alpha=0.8)
        ax.fill_betweenx([V0.lesion_y_min,1.0],V0.lesion_x-0.01,V0.lesion_x+0.01,alpha=0.2)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    if output_path is not None:
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=140)
    return fig
