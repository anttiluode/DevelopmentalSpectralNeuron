from __future__ import annotations
from dataclasses import asdict
import hashlib, json, math, subprocess
import numpy as np
from .config import V0, V0Config
from .controls import DEVELOPMENTAL_CONTROLS, geometry_null, weight_shuffle, fixed_lattice
from .growth import DevelopmentSimulator
from .graph import GraphState
from .lesion import lesion_midline_band, resume_regrowth
from .probe import ProbeSpec, diffusion_trajectory
from .receipt import RunRecord, SeparationResult, ReadbackResult, GateReceipt, serialize_graph
from .spectral import low_modes, spectral_signature


def config_hash(config: V0Config) -> str:
    payload=json.dumps(asdict(config),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _edge_density_hist(graph: GraphState, bins: int=4, weighted: bool=True) -> np.ndarray:
    h=np.zeros((bins,bins),float)
    for (u,v),e in graph.edges.items():
        mid=(graph.nodes[u]+graph.nodes[v])/2.0
        ix=min(bins-1,max(0,int(mid[0]*bins)))
        iy=min(bins-1,max(0,int(mid[1]*bins)))
        h[iy,ix] += e.weight if weighted else 1.0
    total=float(h.sum())
    if total>0: h/=total
    return h.ravel()


def _probe_spatial_signature(graph: GraphState, states: np.ndarray, bins: int=4) -> np.ndarray:
    ids=tuple(sorted(graph.nodes))
    out=[]
    for row in states:
        h=np.zeros((bins,bins),float)
        for i,n in enumerate(ids):
            p=graph.nodes[n]
            ix=min(bins-1,max(0,int(p[0]*bins)))
            iy=min(bins-1,max(0,int(p[1]*bins)))
            h[iy,ix]+=float(row[i])
        out.extend(h.ravel())
    return np.asarray(out,float)


def _probe_spec(config: V0Config) -> ProbeSpec:
    return ProbeSpec(config.left_target,config.target_sigma,config.right_target,config.target_sigma)


def record_from_graph(graph: GraphState, history: str, seed: int, control: str, config: V0Config=V0, invariants: list[str] | None=None, field_data=None) -> RunRecord:
    spec=low_modes(graph,k=config.spectral_modes)
    probe=diffusion_trajectory(graph,_probe_spec(config),config.diffusion_times,config.diffusion_beta)
    psig=_probe_spatial_signature(graph,probe.states)
    return RunRecord(history,int(seed),control,graph.copy(),spec,probe,psig,list(invariants or []),config_hash(config),None if field_data is None else np.asarray(field_data,float).copy())


def _run_one_with_sim(history: str, seed: int, control: str='full', config: V0Config=V0):
    sim=DevelopmentSimulator(config,history,seed,control=control)
    sim.run()
    rec=record_from_graph(sim.graph,history,seed,control,config,sim.validate_invariants(),sim.field.data)
    return sim,rec


def run_one(history: str, seed: int, control: str='full') -> RunRecord:
    return _run_one_with_sim(history,seed,control,V0)[1]


def history_features(record: RunRecord) -> np.ndarray:
    eig=np.zeros(6,float); n=min(6,len(record.spectrum.eigenvalues)); eig[:n]=record.spectrum.eigenvalues[:n]
    lengths=[]; weights=[]
    for (u,v),e in record.graph.edges.items():
        lengths.append(float(np.linalg.norm(record.graph.nodes[u]-record.graph.nodes[v])))
        weights.append(float(e.weight))
    scalar=np.array([
        len(record.graph.nodes),len(record.graph.edges),sum(lengths),
        float(np.mean(weights)) if weights else 0.0,
        float(np.std(weights)) if weights else 0.0,
    ],float)
    return np.concatenate([eig,scalar,_edge_density_hist(record.graph,bins=4,weighted=True)])


def _structural_vector(record: RunRecord) -> np.ndarray:
    eig=spectral_signature(record.spectrum,6)
    nz=eig[eig>0]
    scale=float(np.mean(nz)) if len(nz) else 1.0
    eig=eig/max(scale,1e-12)
    return np.concatenate([eig,_edge_density_hist(record.graph,4,weighted=True)])


def nearest_centroid_leave_one_seed_out(records: list[RunRecord]) -> ReadbackResult:
    records=[r for r in records if r.history in ('H_A','H_B')]
    predictions=[]; truths=[]; correct=0; total=0
    for seed in sorted({r.seed for r in records}):
        train=[r for r in records if r.seed!=seed]
        test=[r for r in records if r.seed==seed]
        if not train or {r.history for r in train}!={'H_A','H_B'}: continue
        X=np.vstack([history_features(r) for r in train])
        mean=X.mean(axis=0); std=X.std(axis=0); std[std<1e-9]=1.0
        centroids={h:np.mean([(history_features(r)-mean)/std for r in train if r.history==h],axis=0) for h in ('H_A','H_B')}
        for r in test:
            x=(history_features(r)-mean)/std
            pred=min(centroids,key=lambda h:float(np.linalg.norm(x-centroids[h])))
            predictions.append(pred); truths.append(r.history); total+=1; correct+=int(pred==r.history)
    acc=correct/total if total else 0.0
    return ReadbackResult(acc,correct,total,tuple(predictions),tuple(truths),bool(total and acc>=0.75))


def separation_gate(a: dict[int,np.ndarray], b: dict[int,np.ndarray]) -> SeparationResult:
    common=sorted(set(a)&set(b))
    cross=[float(np.linalg.norm(np.asarray(a[s])-np.asarray(b[s]))) for s in common]
    within=[]
    for group in (a,b):
        seeds=sorted(group)
        for i in range(len(seeds)):
            for j in range(i+1,len(seeds)):
                within.append(float(np.linalg.norm(np.asarray(group[seeds[i]])-np.asarray(group[seeds[j]]))))
    if not cross or not within:
        return SeparationResult(0.0,float('nan'),len(cross),0,max(1,math.ceil(0.75*len(cross))),False,float(np.median(within)) if within else 0.0,float(np.median(cross)) if cross else 0.0)
    medw=float(np.median(within)); medc=float(np.median(cross)); threshold=float(np.percentile(within,75))
    count=sum(d>threshold for d in cross); required=math.ceil(0.75*len(cross)); ratio=medc/max(1e-12,medw)
    return SeparationResult(ratio,threshold,len(cross),count,required,bool(count>=required and ratio>1.0),medw,medc)


def task_alignment(graph: GraphState, spectrum) -> float:
    if spectrum.eigenvectors.size==0: return 0.0
    ids=tuple(sorted(graph.nodes)); f=np.array([graph.nodes[n][0]-0.5 for n in ids],float); f-=f.mean()
    denom=float(f@f)
    if denom<1e-12: return 0.0
    U=spectrum.eigenvectors[:,:min(3,spectrum.eigenvectors.shape[1])]
    score=float(np.linalg.norm(U.T@f)**2/denom)
    return float(np.clip(score,0.0,1.0))


def _vectors(records: list[RunRecord], which: str):
    out={'H_A':{},'H_B':{}}
    for r in records:
        if r.history not in out: continue
        out[r.history][r.seed]=_structural_vector(r) if which=='structural' else r.probe_signature
    return out


def _git_commit() -> str | None:
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def _graph_control_records(full_records: list[RunRecord], names: tuple[str,...], config: V0Config):
    records=[]; errors=[]
    for r in full_records:
        for name in names:
            try:
                if name=='geometry_null': g=geometry_null(r.graph,r.seed+1000)
                elif name=='weight_shuffle': g=weight_shuffle(r.graph,r.seed+2000)
                elif name=='fixed_lattice': g=fixed_lattice(config,len(r.graph.nodes),len(r.graph.edges),r.seed+3000)
                else: raise ValueError(f'unknown graph control {name}')
                records.append(record_from_graph(g,r.history,r.seed,name,config))
            except Exception as exc:
                errors.append({'control':name,'history':r.history,'seed':r.seed,'error':f'{type(exc).__name__}: {exc}'})
    return records,errors


def _separation_for(records: list[RunRecord], which: str) -> SeparationResult:
    v=_vectors(records,which); return separation_gate(v['H_A'],v['H_B'])


def _spectral_alignment_gate(full_records: list[RunRecord], config: V0Config):
    pairs=[]; errors=[]
    for r in full_records:
        try:
            null=geometry_null(r.graph,r.seed+9000)
            ns=low_modes(null,k=config.spectral_modes)
            pairs.append((task_alignment(r.graph,r.spectrum),task_alignment(null,ns)))
        except Exception as exc:
            errors.append({'history':r.history,'seed':r.seed,'error':f'{type(exc).__name__}: {exc}'})
    wins=sum(a>b for a,b in pairs); required=math.ceil(0.75*len(pairs)) if pairs else 1
    return {'pairs':len(pairs),'full_gt_null':wins,'required':required,'passed':bool(pairs and wins>=required),
            'median_full':float(np.median([a for a,_ in pairs])) if pairs else None,
            'median_null':float(np.median([b for _,b in pairs])) if pairs else None,'errors':errors}


def _lesion_regrowth(full_sims: dict[tuple[str,int],DevelopmentSimulator], full_records: list[RunRecord], seeds: tuple[int,...], config: V0Config):
    rows=[]; direction=0
    recmap={(r.history,r.seed):r for r in full_records}
    for seed in seeds:
        key=('H_A',seed)
        if key not in full_sims: continue
        sim=full_sims[key]
        pre=recmap[key]
        pre_l2=float(pre.spectrum.all_eigenvalues[1]) if len(pre.spectrum.all_eigenvalues)>1 else 0.0
        pre_transfer=float(pre.probe.transfer[-1])
        lesion=lesion_midline_band(sim.graph,config)
        if not lesion.removed_edges:
            rows.append({'seed':seed,'removed_edges':0,'status':'not_applicable','pre_lambda2':pre_l2,'pre_transfer':pre_transfer})
            continue
        lesrec=record_from_graph(lesion.graph,'H_A',seed,'lesion',config)
        les_l2=float(lesrec.spectrum.all_eigenvalues[1]) if len(lesrec.spectrum.all_eigenvalues)>1 else 0.0
        les_transfer=float(lesrec.probe.transfer[-1])
        direction += int(les_l2 < pre_l2 and les_transfer < pre_transfer)
        sim.graph=lesion.graph
        resume_regrowth(sim,config.regrowth_steps)
        reg=record_from_graph(sim.graph,'H_A',seed,'regrowth',config,sim.validate_invariants(),sim.field.data)
        reg_l2=float(reg.spectrum.all_eigenvalues[1]) if len(reg.spectrum.all_eigenvalues)>1 else 0.0
        reg_transfer=float(reg.probe.transfer[-1])
        new_cross=len(lesion_midline_band(reg.graph,config).removed_edges)
        if reg_l2>les_l2+1e-12 or reg_transfer>les_transfer+1e-12:
            status='partial_recovery'
        else: status='no_recovery'
        rows.append({'seed':seed,'removed_edges':len(lesion.removed_edges),'pre_lambda2':pre_l2,'lesion_lambda2':les_l2,
                     'regrowth_lambda2':reg_l2,'pre_transfer':pre_transfer,'lesion_transfer':les_transfer,
                     'regrowth_transfer':reg_transfer,'new_band_crossing_edges':new_cross,'status':status})
    applicable=sum('lesion_lambda2' in r for r in rows); required=math.ceil(0.75*applicable) if applicable else 1
    return {'runs':rows,'direction_observed':direction,'applicable':applicable,'required':required,'passed':bool(applicable and direction>=required)}


def classify_overall(*, gate0_passed: bool, structural_passed: bool, probe_passed: bool, readback_passed: bool, alignment_passed: bool, lesion_passed: bool, any_material_weakening: bool) -> str:
    if not gate0_passed:
        return 'engineering_failure'
    primary = (structural_passed, probe_passed, readback_passed)
    if not any(primary):
        return 'negative'
    if all(primary) and alignment_passed and lesion_passed and any_material_weakening:
        return 'supported'
    return 'inconclusive'


def run_gate_suite(seeds=range(8), developmental_controls=DEVELOPMENTAL_CONTROLS, graph_controls=('geometry_null','weight_shuffle','fixed_lattice')) -> GateReceipt:
    seeds=tuple(int(s) for s in seeds); controls=tuple(developmental_controls); graph_controls=tuple(graph_controls)
    records=[]; full_sims={}
    for control in controls:
        for history in ('H_A','H_B'):
            for seed in seeds:
                sim,rec=_run_one_with_sim(history,seed,control,V0)
                records.append(rec)
                if control=='full': full_sims[(history,seed)]=sim
    full=[r for r in records if r.control=='full']
    if not full: raise ValueError('full control arm is required')
    # Gate 0: exact deterministic replay of one run plus validity of every full run.
    ref=full[0]; repeat=run_one(ref.history,ref.seed,'full')
    deterministic=(serialize_graph(ref.graph)==serialize_graph(repeat.graph) and np.allclose(ref.probe.transfer,repeat.probe.transfer))
    valid=sum((not r.invariants) and r.spectrum.component_count==1 for r in full)
    gate0={'deterministic':bool(deterministic),'valid_runs':int(valid),'total_full_runs':len(full),'passed':bool(deterministic and valid==len(full))}
    structural=_separation_for(full,'structural'); probe=_separation_for(full,'probe'); readback=nearest_centroid_leave_one_seed_out(full)
    alignment=_spectral_alignment_gate(full,V0)
    graph_records,graph_errors=_graph_control_records(full,graph_controls,V0)
    records.extend(graph_records)
    controls_summary={'errors':graph_errors,'arms':{}}
    for name in tuple(c for c in controls if c!='full')+graph_controls:
        arm=[r for r in records if r.control==name]
        if not arm: continue
        sg=_separation_for(arm,'structural'); pg=_separation_for(arm,'probe'); rb=nearest_centroid_leave_one_seed_out(arm)
        weakening=bool((structural.ratio>0 and sg.ratio<=0.75*structural.ratio) or (probe.ratio>0 and pg.ratio<=0.75*probe.ratio))
        controls_summary['arms'][name]={'structural':sg.to_dict(),'probe':pg.to_dict(),'readback':rb.to_dict(),'materially_weakens':weakening}
    controls_summary['any_material_weakening']=any(v['materially_weakens'] for v in controls_summary['arms'].values())
    lesion=_lesion_regrowth(full_sims,full,seeds,V0)
    status=classify_overall(
        gate0_passed=gate0['passed'],
        structural_passed=structural.passed,
        probe_passed=probe.passed,
        readback_passed=readback.passed,
        alignment_passed=alignment['passed'],
        lesion_passed=lesion['passed'],
        any_material_weakening=controls_summary['any_material_weakening'],
    )
    return GateReceipt(config_hash(V0),asdict(V0),seeds,gate0,structural,probe,readback,alignment,controls_summary,lesion,status,records,_git_commit())
