# Developmental Spectral Neuron v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic local developmental simulator in which temporal history grows a sparse weighted graph, then test whether that grown anatomy changes later diffusion, retains decodable history, develops task-aligned low-order structure, and responds causally to a predeclared lesion/regrowth assay.

**Architecture:** A 2-D synthetic world drives local resonant growth tips with fast activity, medium resource, and slow structural persistence. Growth maintains both a stigmergic field and an explicit weighted graph; downstream modules compute Laplacian modes, matched diffusion probes, destructive controls, blind history readback, task-alignment/null comparisons, and lesion/regrowth receipts. The scientific runner freezes one v0 parameter set before multi-seed runs and never lets growth access global graph or spectral quantities.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib (visualizer only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-developmental-spectral-neuron-v0-design.md`

## Global Constraints

- Growth tips may use only local cue/resource/stigmergic neighborhoods plus their own state; no global graph metric, eigenmode, future lesion location, or target topology may influence growth.
- Keep the stigmergic `field` separate from the explicit computational graph.
- Three clocks have distinct jobs: fast resonant state, medium resource balance, slow edge strength/survival.
- No gradient descent and no learned recurrent weights in v0.
- The same frozen parameter set is used for `H_A`, `H_B`, lesion runs, and all multi-seed scientific runs.
- `H_A` and `H_B` use the same spatial world and equal left/right exposure counts; they differ only in temporal ordering of resource epochs.
- First frozen scientific seed set: `0..7` (8 seeds per history/control arm).
- Scientific gates may fail; failed gates are written to receipts and are not tuned away in the same run series.
- Biological language stays within the spec claim boundary; variables remain `resource`, `activity`, and `structure`.
- First propagation model is low-frequency graph diffusion only; richer windowed/resonant propagation is deferred.

## Frozen v0 constants

All values live in immutable `V0Config` before the first multi-seed receipt:

- domain `[0,1] x [0,1]`, world/field grid `64 x 64`
- soma `(0.50, 0.08)`
- left target `(0.25, 0.78)`, right target `(0.75, 0.78)`, Gaussian sigma `0.11`
- static cue field: equal attraction to both targets plus soft boundary repulsion; identical for both histories
- development: `12` epochs x `50` steps = `600` steps
- `H_A = L,R,L,R,L,R,L,R,L,R,L,R`
- `H_B = L,L,L,L,L,L,R,R,R,R,R,R`
- initial tips `4`; maximum active tips `48`
- graph-node spacing `0.025`; local merge radius `0.035`; tip step `0.012`
- field decay `rho_F = 0.997`
- resonator persistence palette `{0.86, 0.90, 0.94, 0.97}` cyclic at birth
- resonator angular-frequency palette `{0.08, 0.12, 0.18, 0.26}` rad/tick cyclic at birth
- resource `E0=1.0`, floor `0.0`, cap `2.0`
- maintenance `c0=0.0015`; activity cost `c1=0.0008`; harvest gain `eta=0.012`
- branch threshold `1.35`; minimum productive age `35`; daughter heading perturbation `±0.35` rad seeded
- edge initial weight `0.30`; productive reinforcement `+0.010`; per-step decay `0.9995`
- prune threshold `0.06`; prune grace `120`
- movement weights: cue `1.0`, field `0.35`, resource-gradient `0.55`, curvature `0.15`; exploration SD `0.08` before direction normalization
- diffusion beta `0.75`; score times `[0.0, 0.25, 0.5, 1.0, 2.0]` using `scipy.linalg.expm`
- smallest `6` nontrivial modes when graph size permits
- lesion: edges crossing `x=0.50` with both endpoints `y>=0.55`
- regrowth `200` steps under balanced alternating resource schedule and the same local rules; no repair signal

If Gate 0 proves this set cannot create any viable soma-connected structure, a revised `V0Config` requires a documented engineering-fix commit and **all** scientific receipts restart from seed 0. Never tune per history or on lesion outcome.

## Review Focus

1. **Pruning disconnects live structure from soma:** Task 3 pins atomic retirement/reconnection behavior; no orphan active tips.
2. **Near-degenerate eigenvalues reorder/rotate modes:** Task 4 uses sign-invariant comparison and projector/subspace distance for clustered modes.
3. **Geometry-null changes wiring cost or connectivity:** Task 6 preserves node coordinates, edge count, approximate length histogram, nonnegative weights, and soma connectivity.
4. **Disconnected/invalid graphs fool diffusion scoring:** Task 5 validates graph/Laplacian assumptions and explicitly reports components.
5. **Lesion selection becomes post-hoc:** Task 7 accepts only the frozen geometric lesion band and never reads centrality/eigenmodes/damage magnitude.

---

### Task 1: Package, frozen configuration, and matched histories

**Files:**
- Create: `pyproject.toml`
- Create: `src/devspectral/__init__.py`
- Create: `src/devspectral/config.py`
- Create: `src/devspectral/world.py`
- Test: `tests/test_world.py`

**Interfaces:**
- `V0Config` frozen dataclass; `V0 = V0Config()`
- `HistoryId = Literal["H_A", "H_B"]`
- `epoch_schedule(history, config) -> tuple[str, ...]`
- `DevelopmentalWorld(config)` with `cue_and_gradient(pos)`, `resource_and_gradient(pos, step, history)`, `stimulus(pos, step, history)`

- [ ] Write tests asserting all frozen constants, exact schedules, six L/six R exposures in each history, identical static cue samples, and deterministic stimulus values.
- [ ] Run `pytest tests/test_world.py -q`; expect RED because package/world do not exist.
- [ ] Implement package metadata, frozen config, analytic Gaussian fields/gradients, schedules, and deterministic localized sinusoidal stimulus. History ID must not be exposed as a direct feature.
- [ ] Run `pytest tests/test_world.py -q`; expect PASS.
- [ ] Commit: `feat: define frozen developmental world`.

### Task 2: Explicit weighted graph and local growth state

**Files:**
- Create: `src/devspectral/graph.py`
- Create: `src/devspectral/growth.py`
- Test: `tests/test_growth_core.py`

**Interfaces:**
- `GraphState`: positions, undirected weighted edges, edge age/traffic/last-use, soma id, connectivity helpers
- `GrowthTip`: `tip_id`, `node_id`, `position`, `heading`, `q:(2,)`, `resource`, `age`, `productive_age`, `parent_tip_id`, `alive`
- `DevelopmentSimulator(config, history, seed, control="full")` with `step()`, `run(steps=None)`, `snapshot()`

- [ ] Write tests for exact two-state resonator rotation, resource clipping `[0,2]`, movement step bound, deterministic RNG, graph symmetry/nonnegative weights, and a direction-selection signature that receives only local values and tip state—not graph-spectrum objects.
- [ ] Run `pytest tests/test_growth_core.py -q`; expect RED.
- [ ] Implement `GraphState`, `GrowthTip`, one-step resonator/resource dynamics, and local direction combination. Keep one `np.random.Generator` per simulator.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `feat: add local resonant growth core`.

### Task 3: Stigmergic field, branching, reinforcement, pruning, Gate-0 invariants

**Files:**
- Create: `src/devspectral/field.py`
- Modify: `src/devspectral/growth.py`
- Modify: `src/devspectral/graph.py`
- Test: `tests/test_structure.py`

**Interfaces:**
- `StructuralField(config)` with `sample_and_gradient`, `decay`, `deposit_segment`
- extended `DevelopmentSimulator.step()` for deposition/reinforcement/branch/merge/prune
- `validate_invariants() -> list[str]`

- [ ] Write tests for field deposition/decay, bounded field attraction, branch cap, local merge radius, edge reinforcement/decay/pruning, graph indices, nonnegative weights, and all active structural components connected to soma.
- [ ] Add Review-Focus regression: force a below-threshold edge supporting an active descendant; pruning must retain the path or retire/re-home dependent tips atomically—never orphan them.
- [ ] Add determinism regression: identical seed+history for a short run yields byte-equivalent serialized graph/field/tips; different seed yields a finite valid but not necessarily identical state.
- [ ] Run `pytest tests/test_structure.py -q`; expect RED.
- [ ] Implement bilinear field sampling, Gaussian segment deposition, local branch/merge, edge usage/reinforcement, grace-period pruning, invariant validation.
- [ ] Run Tasks 1-3 tests; expect PASS.
- [ ] Commit: `feat: grow and prune stigmergic anatomy`.

### Task 4: Laplacian, eigenmodes, robust spectral comparison

**Files:**
- Create: `src/devspectral/spectral.py`
- Test: `tests/test_spectral.py`

**Interfaces:**
- `weighted_adjacency(graph) -> np.ndarray`
- `laplacian(graph) -> np.ndarray`
- `low_modes(graph, k=6) -> Spectrum`
- `mode_alignment(a,b)`, `subspace_distance(U,V)`, `spectral_signature(spectrum)`

- [ ] Write tests on hand-built path/cycle/two-lobe graphs: symmetric PSD Laplacian, connected graph has one zero eigenvalue, positive `lambda2`, sorted values, sign-invariant alignment.
- [ ] Add near-degenerate regression: rotate basis inside a repeated eigenspace; projector/subspace distance must remain ~0 although naive column comparison changes.
- [ ] Run `pytest tests/test_spectral.py -q`; expect RED.
- [ ] Implement with `scipy.linalg.eigh`; group adjacent modes when eigenvalue gap `<1e-7*max(1,|lambda|)` and compare grouped subspaces with projectors/principal angles.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `feat: add robust graph spectral analysis`.

### Task 5: Matched post-development diffusion probes

**Files:**
- Create: `src/devspectral/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- `ProbeSpec(source_center, source_radius, readout_center, readout_radius)`
- `diffusion_trajectory(graph, probe, times, beta) -> ProbeResult`
- `probe_distance(a,b) -> float`

- [ ] Write tests for identical-graph reproducibility, finite trajectories, exact mass conservation within tolerance for connected undirected graphs, and expected slower transfer on a path than on a path plus shortcut.
- [ ] Add disconnected/invalid Review-Focus tests: component metadata required; no false cross-component arrival; asymmetric/negative-weight malformed input raises `ValueError`.
- [ ] Run `pytest tests/test_probe.py -q`; expect RED.
- [ ] Implement exact `expm(-beta*L*t)` propagation. Freeze source around left target and readout around right target independent of history/seed.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `feat: add matched diffusion probe`.

### Task 6: Destructive controls and geometry-matched nulls

**Files:**
- Create: `src/devspectral/controls.py`
- Modify: `src/devspectral/growth.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- developmental arms: `full`, `no_stigmergy`, `no_metabolic_selection`, `no_pruning`
- graph arms: `geometry_null(graph, seed)`, `weight_shuffle(graph, seed)`, `fixed_lattice(config, node_budget, edge_budget, seed)`

- [ ] Test that each developmental control disables only its declared mechanism: no-stigmergy ignores field gradient but may still deposit; no-metabolic-selection removes usefulness dependence from continuation/branch opportunity but preserves costs/geometry; no-pruning decays weights but suppresses deletion.
- [ ] Geometry-null tests: same node coordinates, edge count, nonnegative weights, soma connectivity, and fixed 8-bin edge-length histogram counts within one edge of source per bin over `[0,sqrt(2)]`.
- [ ] Weight-shuffle test preserves topology and weight multiset. Fixed-lattice test respects requested node/edge budget within construction feasibility.
- [ ] Run `pytest tests/test_controls.py -q`; expect RED.
- [ ] Implement seeded controls. Geometry null gets at most 200 reconstruction attempts then raises `RuntimeError`; never silently relax matching constraints.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `feat: add developmental and graph null controls`.

### Task 7: Predeclared lesion and local regrowth

**Files:**
- Create: `src/devspectral/lesion.py`
- Test: `tests/test_lesion.py`

**Interfaces:**
- `lesion_midline_band(graph, config) -> LesionResult`
- `resume_regrowth(simulator, steps=200) -> DevelopmentSnapshot`

- [ ] Write Review-Focus test with high-centrality edges inside/outside lesion band; only geometrically qualifying midline edges are removed, unchanged by attached/permuted centrality or spectral metadata.
- [ ] On hand-built two-lobe bridge graph, lesion must lower `lambda2` and left-to-right transfer.
- [ ] Run `pytest tests/test_lesion.py -q`; expect RED.
- [ ] Implement lesion from frozen geometry only. Regrowth reuses original local rules and balanced alternating schedule; no repair signal or lesion-location feature. Record new band-crossing edges.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `feat: add predeclared lesion and local regrowth`.

### Task 8: Scientific metrics, Gate 0-5 receipts, history readback, and spectral compactness

**Files:**
- Create: `src/devspectral/experiment.py`
- Create: `src/devspectral/receipt.py`
- Create: `scripts/run_v0.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- `run_one(history, seed, control="full") -> RunRecord`
- `run_gate_suite(seeds=range(8)) -> GateReceipt`
- `history_features(record) -> np.ndarray`
- `nearest_centroid_leave_one_seed_out(records) -> ReadbackResult`
- `task_alignment(record) -> AlignmentResult`
- JSON-safe receipts and compact graph serialization (`positions`, edge pairs, weights)

- [ ] Write two-seed end-to-end RED smoke tests: deterministic same-seed equality; valid invariants; serializable graph/spectrum/probe; readback features contain anatomy/spectrum only, not developmental schedule/log labels.
- [ ] Pin **Gate 0**: for every arm/seed, record invariant failures, component count, NaN count, negative weight count, orphan tip count, graph size. Gate 0 passes only when all full-arm runs are valid and rerunning one fixed seed/history reproduces graph+spectrum exactly.
- [ ] Pin **Gate 1** developmental specificity. Structural feature vector = normalized first six nontrivial eigenvalues (zero padded) + node/edge count + total edge length + mean/SD edge weight + fixed `4x4` spatial edge-density histogram. `structural_separation_ratio = median(between) / max(eps,median(within))`. Pass when at least 6/8 matched cross-history distances exceed the 75th percentile of pooled within-history distances and ratio >1.
- [ ] Pin **Gate 2** same-present/different-history operator test. Compute probe separation on flattened fixed-time trajectories; pass with the same 6/8-vs-within rule. Zeroing all tip fast states before probing must leave the graph-based probe result unchanged. A graph-swap helper must demonstrate that response follows graph, not history label.
- [ ] Pin **Gate 3** controls. A control materially weakens an effect if its Gate-1 or Gate-2 separation ratio is at least 25% lower than full. Overall scientific status cannot be `supported` unless at least one declared destructive control weakens a primary effect.
- [ ] Pin **Gate 4** blind readback. Features are the structural vector above; leave-one-seed-out nearest-centroid accuracy `>=0.75` marks readback pass. Require at least one destructive control to reduce readback accuracy or record the readback evidence as non-specific.
- [ ] Pin **Gate 5** lesion/regrowth. Per seed record pre-lesion/lesion/regrowth `lambda2`, component count, transfer, and new crossing edges. Lesion direction is observed when both `lambda2` and transfer decrease in at least 6/8 full-arm seeds. Regrowth status is `partial_recovery`, `no_recovery`, or `not_applicable`; no-recovery is a scientific result, not a software failure.
- [ ] Add **spectral compactness/task-alignment test** from spec section 18. Define fixed task function `g(x,y)=sign(x-0.5)` on graph nodes (left/right axis, predeclared from world geometry). Measure squared projection fraction of centered `g` into first `k=3` nontrivial modes. Compare full developed graphs to `geometry_null` and `weight_shuffle` for the same graph/seed. Report `alignment_gain = full_alignment - median(null_alignments)` and within-history subspace stability using projector distance. Never use diffusion reconstruction alone as evidence because that would be tautological.
- [ ] Run `pytest tests/test_experiment.py -q`; expect RED.
- [ ] Implement runner/receipt schema and all metrics without parameter search.
- [ ] Run `pytest -q`; expect all PASS.
- [ ] Commit: `feat: add frozen scientific gate suite`.

### Task 9: Diagnostic visualizer and documentation

**Files:**
- Create: `src/devspectral/visualize.py`
- Create: `scripts/visualize_run.py`
- Create: `README.md`
- Test: `tests/test_visualize.py`

**Interfaces:**
- `render_snapshot(record, layer, output_path=None)` for `field`, `graph`, `usage`, `mode1`, `mode2`, `mode3`, `probe`, `lesion`

- [ ] With Matplotlib `Agg`, write RED smoke tests rendering every layer from a tiny record and asserting no mutation of source data.
- [ ] Implement two-pane diagnostic renderer: developmental field/graph on left or source snapshot; graph/modes/probe/lesion layer on right. Visualization is never a gate.
- [ ] README must state claim boundary, exact H_A/H_B schedules, frozen-v0 discipline, run commands, and that spectral pictures alone are not evidence.
- [ ] Run `pytest -q`; expect PASS.
- [ ] Commit: `docs: add developmental spectral neuron visualizer`.

### Task 10: Freeze and run first v0 scientific receipt

**Files:**
- Create: `results/v0_receipt.json`
- Create: `docs/V0_STATUS.md`
- Modify: `README.md`

- [ ] Fresh software gate: run `python -m compileall -q src scripts && pytest -q`; require exit 0.
- [ ] Run exactly once on the frozen set: `python scripts/run_v0.py --output results/v0_receipt.json` over seeds `0..7` and all declared controls/nulls.
- [ ] Validate without re-running development: `python scripts/run_v0.py --check-receipt results/v0_receipt.json`; require schema/invariant/config-hash PASS.
- [ ] Write `docs/V0_STATUS.md` directly from receipt: Gate 0-5 outcomes, spectral alignment/null result, controls, lesion/regrowth, what died, what survived, biological claim boundary.
- [ ] Do **not** tune after reading results. A genuine engineering bug requires a documented fix commit, invalidation of the old receipt, and a complete fresh run from seed 0.
- [ ] Run final `python -m compileall -q src scripts && pytest -q`; require exit 0.
- [ ] Commit: `results: freeze developmental spectral neuron v0`.

## Execution notes

- Implement on an isolated feature branch/worktree created from the approved-plan commit.
- Use RED -> GREEN TDD for every task.
- Do not tune scientific constants after Task 8 except under the explicit Gate-0 engineering-fix rule; a change restarts all multi-seed receipts.
- Preserve negative outcomes exactly. A software-complete branch may legitimately conclude that the developmental mechanism is unsupported under v0.
- Before handoff or merge, whole-branch review must specifically check hidden global information in local growth, control fairness, history leakage into readback/probes, spectral comparison under degeneracy, post-hoc lesion selection, and numerical diffusion errors.
