# Developmental Spectral Neuron v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic local developmental simulator in which temporal history grows a sparse weighted graph, then show whether that grown anatomy changes later diffusion, retains decodable history, and responds causally to a predeclared lesion/regrowth assay.

**Architecture:** A 2-D synthetic world drives local resonant growth tips with fast activity, medium resource, and slow structural persistence. Growth maintains both a stigmergic field and an explicit weighted graph; downstream modules compute Laplacian modes, matched diffusion probes, destructive controls, blind history readback, and lesion/regrowth receipts. The scientific runner freezes one v0 parameter set before multi-seed runs and never lets growth access global graph or spectral quantities.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib (visualizer only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-developmental-spectral-neuron-v0-design.md`

## Global Constraints

- Growth tips may use only local cue/resource/stigmergic neighborhoods plus their own state; no global graph metric, eigenmode, future lesion location, or target topology may influence growth.
- Keep `field` (local stigmergic guidance/visualization) separate from the explicit computational graph.
- Three clocks have distinct jobs: fast resonant state, medium resource balance, slow edge strength/survival.
- No gradient descent and no learned recurrent weights in v0.
- The same frozen parameter set is used for `H_A`, `H_B`, lesion runs, and all multi-seed scientific runs.
- `H_A` and `H_B` use the same spatial world and equal left/right exposure counts; they differ only in temporal ordering of resource epochs.
- The first frozen scientific seed set is `0..7` (8 seeds per history/control arm).
- Scientific gates may fail; failed gates are written to receipts and are not tuned away in the same run series.
- Biological language stays within the claim boundary in the spec; variables are `resource`, `activity`, `structure`, not ATP/dopamine/neurotrophin analogues.
- The first propagation model is low-frequency graph diffusion only; richer windowed/resonant propagation is explicitly deferred.

## Frozen v0 constants

These values are fixed before the first multi-seed receipt and live in one immutable `V0Config` object:

- world grid: `64 x 64`, normalized continuous domain `[0,1] x [0,1]`
- soma/root: `(0.50, 0.08)`
- left target: `(0.25, 0.78)`, right target: `(0.75, 0.78)`, Gaussian sigma `0.11`
- static cue field: equal attraction to both target Gaussians plus soft boundary repulsion; identical for both histories
- development: `12` epochs x `50` steps = `600` steps
- `H_A`: alternating target schedule `L,R,L,R,...` for 12 epochs
- `H_B`: blocked schedule `L,L,L,L,L,L,R,R,R,R,R,R`
- both histories therefore contain exactly six left and six right resource epochs
- initial tips: `4`, headings symmetrically spread around upward direction
- maximum active tips: `48`
- graph-node spacing along a branch: `0.025`
- local merge radius: `0.035`
- tip step length: `0.012`
- field grid: same `64 x 64` raster as world
- field decay `rho_F = 0.997`
- resonator persistence palette `r in {0.86, 0.90, 0.94, 0.97}` assigned cyclically at birth
- resonator angular-frequency palette `omega in {0.08, 0.12, 0.18, 0.26}` radians/tick assigned cyclically at birth
- resource initial `E0 = 1.0`, floor `0.0`, cap `2.0`
- maintenance cost `c0 = 0.0015`, activity cost `c1 = 0.0008`
- harvest gain `eta = 0.012`
- branch threshold `E_branch = 1.35`, minimum productive age `35` steps, daughter heading perturbation `±0.35` rad seeded
- edge initial weight `0.30`, productive reinforcement `+0.010`, per-step decay `0.9995`
- prune threshold `0.06`, prune grace `120` steps
- field-follow weight `w_f = 0.35`, cue weight `w_c = 1.0`, resource-gradient weight `w_r = 0.55`, curvature weight `w_k = 0.15`, exploration noise SD `0.08` before direction normalization
- diffusion beta `0.75`; score at fixed times `t = [0.0, 0.25, 0.5, 1.0, 2.0]` using `scipy.linalg.expm`
- spectral report: smallest `6` nontrivial modes when graph size permits
- predeclared lesion band: edges crossing the vertical midline `x=0.50` with both endpoints at `y >= 0.55`
- regrowth: `200` additional steps under the same local rules and a balanced alternating resource schedule; no special repair signal

If Gate 0 shows that this set cannot create any viable soma-connected structure, a revised `V0Config` requires a documented engineering-fix commit and all scientific receipts restart from seed 0. No per-history tuning is allowed.

## Review Focus

1. **Pruning disconnects live structure from the soma:** graph mutation must never leave an active tip attached to a deleted/orphan node; tests in Task 3 pin this.
2. **Near-degenerate eigenvalues make eigenvectors reorder or rotate:** comparisons must use sign-invariant vectors and subspace/projector distance for clustered modes; tests in Task 4 pin this.
3. **A geometry-null generator silently changes wiring cost or disconnects the graph:** nulls must preserve node coordinates, edge count, approximate length histogram, nonnegative weights, and soma connectivity; tests in Task 6 pin this.
4. **Diffusion scoring on disconnected/ill-conditioned graphs produces misleading transfer:** probe functions validate Laplacian shape, use matrix exponentials, conserve total mass for connected undirected graphs, and explicitly report components; tests in Task 5 pin this.
5. **Lesion selection becomes post-hoc:** lesion code accepts only the frozen geometric band from `V0Config`; tests in Task 7 prove it does not inspect edge centrality, eigenvectors, or observed damage magnitude.

---

### Task 1: Package, frozen configuration, and matched developmental histories

**Files:**
- Create: `pyproject.toml`
- Create: `src/devspectral/__init__.py`
- Create: `src/devspectral/config.py`
- Create: `src/devspectral/world.py`
- Create: `tests/test_world.py`

**Interfaces:**
- Produces: `V0Config` frozen dataclass and `V0 = V0Config()`.
- Produces: `HistoryId = Literal["H_A", "H_B"]`.
- Produces: `DevelopmentalWorld(config: V0Config)` with `cue_and_gradient(pos)`, `resource_and_gradient(pos, step, history)`, and `stimulus(pos, step, history)`.
- Produces: `epoch_schedule(history: HistoryId, config: V0Config) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing tests for frozen constants and matched schedules**

Tests assert: grid `64`, total steps `600`, `H_A` alternates, `H_B` blocks, each contains exactly six `L` and six `R`, and the static cue field sampled at fixed points is identical between histories.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `pytest tests/test_world.py -q`

Expected: FAIL because package/config/world do not exist.

- [ ] **Step 3: Implement package metadata, `V0Config`, schedules, and analytic local fields**

Use analytic Gaussian target fields so gradients are exact and local; `resource_and_gradient` activates only the scheduled target while preserving equal epoch counts across histories. `stimulus` is a deterministic sinusoid localized by the active resource Gaussian and must not expose history ID directly.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `pytest tests/test_world.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/devspectral tests/test_world.py
git commit -m "feat: define frozen developmental world"
```

### Task 2: Explicit weighted graph and local growth state

**Files:**
- Create: `src/devspectral/graph.py`
- Create: `src/devspectral/growth.py`
- Create: `tests/test_growth_core.py`

**Interfaces:**
- Consumes: `V0Config`, `DevelopmentalWorld`.
- Produces: `GraphState` with node positions, undirected weighted edges, edge age/traffic/last-use, soma node id, and connectivity helpers.
- Produces: `GrowthTip` dataclass with `tip_id`, `node_id`, `position`, `heading`, `q: np.ndarray(shape=(2,))`, `resource`, `age`, `productive_age`, `parent_tip_id | None`, `alive`.
- Produces: `DevelopmentSimulator(config, history, seed, control="full")` with `step()`, `run(steps=None)`, `snapshot()`.

- [ ] **Step 1: Write failing tests for resonator update, resource update, local-only movement, and graph symmetry**

Tests assert deterministic two-state rotation update, resource clipping to `[0, 2]`, movement step length bound, undirected edge symmetry, nonnegative weights, and no graph/spectrum object is passed into tip direction selection.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `pytest tests/test_growth_core.py -q`

Expected: FAIL with missing graph/growth modules.

- [ ] **Step 3: Implement `GraphState`, `GrowthTip`, and one-step local dynamics**

Direction combines normalized local cue gradient, local field gradient placeholder (zero until Task 3), local resource gradient, curvature penalty, and seeded exploration. Keep all random draws on one `np.random.Generator` owned by the simulator.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `pytest tests/test_growth_core.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/devspectral/graph.py src/devspectral/growth.py tests/test_growth_core.py
git commit -m "feat: add local resonant growth core"
```

### Task 3: Stigmergic field, branching, reinforcement, and pruning

**Files:**
- Create: `src/devspectral/field.py`
- Modify: `src/devspectral/growth.py`
- Modify: `src/devspectral/graph.py`
- Create: `tests/test_structure.py`

**Interfaces:**
- Produces: `StructuralField(config)` with `sample_and_gradient(pos)`, `decay()`, `deposit_segment(p0, p1, amount)`.
- Extends: `DevelopmentSimulator.step()` to deposit useful structure, reinforce traversed edges, branch locally, merge locally, decay/prune edges, and keep active tips soma-connected.
- Produces: `validate_invariants() -> list[str]` returning an empty list for a valid state.

- [ ] **Step 1: Write failing tests for deposition/decay, branching cap, local merge, pruning, and soma-connectivity invariants**

Include the Review Focus regression: force an edge below prune threshold while an active descendant tip depends on it; pruning must either retain the required path or retire/re-home the dependent tip atomically—never leave an orphaned active tip.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `pytest tests/test_structure.py -q`

Expected: FAIL because structural field/slow mutation is absent.

- [ ] **Step 3: Implement slow structural dynamics**

Use bilinear field sampling and small Gaussian segment deposition; bound field attraction before combining directions. Branch only from local resource/productivity criteria. A merge may connect nodes only within `merge_radius`; it must not inspect global shortest paths or spectral metrics.

- [ ] **Step 4: Run Tasks 1-3 tests**

Run: `pytest tests/test_world.py tests/test_growth_core.py tests/test_structure.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/devspectral/field.py src/devspectral/growth.py src/devspectral/graph.py tests/test_structure.py
git commit -m "feat: grow and prune stigmergic anatomy"
```

### Task 4: Laplacian, eigenmodes, and robust spectral comparison

**Files:**
- Create: `src/devspectral/spectral.py`
- Create: `tests/test_spectral.py`

**Interfaces:**
- Consumes: `GraphState`.
- Produces: `weighted_adjacency(graph) -> np.ndarray`.
- Produces: `laplacian(graph) -> np.ndarray`.
- Produces: `low_modes(graph, k=6) -> Spectrum` with sorted eigenvalues/eigenvectors and connected-component count.
- Produces: `mode_alignment(a, b)`, `subspace_distance(U, V)`, `spectral_signature(spectrum)`.

- [ ] **Step 1: Write failing tests on hand-built path, cycle, and two-lobe graphs**

Assert Laplacian symmetry/PSD, one zero eigenvalue for connected graphs, `lambda2 > 0`, sign-invariant mode alignment, and correct ordering.

- [ ] **Step 2: Add the near-degenerate-mode Review Focus regression**

Construct a graph with a repeated/near-repeated eigenspace, rotate its basis internally, and assert projector/subspace distance is ~0 although naive column-wise eigenvector comparison is not.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `pytest tests/test_spectral.py -q`

Expected: FAIL because spectral module is absent.

- [ ] **Step 4: Implement dense NumPy/SciPy spectral utilities**

Use `scipy.linalg.eigh`. Compare clustered modes through orthogonal projectors/principal angles when adjacent eigenvalue gap is `< 1e-7 * max(1, |lambda|)`.

- [ ] **Step 5: Run focused tests and confirm GREEN; commit**

Run: `pytest tests/test_spectral.py -q`

Expected: PASS.

```bash
git add src/devspectral/spectral.py tests/test_spectral.py
git commit -m "feat: add robust graph spectral analysis"
```

### Task 5: Matched post-development diffusion probes

**Files:**
- Create: `src/devspectral/probe.py`
- Create: `tests/test_probe.py`

**Interfaces:**
- Consumes: `GraphState`, `Spectrum`, `V0Config`.
- Produces: `ProbeSpec(source_center, source_radius, readout_center, readout_radius)`.
- Produces: `diffusion_trajectory(graph, probe, times, beta) -> ProbeResult` using `scipy.linalg.expm(-beta * L * t)`.
- Produces: `probe_distance(a, b) -> float` and transfer/arrival summaries.

- [ ] **Step 1: Write failing tests for identical-graph reproducibility and mass conservation**

Use small hand-built connected graphs. Assert same graph/probe gives identical result and total mass stays constant within numerical tolerance.

- [ ] **Step 2: Add disconnected/invalid-graph Review Focus tests**

Disconnected graph must return component metadata and finite trajectories rather than silently reporting cross-component arrival; malformed/asymmetric/negative-weight graphs must raise `ValueError` before propagation.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `pytest tests/test_probe.py -q`

Expected: FAIL because probe module is absent.

- [ ] **Step 4: Implement exact matrix-exponential propagation and summaries**

Use one frozen probe definition: inject around left target and read around right target. Keep probe independent of history and seed.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest tests/test_probe.py -q`

Expected: PASS.

```bash
git add src/devspectral/probe.py tests/test_probe.py
git commit -m "feat: add matched diffusion probe"
```

### Task 6: Destructive developmental and graph controls

**Files:**
- Create: `src/devspectral/controls.py`
- Modify: `src/devspectral/growth.py`
- Create: `tests/test_controls.py`

**Interfaces:**
- Produces developmental control names: `full`, `no_stigmergy`, `no_metabolic_selection`, `no_pruning`.
- Produces graph controls: `geometry_null(graph, seed)`, `weight_shuffle(graph, seed)`, `fixed_lattice(config, node_budget, edge_budget, seed)`.

- [ ] **Step 1: Write failing tests proving each developmental control changes exactly its declared mechanism**

Examples: `no_stigmergy` forces field-gradient contribution to zero but still deposits field; `no_metabolic_selection` removes usefulness dependence from growth resource but leaves costs/geometry; `no_pruning` leaves edge weights decaying but suppresses deletion.

- [ ] **Step 2: Write geometry-null Review Focus tests**

Assert same node coordinates, same edge count, connected-to-soma graph, nonnegative weights, and edge-length histogram counts within one bin of the source graph using fixed 8-bin edges over `[0, sqrt(2)]`.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `pytest tests/test_controls.py -q`

Expected: FAIL because controls do not exist.

- [ ] **Step 4: Implement controls without sharing scientific outputs back into growth**

Geometry null uses seeded length-bin-preserving edge swaps/reconstruction with bounded retry count; if a connected null cannot be produced after 200 attempts, raise a clear `RuntimeError` rather than silently relaxing constraints.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest tests/test_controls.py -q`

Expected: PASS.

```bash
git add src/devspectral/controls.py src/devspectral/growth.py tests/test_controls.py
git commit -m "feat: add developmental and graph null controls"
```

### Task 7: Lesion and local regrowth assay

**Files:**
- Create: `src/devspectral/lesion.py`
- Create: `tests/test_lesion.py`

**Interfaces:**
- Consumes: grown `DevelopmentSimulator` / `GraphState`, `V0Config` lesion band.
- Produces: `lesion_midline_band(graph, config) -> LesionResult`.
- Produces: `resume_regrowth(simulator, steps=200) -> DevelopmentSnapshot`.

- [ ] **Step 1: Write failing test for frozen geometric lesion selection**

Create graphs with high-centrality edges both inside and outside the band. Assert only edges geometrically crossing `x=0.50` with both endpoints `y>=0.55` are removed; lesion result must be unchanged if edge centrality/eigenvector metadata is attached or permuted.

- [ ] **Step 2: Write lesion effect tests on a hand-built two-lobe bridge graph**

Assert lesion lowers `lambda2` and decreases left-to-right probe transfer; exact numeric values need not be hard-coded beyond direction/tolerance.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `pytest tests/test_lesion.py -q`

Expected: FAIL because lesion module is absent.

- [ ] **Step 4: Implement lesion and regrowth plumbing**

Regrowth reuses the original local simulator rules with the frozen balanced alternating schedule and no repair/lesion-location signal. Record whether any new edge crosses the lesion band.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest tests/test_lesion.py -q`

Expected: PASS.

```bash
git add src/devspectral/lesion.py tests/test_lesion.py
git commit -m "feat: add predeclared lesion and local regrowth"
```

### Task 8: Scientific gate runner, blind history readback, and receipts

**Files:**
- Create: `src/devspectral/experiment.py`
- Create: `src/devspectral/receipt.py`
- Create: `scripts/run_v0.py`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Produces: `run_one(history, seed, control="full") -> RunRecord`.
- Produces: `run_gate_suite(seeds=range(8)) -> GateReceipt`.
- Produces: `history_features(record) -> np.ndarray` using anatomy/spectrum only.
- Produces: `nearest_centroid_leave_one_seed_out(records) -> ReadbackResult`.
- Produces JSON-safe `to_dict()` receipts and compact graph serialization (`positions`, edge index pairs, weights).

- [ ] **Step 1: Write failing end-to-end smoke tests on two seeds**

Assert deterministic rerun equality for same history/seed, valid invariants, serializable graph, spectrum, matched probe metrics, and no developmental log fields in readback features.

- [ ] **Step 2: Pin Gate 1 and Gate 2 metrics**

Define `structural_separation_ratio = median(between-history distance) / max(eps, median(within-history distance))` using a composite of normalized low eigenvalues + spatial edge-density summary. Gate 1 reports `pass` if at least 6/8 matched cross-history distances exceed the 75th percentile of pooled within-history distances and ratio > 1.0.

Define `probe_separation_ratio` analogously on flattened probe trajectories. Gate 2 reports `pass` if at least 6/8 matched cross-history probe distances exceed the 75th percentile of within-history distances; rerunning the probe after zeroing all tip fast states must produce the same graph-based trajectory.

- [ ] **Step 3: Pin Gate 4 readback and control weakening metrics**

Readback features: first six nontrivial eigenvalues padded with zeros, node count, edge count, total edge length, mean/SD edge weight, and a fixed `4x4` spatial edge-density histogram. Leave-one-seed-out nearest centroid must reach `>= 0.75` accuracy to mark readback `pass`.

A destructive control is recorded as materially weakening an effect if its separation ratio is at least `25%` lower than full. The suite requires at least one weakening control for the overall v0 scientific status to be `supported`; otherwise report `inconclusive` even if raw history separation is present.

- [ ] **Step 4: Pin Gate 5 lesion/regrowth receipt semantics**

For each seed record pre-lesion, lesion, and regrowth `lambda2`, component count, transfer, and number of new band-crossing edges. Lesion direction is considered observed when `lambda2` and transfer both decrease in at least 6/8 seeds. Regrowth is reported as `partial_recovery`, `no_recovery`, or `not_applicable`; failure is not converted into a failed software test.

- [ ] **Step 5: Run focused tests and confirm RED**

Run: `pytest tests/test_experiment.py -q`

Expected: FAIL because experiment/receipt modules are absent.

- [ ] **Step 6: Implement runner and receipt schema**

Do not add parameter search. Scientific status is computed from the predeclared rules above and stored alongside raw per-seed values.

- [ ] **Step 7: Run all tests and commit**

Run: `pytest -q`

Expected: all tests PASS.

```bash
git add src/devspectral/experiment.py src/devspectral/receipt.py scripts/run_v0.py tests/test_experiment.py
git commit -m "feat: add frozen scientific gate suite"
```

### Task 9: Diagnostic visualizer and project documentation

**Files:**
- Create: `src/devspectral/visualize.py`
- Create: `scripts/visualize_run.py`
- Create: `README.md`
- Create: `tests/test_visualize.py`

**Interfaces:**
- Consumes: serialized `RunRecord` / graph snapshots.
- Produces: `render_snapshot(record, layer, output_path=None)` for `field`, `graph`, `usage`, `mode1`, `mode2`, `mode3`, `probe`, `lesion`.

- [ ] **Step 1: Write failing headless visualizer smoke tests**

With Matplotlib `Agg`, render each supported layer from a tiny synthetic record and assert a figure is produced without mutating the record.

- [ ] **Step 2: Run focused test and confirm RED**

Run: `pytest tests/test_visualize.py -q`

Expected: FAIL because visualizer does not exist.

- [ ] **Step 3: Implement diagnostic renderer and README**

README must state the claim boundary, frozen v0 history schedules, how to run tests/scientific suite/visualizer, and that visual beauty is not a scientific gate.

- [ ] **Step 4: Run full verification and commit**

Run: `pytest -q`

Expected: all tests PASS.

```bash
git add src/devspectral/visualize.py scripts/visualize_run.py README.md tests/test_visualize.py
git commit -m "docs: add developmental spectral neuron visualizer"
```

### Task 10: Freeze and run the first v0 scientific receipt

**Files:**
- Create: `results/v0_receipt.json`
- Create: `docs/V0_STATUS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: exact tested code and frozen `V0` config from Tasks 1-9.
- Produces: one immutable first-run receipt over seeds `0..7` and all declared controls.

- [ ] **Step 1: Run fresh software verification before science**

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 2: Run the frozen suite once**

Run: `python scripts/run_v0.py --output results/v0_receipt.json`

Expected: exits 0, writes all per-seed raw records and gate summaries; scientific gates may be supported, inconclusive, or negative.

- [ ] **Step 3: Validate receipt without changing parameters**

Run: `python scripts/run_v0.py --check-receipt results/v0_receipt.json`

Expected: schema/invariant check PASS and frozen config hash matches `V0`.

- [ ] **Step 4: Write `docs/V0_STATUS.md` from the receipt**

Report what held, what failed, controls, lesion/regrowth result, and biological claim boundary. Do not rerun after seeing results unless fixing a documented software/engineering defect; any such fix invalidates and replaces the entire scientific receipt.

- [ ] **Step 5: Final verification**

Run: `python -m compileall -q src scripts && pytest -q`

Expected: exit 0 and all tests PASS.

- [ ] **Step 6: Commit frozen receipt and status**

```bash
git add results/v0_receipt.json docs/V0_STATUS.md README.md
git commit -m "results: freeze developmental spectral neuron v0"
```

## Execution notes

- Implementation should occur on an isolated feature branch/worktree created from the approved-plan commit.
- Use RED -> GREEN TDD for every task above.
- Do not tune scientific constants after Task 8 except under the explicit Gate-0 engineering-fix rule; if changed, restart all multi-seed receipts.
- Preserve negative scientific outcomes exactly. A software-complete branch may legitimately conclude that the developmental mechanism is unsupported under v0.
- Before handoff or merge, run a whole-branch review specifically for hidden global information in local growth, control fairness, history leakage into readback/probes, post-hoc lesion selection, and numerical spectral/diffusion errors.
