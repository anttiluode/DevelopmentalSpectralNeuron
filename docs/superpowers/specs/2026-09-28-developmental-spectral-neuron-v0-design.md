# Developmental Spectral Neuron v0 — Design

Date: 2026-09-28

## Purpose

Build a small artificial system in which **local nonlinear developmental rules construct the structure that later performs computation**.

The project starts from a different place than a conventional neural network. We do not choose a fixed adjacency matrix and then train weights on it. We let local agents grow, reinforce, weaken, and prune connections under a resource budget. That developmental history produces a graph. The graph then has its own natural spatial modes, obtained from the graph Laplacian, and later signals propagate through the structure that development created.

The central causal chain is:

```text
environmental history
    -> local metabolic growth / reinforcement / pruning
    -> developed structural graph A_H
    -> graph Laplacian L_H = D_H - A_H
    -> eigenmodes {u_k, lambda_k}_H
    -> response to a later probe
```

The core requirement is therefore:

> Different developmental histories must be able to construct different operators, so the same later probe can propagate differently even when the present input is identical.

Formally:

\[
H_A \ne H_B
\Rightarrow
L_{H_A} \ne L_{H_B}
\Rightarrow
O_{H_A}(x) \ne O_{H_B}(x)
\]

for at least some matched probe \(x\).

This is the first project in the line where history is intended to become **anatomy**, not merely another hidden vector.

## Motivation

Three threads meet here.

### 1. Local nonlinear ecology

The motivating intuition is that computation can emerge from local units with:

- limited resources,
- activity-dependent cost,
- local reinforcement,
- competition,
- cooperation,
- persistence when useful,
- weakening or removal when persistently unproductive.

The important design idea is not that "death is ReLU" or that a software resource variable literally models ATP. It is that fast activity, medium resource balance, and slow structural persistence are different causal processes.

### 2. Stigmergic growth

Local agents can alter the environment that guides later agents. A successful path can leave persistent structural material; repeatedly used paths strengthen; unused paths fade. The developing system therefore has an externalized memory of where useful activity has occurred.

This gives the system a visible developmental substrate rather than an invisible learned weight matrix.

### 3. Structural eigenmodes

Wang, Owen, Mukherjee & Raj (2017), *Brain network eigenmodes provide a robust and compact representation of the structural connectome in health and disease*, analyze low-frequency macroscopic diffusion on the structural connectome with

\[
\dot{x} = -\beta Lx,
\]

where \(L\) is the graph Laplacian. Their work motivates using the low-order Laplacian eigenmodes as a compact description of large-scale structure and spread.

We will use that mathematics carefully. The paper explicitly treats a linear, low-frequency macroscopic diffusion regime and notes that fast oscillatory behavior and axonal conduction delays require richer models. Therefore v0 does **not** claim that Laplacian diffusion is a full neuron model or that connectome eigenmodes explain local cortical computation.

The paper is used here for one narrower idea:

> once local development has produced a graph, the graph's spectrum provides a principled description of its large-scale transmission geometry.

## Scientific question

Can simple local growth rules, driven by a resource budget and stigmergic reinforcement, create a graph whose large-scale modes reflect the developmental environment and causally alter later signal propagation?

A successful v0 should show all of the following:

1. local rules can grow a nontrivial sparse structure without a global wiring plan;
2. different histories produce measurably different structures;
3. those structural differences produce different responses to the same later probe;
4. at least part of the developmental history can be read back from the final structure alone;
5. the effect disappears or weakens under destructive controls such as randomized edges or disabled reinforcement;
6. lesioning an important bridge changes large-scale transmission, and renewed local development can sometimes restore a functional bridge without being told where to place it.

No claim of novelty or biological fidelity is required for v0. The goal is a falsifiable computational mechanism.

## Scope

### In v0

- A deterministic 2-D synthetic developmental world.
- One soma/root region and multiple local growth tips.
- Fast local activity per tip.
- Medium resource balance per tip.
- Slow deposited structural material.
- Local chemotactic / sensory guidance.
- Stigmergic attraction to previously useful structure.
- Maintenance cost and activity cost.
- Reinforcement of productive paths.
- Weakening / pruning of persistently unused paths.
- Explicit graph extraction from the grown structure.
- Graph Laplacian and low-order eigenmodes.
- Diffusion probes on the developed graph.
- Same-present/different-history test.
- Geometry/topology destructive controls.
- Bridge lesion experiment.
- Regrowth experiment after lesion.
- Blind history readback from final structure.
- Deterministic seeds and frozen result receipts.
- A simple visualizer after the headless mechanism is verified.

### Explicitly not in v0

- No webcam requirement for the scientific gate.
- No fly bodies or population reproduction requirement.
- No supervised clicker / manual teaching signal.
- No apical tuft, SST/Martinotti, PV/basket, or chandelier/AIS modules yet.
- No multi-neuron network.
- No two-observer resonance experiment.
- No claim that the local growth law reproduces Sperry chemoaffinity, axonal pathfinding, Hebbian structural plasticity, or cortical development.
- No claim that low graph eigenmodes are themselves thoughts, memories, or cognitive states.
- No claim that linear diffusion is valid for fast local neural dynamics.

Those are later layers only if this substrate earns them.

## Architecture

```text
DEVELOPMENTAL WORLD
  spatial cue field C(x,t)
  temporal stimulus S(x,t)
  obstacles / boundaries
          |
          v
LOCAL GROWTH TIPS
  position p_i
  heading h_i
  fast activity z_i
  resource E_i
          |
          +---------- useful local activity ----------+
          |                                           |
          v                                           v
      movement                                  resource change
          |                                           |
          v                                           |
STIGMERGIC STRUCTURAL FIELD F(x,t) <------------------+
          |
     reinforce / decay
          |
          v
DEVELOPED ARBOR / GRAPH A_H
          |
          v
LAPLACIAN L_H = D_H - A_H
          |
          v
LOW-ORDER MODES {u_k, lambda_k}
          |
          v
PROBE PROPAGATION / LESION / HISTORY READBACK
```

## 1. Developmental world

The first world must be simpler than a webcam scene so the causal effects are identifiable.

Use a bounded 2-D domain with a soma/root region and controllable cue sources. The world exposes two classes of information:

### Spatial guidance field

A smooth scalar field \(C(x,t)\) that growth tips can sample locally. It may contain attraction, repulsion, barriers, and target zones.

### Temporal local stimulus

A local time-varying signal \(S(x,t)\). Different regions may contain different temporal patterns or resource opportunities.

The first two developmental histories should have matched gross exposure but different organization:

- `H_A`: useful activity is distributed so a left-right bridge is repeatedly rewarded;
- `H_B`: useful activity is distributed so two lobes are reinforced more independently.

A later probe is identical for both developed structures.

The histories must be generated from fixed seeds and recorded so all arms receive the same intended schedule.

## 2. Growth tips

Use a small population of local growth tips. Each tip has:

- position \(p_i=(x_i,y_i)\),
- heading \(h_i\),
- fast internal activity \(z_i\),
- medium resource \(E_i\),
- parent branch / current edge identity,
- age and stagnation counters.

Tips do not know the global target graph, Laplacian, eigenmodes, or future lesion test.

A tip may only use local information from its neighborhood.

## 3. Fast local activity

The tip receives the local temporal stimulus and carries a simple resonant or leaky state.

The initial v0 choice is a real two-state resonator:

\[
q_i(t+1)=r_i R(\omega_i)q_i(t)+(1-r_i)b_iS(p_i,t),
\]

where \(R(\omega_i)\) is a 2-D rotation matrix.

Local activity magnitude is

\[
a_i(t)=\|q_i(t)\|.
\]

The intrinsic \(\omega_i\) and \(r_i\) are fixed at birth in v0. They are not optimized by gradient descent.

This fast state influences resource harvesting but is not itself long-term memory.

## 4. Resource dynamics

Each tip pays maintenance and activity costs:

\[
E_i(t+1)=E_i(t)-c_0-c_1 a_i(t)+G_i(t),
\]

where \(G_i(t)\) is locally harvested resource.

Initial v0 harvesting is based on coincidence between local activity and the local environmental opportunity field:

\[
G_i(t)=\eta\,a_i(t)\,R_{env}(p_i,t).
\]

The exact variable is deliberately called `resource`, not ATP, dopamine, transmitter, calcium, or neurotrophin.

Resource determines whether a tip can continue extending and how strongly it deposits structural material.

## 5. Local movement / growth law

A tip chooses a small displacement using only local terms:

\[
\Delta p_i
\propto
w_c \nabla C(p_i,t)
+ w_f \nabla F(p_i,t)
+ w_r d_i(t)
- w_k \kappa_i(t)
+ \xi_i(t).
\]

Terms:

- `cue`: chemotactic / sensory gradient,
- `stigmergy`: attraction to useful existing structure,
- `resource direction`: bias toward locally productive directions,
- `curvature cost`: discourages unrealistic jitter,
- `exploration`: small seeded noise.

The stigmergic term must be bounded so the first trail cannot simply trap every later tip.

Movement stops or slows when resource is depleted.

## 6. Structural deposition

As a tip moves from \(p_i(t)\) to \(p_i(t+1)\), it deposits material along that segment.

Let \(F(x,t)\) be a 2-D structural support field:

\[
F(x,t+1)=\rho F(x,t)+\sum_i d_i(t)K(x; p_i(t),p_i(t+1)).
\]

with \(0<\rho<1\).

Deposit strength is a bounded increasing function of resource and useful activity, for example:

\[
d_i(t)=d_{min}+d_{gain}\,\sigma(E_i-E_*)\,\sigma(a_i-a_*).
\]

The field is a developmental trace, not the final graph itself.

Repeated useful traversal thickens a path. Unused material decays.

## 7. Reinforcement and pruning

The explicit graph keeps edge-level usage statistics.

Each edge stores:

- geometric length,
- structural weight,
- recent traffic / activity,
- age,
- last-use time.

When productive activity traverses an edge, its structural weight increases slowly.

When an edge remains unused, its weight decays.

Edges below a minimum weight for a sustained grace period are pruned.

This gives slow structural memory:

```text
fast:    resonant local activity
medium:  tip resource
slow:    edge strength / survival
```

The three clocks have different jobs and must not be implemented as interchangeable generic EMAs.

## 8. Branching

A tip may create a daughter tip only when all of the following hold:

- local resource is above a threshold,
- the parent has been productive for a minimum time,
- branch spacing rules permit it,
- the global hard cap on tips has not been reached.

The daughter heading is a bounded perturbation of the parent heading.

Branching must not inspect global graph metrics.

The purpose is to let useful regions acquire more exploratory capacity without giving the system a central planner.

## 9. Graph extraction

The simulator should maintain the graph directly while growth occurs rather than attempting to infer all topology later from image skeletonization.

Nodes are created at:

- soma/root,
- branch points,
- tip positions at configurable spatial intervals,
- merges where two growing branches come sufficiently close and a local merge rule allows connection.

Edges carry the slow structural weights described above.

The rendered field \(F\) and the explicit graph must remain separate objects:

- `F` is the stigmergic local guidance / visualization substrate;
- `A` is the computational structural graph.

This avoids making scientific results depend on arbitrary image-thinning parameters.

## 10. Graph Laplacian and spectrum

For the developed weighted undirected graph, construct

\[
L=D-A.
\]

Compute the smallest few eigenpairs:

\[
Lu_k=\lambda_k u_k.
\]

Ignore the trivial constant mode when reporting nontrivial structure.

Primary spectral diagnostics:

- \(\lambda_2\) algebraic connectivity,
- first 3-6 nontrivial eigenvectors,
- eigengaps,
- mode stability across seeds within one developmental condition,
- mode distance across developmental conditions.

Eigenvector sign is arbitrary and comparisons must be sign-invariant. Near-degenerate modes must be compared as subspaces rather than by naive index when necessary.

No claim should be based only on aesthetically interesting eigenmode pictures.

## 11. Probe propagation

After development is frozen, inject the same probe into both `H_A` and `H_B` graphs.

Initial v0 propagation uses the same low-frequency diffusion family that motivates the spectral analysis:

\[
\dot{x}=-\beta Lx.
\]

Discrete implementation may use a stable matrix-exponential solution or a sufficiently small explicit step verified against it.

Measure:

- arrival time / mass transfer between designated regions,
- response trajectory at readout nodes,
- projection coefficients \(c_k(t)=u_k^T x(t)\),
- reconstruction from the first \(k\) modes.

The important causal test is not merely that the graphs look different. The same probe must produce detectably different propagation because the developed operator differs.

## 12. Gate 0 — deterministic local growth

Before scientific claims:

- identical seed + history gives identical graph and spectrum;
- changing only seed creates bounded variation;
- no NaNs, negative structural weights, orphaned active tips, or invalid graph indices;
- edge creation and pruning conserve stated invariants;
- graph remains connected to the soma for all active structural components unless a test intentionally allows disconnected fragments.

## 13. Gate 1 — developmental specificity

Train/develop multiple seeds under `H_A` and `H_B`.

Pass condition:

- within-condition graph/spectral distance is smaller than between-condition distance for a clear majority of matched seed comparisons;
- the result survives a geometry-aware comparison rather than depending only on total edge count.

This establishes that developmental history leaves a structural trace.

## 14. Gate 2 — same-present / different-history operator test

Freeze two developed graphs from different histories.

Set the instantaneous probe input to be identical.

Pass condition:

- probe responses differ beyond within-condition seed variability;
- resetting fast tip states before the probe does not remove the difference because the cause is the slow developed graph;
- swapping only the graph while keeping the probe and propagation rule fixed swaps the response.

This is the core project claim.

## 15. Gate 3 — destructive controls

At minimum compare against:

### `no_stigmergy`

Growth cannot sense \(F\), but all other resource rules remain.

### `no_metabolic_selection`

Tips receive enough resource to grow regardless of local usefulness. Geometry and noise remain.

### `no_pruning`

All deposited edges persist.

### `geometry_null`

Preserve node coordinates, approximate edge-length distribution, and edge count while randomizing allowable connectivity.

### `weight_shuffle`

Preserve topology but shuffle structural weights.

### `fixed_lattice`

Use a nondeveloped geometric graph with a matched node/edge budget.

A useful result is one where at least one specific developmental ingredient can be shown to cause the history-dependent structural effect rather than all arms behaving equivalently.

## 16. Gate 4 — blind history readback

The reader receives only the final developed structure, never the developmental stimulus log.

Features may include:

- low-order eigenvalues,
- simple graph statistics,
- spatial edge-density summaries,
- low-order eigenmode projections on fixed spatial basis functions.

Use a deliberately weak classifier or nearest-centroid rule with held-out seeds.

Pass condition:

- developmental history can be decoded above chance on held-out seeds;
- decoding falls toward chance under at least one destructive control.

The purpose is to demonstrate that anatomy contains recoverable history, not to maximize classifier performance.

## 17. Gate 5 — lesion and regrowth

Construct a developmental condition that tends to create two lobes joined by one or a few bridges.

After development:

1. identify a bridge region by a predeclared geometric cut, not by choosing the most dramatic edge after seeing results;
2. remove edges crossing that region;
3. recompute \(\lambda_2\), low-order modes, and probe transfer;
4. resume only the same local growth rules;
5. test whether a new connection can form across the damaged region;
6. measure whether algebraic connectivity and probe transfer recover.

Expected qualitative signature of a successful lesion:

- reduced \(\lambda_2\),
- slower inter-lobe transfer,
- stronger localization / splitting of low-order modes.

A successful regrowth result requires partial recovery relative to the lesioned state. It does **not** require restoration of the exact original edge or exact original eigenvectors.

Failure to regrow is an acceptable negative result and must be preserved.

## 18. Spectral compactness test

The low-order eigenbasis is mathematically privileged for diffusion, so simply showing that it reconstructs diffusion well would be close to tautological.

Instead, the test should ask a nontrivial question:

> Does the *developed* low-order subspace align with the spatial task/history better than matched null graphs do?

Possible metrics:

- alignment of low-order modes with predeclared left/right or source/target spatial functions,
- variance of probe responses captured by the first \(k\) modes relative to geometry-matched null graphs,
- stability of the low-order subspace across seeds within the same history.

Do not claim discovery of meaningful modes from compactness alone.

## 19. Visualization

The first UI should make the causal hierarchy visible.

Recommended two-pane display:

### Left — developmental world

- cue / resource field,
- growth tips,
- active branches,
- resource level encoded by size or intensity,
- current local activity.

### Right — anatomy and modes

Selectable layers:

1. stigmergic field \(F\),
2. explicit weighted graph,
3. pruning / recent-use overlay,
4. first nontrivial eigenmode,
5. second/third nontrivial eigenmodes,
6. probe propagation heat map,
7. lesion/regrowth state.

The visualizer is diagnostic, not part of the scientific gate.

A later live-camera version may project development over webcam geometry, but v0 must first work in the synthetic world.

## 20. Data and receipts

Every scientific run writes a machine-readable receipt containing:

- git commit SHA if available,
- seed,
- complete parameter set,
- developmental history ID,
- graph node/edge counts,
- edge-weight distribution summary,
- connected-component count,
- low-order eigenvalues,
- mode-alignment metrics,
- probe-transfer metrics,
- history-readback result,
- control-arm name,
- lesion and regrowth metrics when applicable.

Raw graphs should be serializable in a compact NumPy/JSON-friendly format so a result can be reanalyzed without rerunning development.

## 21. Parameter discipline

The first parameter set is fixed before multi-seed scientific runs.

Do not tune parameters separately for `H_A` and `H_B`.

Do not tune on the lesion result.

If the first parameter set fails to produce any viable connected development, debugging changes are allowed but must be documented as engineering fixes, and scientific gates restart from a fresh frozen parameter set.

Small seed counts are acceptable for v0 as long as claims are correspondingly modest and raw results are preserved.

## 22. Biological claim boundary

Allowed language:

- inspired by local growth, resource competition, stigmergy, structural plasticity, and graph diffusion;
- neuron-like developmental machine;
- developmental graph whose spectrum constrains later propagation;
- computational analogy to anatomy shaping dynamics.

Not allowed without separate evidence:

- "this is how dendrites grow";
- "the resource variable is ATP/dopamine/neurotrophin";
- "the modes are cortical thoughts";
- "the growth tips model a particular biological growth-cone pathway";
- "the model explains the connectome";
- "lesion regrowth models AgCC compensation".

The 2017 Wang et al. connectome paper is macroscopic and low-frequency. Its role here is mathematical motivation for the graph-spectrum readout, not validation of our local developmental rules.

## 23. Success criteria for v0

The project earns a next stage if:

1. development produces stable sparse connected structures across several seeds;
2. `H_A` and `H_B` leave distinguishable structural/spectral traces;
3. the same post-development probe produces history-dependent responses because the graph differs;
4. history can be decoded from final anatomy above chance on held-out seeds;
5. at least one destructive control materially weakens these effects;
6. a predeclared bridge lesion alters algebraic connectivity and transfer in the expected direction;
7. local regrowth either restores some function or fails honestly under a frozen rule set.

The project does **not** need to beat a neural-network benchmark to pass v0.

## 24. Next stages, only if earned

If v0 survives:

### v1 — temporal computation on the grown arbor

Replace pure diffusion with richer local resonant/windowed propagation while retaining the same developed structure.

### v2 — apical context

Add a second contextual input stream that modulates susceptibility of distal branches without directly providing the basal content.

### v3 — branch-local inhibition

Add SST/Martinotti-like route suppression and rhythmic gating only after there are actual routes for those mechanisms to control.

### v4 — multiple developmental neurons

Allow several grown units to connect and ask whether local development produces useful inter-unit topology.

### v5 — interacting observers

Place two history-shaped systems in a shared world and define resonance operationally through reciprocal predictive/structural alignment rather than raw hidden-state similarity.

## Summary

Developmental Spectral Neuron v0 is deliberately built around one idea:

```text
local nonlinear life-like rules
        -> grow anatomy
        -> anatomy defines a graph
        -> graph defines large-scale modes
        -> modes constrain future propagation
```

The ambition is not to draw a brain-shaped network. It is to test whether **development can manufacture the operator** that later performs computation.
