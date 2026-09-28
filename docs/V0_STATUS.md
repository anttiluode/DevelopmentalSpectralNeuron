# Developmental Spectral Neuron v0 — Final Frozen Receipt

Date: 2026-09-28

Receipt: `results/v0_receipt.json`

Overall frozen status: **NEGATIVE**.

The software substrate works, but the v0 scientific claim does not survive the final reviewed implementation. With local cue magnitude preserved, branch resource conserved, and branch merging repaired, temporal ordering of the same left/right developmental exposures does **not** produce a reproducible history-specific anatomy or later operator.

This is the result to carry forward.

## Why this differs from the provisional receipt

An earlier provisional run appeared to show very strong history separation. Whole-branch review found three Important implementation defects before handoff:

1. **Vanishing gradients were normalized to unit direction.** Tiny far-field resource gradients therefore acted like full-strength directional oracles, letting the current schedule steer growth globally rather than locally.
2. **Branching created resource.** Parent and daughter each received 75% of the parent's pre-branch resource, increasing total resource by 50% at every branch event.
3. **The parent node masked local merges.** After a new growth node was created, the nearest-node search usually rediscovered its already-connected parent and therefore failed to connect to another nearby branch.

Each defect received a RED→GREEN regression test. The provisional receipt was deleted, the same frozen v0 constants were retained, and the complete 8-seed suite was restarted from seed 0.

The numbers below are from that corrected run only.

## Gate 0 — deterministic valid development: PASS

- exact rerun deterministic: yes
- valid full runs: `16 / 16`
- all full graphs were connected and invariant-clean

So the negative scientific result is not a software-crash or invalid-graph result.

## Gate 1 — developmental history leaves a structural trace: FAIL

`H_A` and `H_B` each receive exactly six left and six right resource epochs. Only temporal order differs.

- matched cross-history distances above the within-history 75th percentile: `0 / 8`
- required: `6 / 8`
- structural separation ratio: `0.0002135`
- median matched cross-history distance: `0.0001193`
- median within-history distance: `0.5587052`

This is the opposite of the desired signature. For a fixed seed, changing `H_A` to `H_B` barely changes the final structural descriptor, while changing the seed within one history changes it enormously.

In v0, **seed-to-seed developmental variation dominates history order**.

## Gate 2 — same later probe, different developed operator: FAIL

The later diffusion probe is identical across histories and represented in a fixed 4×4 spatial mass basis across frozen times, so different graph sizes do not require node-by-node matching.

- matched cross-history probe distances above within-history 75th percentile: `0 / 8`
- required: `6 / 8`
- probe separation ratio: `1.265e-6`
- median matched cross-history probe distance: `6.216e-7`
- median within-history probe distance: `0.4912166`

Thus the same-seed `H_A` and `H_B` graphs produce effectively the same later probe behavior compared with ordinary seed variability.

The core v0 claim

```text
history A != history B
    -> different developed operator
    -> different response to the same later probe
```

is **not supported**.

## Gate 4 — blind history readback from final anatomy: FAIL

A leave-one-seed-out nearest-centroid reader receives only final graph/spectral features.

- accuracy: `8 / 16 = 0.50`

That is chance for two histories. Final anatomy does not retain a reliably decodable record of whether exposures were alternating or blocked.

## Spectral alignment against matched geometry nulls: FAIL

- developed graph alignment greater than geometry-matched null: `8 / 16`
- required: `12 / 16`
- median developed alignment: `0.07513`
- median geometry-null alignment: `0.09661`

The grown low-order subspace is therefore not more task-aligned than the matched null under the frozen criterion. Its median alignment is actually lower.

## Destructive controls

Because the full model has no primary history effect, control-arm "weakening" cannot rescue the claim. A smaller ratio than an already near-zero full ratio is not positive evidence.

| arm | structural separation ratio | probe separation ratio | blind readback |
|---|---:|---:|---:|
| full | 0.000213 | 0.00000127 | 0.500 |
| no stigmergy | 0.298014 | 0.553180 | 0.438 |
| no metabolic selection | 0.346850 | 0.012489 | 0.500 |
| no pruning | 0.000213 | 0.00000127 | 0.500 |
| geometry null | 0.000199 | 0.0000349 | 0.375 |
| weight shuffle | 0.000407 | 0.0000117 | 0.500 |
| fixed lattice | 0.000000 | 0.000000 | 0.500 |

Notable negatives:

- `no_pruning` is again numerically identical to full on the separation metrics, so current pruning still does not earn its complexity.
- Removing metabolic selection or stigmergy increases some raw ratios relative to full, but neither arm meets the predeclared separation/readback gates. That is not evidence that the ablations are better mechanisms; it shows how dominant seed variability is in this assay.

## Gate 5 — lesion and regrowth: NOT APPLICABLE

Frozen lesion rule:

```text
remove edges crossing x = 0.50
when both endpoints have y >= 0.55
```

Across all eight `H_A` runs:

- edges removed: `0, 0, 0, 0, 0, 0, 0, 0`
- applicable lesion runs: `0 / 8`

So lesion/regrowth was never engaged. The band is not moved after inspecting the result.

This does not change the overall conclusion because the primary history/operator gates already fail.

## What v0 *does* establish

The implementation demonstrates an auditable engineering substrate:

- deterministic local growth under fixed seeds;
- fast resonant tip state, medium resource, and slow structural state as distinct variables;
- an external stigmergic field separate from the explicit computational graph;
- local branching and local branch merging;
- resource-conserving branch events;
- pruning invariants that do not orphan active tips;
- graph Laplacian/eigenmode analysis robust to sign and degenerate subspaces;
- fixed-coordinate diffusion probes;
- geometry, weight, developmental and lattice controls;
- predeclared lesion plumbing and local regrowth machinery;
- machine-readable frozen receipts.

Those are tools, not evidence for the proposed developmental computation.

## Scientific conclusion

The corrected v0 says something useful and restrictive:

> **Simply combining local resonators, a resource budget, stigmergic growth, branching and graph eigenmodes is not enough to make temporal developmental order become anatomy.**

With the obvious global-gradient shortcut removed, the system mostly grows according to its seed-specific trajectory. The current resource signal does not write the alternating-vs-blocked history strongly enough into structure.

That means the next step should **not** be to add apical tuft, SST, rhythmic windows, or more neuronal decoration. The substrate has not earned those layers.

A new version would need a new, predeclared developmental mechanism that gives temporal success a genuinely local structural consequence—for example, delayed local credit, resource transport along already-active branches, or branch-specific reinforcement tied to propagated activity—then start a new frozen receipt from scratch.

## Biological claim boundary

Nothing here establishes that:

- the resource variable corresponds to ATP, dopamine, calcium, transmitter, or neurotrophin;
- the tips reproduce biological growth cones or Sperry chemoaffinity;
- the graph eigenmodes are thoughts or memories;
- diffusion models fast cortical computation;
- the non-engaging lesion assay models callosal injury or developmental compensation.

The motivating analogy remains only this: local development constructs anatomy, and anatomy constrains later dynamics. v0 did not demonstrate that the chosen temporal history controls that construction.
