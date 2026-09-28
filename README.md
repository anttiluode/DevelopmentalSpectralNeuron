# Developmental Spectral Neuron

A small artificial-development experiment in which **local nonlinear growth rules manufacture the graph that later performs computation**.

The v0 causal chain is:

```text
environmental history
  -> local resonant activity + resource budget
  -> stigmergic growth / reinforcement / pruning
  -> explicit sparse weighted graph
  -> graph Laplacian and low-order modes
  -> matched diffusion probe
  -> lesion / local regrowth assay
```

The main claim is deliberately narrow: two different developmental histories may grow two different structural operators, so the same later probe can propagate differently even after fast internal states are irrelevant. The project does **not** claim that its resource variable is ATP/dopamine/neurotrophin, that its growth tips reproduce biological growth cones, that its eigenmodes are thoughts, or that graph diffusion is a full cortical-neuron model.

## Frozen v0 histories

Both conditions receive exactly six left-target and six right-target resource epochs over 600 steps. Only order differs:

- `H_A`: `L,R,L,R,L,R,L,R,L,R,L,R`
- `H_B`: `L,L,L,L,L,L,R,R,R,R,R,R`

The v0 parameter set is fixed in `devspectral.config.V0`. Scientific runs do not tune separately by history or lesion outcome.

## What is measured

The suite records deterministic-growth validity, structural/spectral separation, matched post-development probe separation, blind history readback from final anatomy, low-mode alignment against geometry-matched nulls, destructive controls, and a predeclared midline lesion/regrowth assay.

A negative or inconclusive receipt is a valid result. Visual beauty is not a gate.

## Install / test

```bash
python -m pip install -e .
pytest -q
```

## Run the frozen v0 receipt

```bash
python scripts/run_v0.py --output results/v0_receipt.json
python scripts/run_v0.py --check-receipt results/v0_receipt.json
```

## Visualize one developed run

```bash
python scripts/visualize_run.py --history H_A --seed 0 --layer graph
python scripts/visualize_run.py --history H_A --seed 0 --layer mode1 --output mode1.png
```

Supported diagnostic layers: `field`, `graph`, `usage`, `mode1`, `mode2`, `mode3`, `probe`, `lesion`.

## Interpretation boundary

The motivating mathematics is spectral graph theory: once development has produced a graph, its Laplacian gives a principled description of large-scale transmission geometry. v0 uses low-frequency diffusion only. Fast oscillatory/windowed propagation, apical context, branch-local inhibition, multi-neuron networks, and interacting observer systems are later stages only if this substrate earns them.

## First frozen v0 result

The final reviewed 8-seed receipt is in `results/v0_receipt.json`; the detailed interpretation is in `docs/V0_STATUS.md`.

**Status: negative.** Gate 0 passes, but the core science does not: changing alternating vs blocked temporal order produces less structural and probe variation than changing the random seed, blind history readback is exactly chance (`0.50`), and developed low-order modes do not beat geometry-matched nulls on the frozen alignment criterion. The predeclared lesion band also intersects no edges, so lesion/regrowth is not applicable.

A provisional pre-review receipt had looked strongly positive; it was invalidated after code review found a far-field gradient normalization shortcut, non-conserving branch resource, and a masked local-merge bug. All three received RED→GREEN regression tests before the final receipt was rerun with the same constants.

The useful conclusion is therefore restrictive: **the current local resonator/resource/stigmergy rules do not make temporal developmental order become anatomy.** The next version needs a new local credit mechanism before adding more neuron-inspired layers.
