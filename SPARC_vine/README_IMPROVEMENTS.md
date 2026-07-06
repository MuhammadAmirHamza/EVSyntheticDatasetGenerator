# SPARC_vine — improved variant (dependence upgrade + model selection)

Isolated copy of the SMC framework so improvements can be tested without touching
the canonical `smc_core.jl` / `run_smc.jl`. Writes to `../artifacts/<name>_vine/`.

Run from this folder:
```bash
julia run_smc_vine.jl combined          # -> ../artifacts/combined_vine/
```
Then compare against the canonical combined run (see "Compare" below).

---

## Why (evidence, from `vine_evidence_poc.py` on the REAL combined pairs, N=32,712)

Real Kendall-τ among the four session variables (A=arrival, D=duration, G=idle gap, E=energy):

| pair | τ | modelled now? |
|------|-----|---------------|
| **D–E** | **0.232** | ✗ ignored |
| **G–E** | **0.231** | ✗ ignored |
| A–D | 0.110 | ✗ ignored |
| A–E | 0.098 | ✗ ignored |
| A–G | 0.059 | ✗ ignored |
| D–G | 0.038 | ✓ (Gaussian copula) |

**SPARC currently couples only D–G — the *weakest* pair (τ=0.038) — and ignores the two
strongest, both involving energy (D–E, G–E ≈ 0.23).** Energy is tied to dwell and gap, and the
generator does not condition on that; this is the likely cause of the combined load-profile
overshoot.

Copula model comparison on (A,D,G,E) (lower AIC and τ-error = better):

| model | log-lik | AIC | mean \|τ error\| |
|-------|---------|-----|------------------|
| Gaussian (≈ current) | 4,653 | −9,294 | 0.0150 |
| Student-t | 4,953 | −9,883 | 0.0117 |
| **Vine (full family)** | **7,459** | **−14,905** | **0.0100** |

The full vine fits ~1.6× the log-likelihood of the Gaussian and best recovers the dependence.

---

## Changes in this folder

**1. [APPLIED, low-risk] Extended parametric family set** in `fit_best`
(`smc_core_vine.jl`): added **Gamma** and **InverseGaussian** to `{LogNormal, Exponential,
Weibull}`, still BIC-selected. Should help dwell/energy marginals. *Untested — needs a run.*

**2. [TO IMPLEMENT + TEST] Vine / energy-coupled correction** (the main upgrade).
Stage 3 (`# ----- L2 ... convolution/copula`) currently fits a **bivariate Gaussian copula on
(D,G)** per failing bin and stores ρ; Stage 4 samples (D,G) jointly from it. Replace with a
joint over **(D, G, E)** — at minimum couple **E** to **D** and **G** (that is where the
dependence is), ideally a small **C-vine** on (A,D,G,E) per profile:
- fit: rank-transform to pseudo-obs, fit pair-copulas (Gaussian/t/Clayton/Gumbel), store the vine;
- sample: draw U from the vine, invert each variable through its stored marginal CDF.
- Julia has no mature vine package; either implement the 3–4 var C-vine by hand (pair-copula
  construction) or call `pyvinecopulib` via `PythonCall.jl`. Validate against
  `vine_evidence_poc.py` numbers before trusting it.

**3. [TO IMPLEMENT + TEST] AICc model selection.** Swap the BIC penalty in `fit_best`
(and in the GMM path) for AICc = `2k − 2ℓ + 2k(k+1)/(n−k−1)`, exposed as a `SELECTION`
constant (`"BIC"|"AIC"|"AICc"`). Report a BIC/AIC/AICc ablation on combined L4. Requires
plumbing the GMM fit to return (ℓ, p) so one criterion is applied uniformly — left undone
here to avoid shipping untested selection code.

---

## Compare (after the vine run finishes)

```bash
# 1. point the comparison at the vine SPARC sim (edit or copy):
#    plots_and_metrics.py 'Ours (SMC)' path -> artifacts/combined_vine/sim_sessions.csv
# 2. baseline synthetic files (copula/gmmnet/evsdg) are already built for combined-200.
python benchmarking/plots_and_metrics.py combined
```
Key things to check vs the canonical combined run: **L4 idle-gap & energy KL**, the
**Kendall-τ / tail-dependence** recovery, and whether the **evening load-profile overshoot**
shrinks (that is the metric the energy-coupling should move).

---

## Status / caveats

- The **evidence PoC is validated** (runs here). The **Julia edits are staged but UNTESTED** —
  Julia is not runnable in the assistant sandbox, so the safe family change and any vine/AICc
  work must be compiled and run on your machine (`julia run_smc_vine.jl combined`).
- Nothing here overwrites the canonical `smc_core.jl`, `run_smc.jl`, or `artifacts/combined_module/`.
