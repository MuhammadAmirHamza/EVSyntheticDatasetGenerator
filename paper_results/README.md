# paper_results — final SPARC runs for the manuscript

Self-contained copy of the framework for regenerating the paper's per-dataset
results. Julia is required (not available in the assistant sandbox), so the
runs below are executed by you.

## Files
- `smc_core.jl`, `run_smc.jl`  — framework (byte-identical to repo root)
- `run_norway_N.jl`            — sweep wrapper: Norway at an arbitrary TOP_N
- `collate_sweep.py`           — reads artifacts/norway_N*/ -> pass-rate table
- `results_generation_norway.jl` — paper plots for the chosen run
- `dataset/norway_sorensen/`   — input sessions

## Norway: choose the results-optimal N, then lock it
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
for N in 100 120 140 161; do julia --project=. run_norway_N.jl $N; done
python3 collate_sweep.py
```
`collate_sweep.py` prints L1(A/D/G/E), L2, L3 pass-% and L4 KL per N. Pick the
largest N whose pass rates do not drop vs N=100 (see NOTE).

### NOTE — why "more users" tends to *raise* pass-%
Pass rates rise monotonically 40->100 in the existing top-100 run (L1 85.9->90.1,
L3 74.7->86.4). This is largely a KS-power effect: users with more sessions fail
the KS test more often even for good fits. So the pass-% metric is confounded by
per-user sample size. Recommendation: report KS *effect size* (median KS stat)
alongside pass-% so the numbers are not read as pure fit quality.

n_min=50 (uniform inclusion rule) => TOP_N≈161 is the full qualifying Norway pop.
