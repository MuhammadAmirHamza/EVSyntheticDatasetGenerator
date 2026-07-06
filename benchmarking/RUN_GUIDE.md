# How to run the baseline comparison

Two environments are involved:
- **Python** (baselines + comparison) — a normal Python 3.10+ env.
- **Julia** — only for our own SMC generator (`run_smc.jl`).

## 0. Python dependencies (one-time)

```
pip install numpy pandas scipy scikit-learn matplotlib torch
pip install pyvinecopulib ctgan
# EV-SDG needs pandas < 2.0 — run it in its own venv:
#   python -m venv benchmarking/evsdg/.venv ; activate ; pip install -r benchmarking/evsdg/requirements.txt
```

## 1. Generate each generator's synthetic sessions

Run from the project root (`D:\codes\SDG\SDG`).

```
# (a) Our method — Julia (produces artifacts/<dataset>/sim_sessions.csv)
julia run_smc.jl combined
julia run_smc.jl norway

# (b) EV-SDG — writes benchmarking/evsdg/results/EVSDG_*_generated.csv
python benchmarking/evsdg/run_baseline.py            # (uses the preloaded transactions)

# (c) GMMNet (conditional density + Gibbs)
python benchmarking/gmmnet/run_gmmnet.py combined --epochs 200
python benchmarking/gmmnet/run_gmmnet.py norway   --epochs 200

# (d) Vine copula (dropped from the paper, but runnable)
python benchmarking/copula/run_copula.py combined

# (e) Deep generative — TVAE or CTGAN. Use FULL data + 300 epochs for the real number:
python benchmarking/ctgan/run_ctgan.py combined --model tvae  --epochs 300
python benchmarking/ctgan/run_ctgan.py combined --model ctgan --epochs 300
```

Each writes `results/<generator>_<dataset>.csv` with columns `arrival_hour, duration_h, energy_kwh`.

## 2. Compare + plots

```
python benchmarking/compare.py                    # marginal KL/JS/KS/W1 + ranking (combined)
python benchmarking/metrics.py                    # full panel: per-user, dependence, C2ST, load, gap, privacy
python benchmarking/plots_and_metrics.py combined # marginal/kl/load plots -> plots/comparison/combined/
python benchmarking/plots_and_metrics.py norway
python benchmarking/paper_results.py              # the paper figure+table (per-user + sequential story)
```

## Notes
- **Only `run_smc.jl` needs Julia**; everything else is Python. The comparison scripts read whatever
  CSVs are present in each `results/` folder, so you can re-run one generator and re-compare.
- **TVAE/CTGAN are compute-heavy** (~300 epochs on full data; GPU preferred). Sandbox-limited runs
  under-train them and under-state their quality.
- **EV-SDG discards sessions > 24 h** by design (public-charging assumption) — hence its poor
  residential duration/energy fidelity; this is inherent, not a bug.
- Add a new baseline by dropping its `results/*.csv` and adding one line to the `GENERATORS`
  lists in `compare.py` / `metrics.py` / `plots_and_metrics.py`.
