# benchmarking/

Baselines for the per-user semi-Markov EV-charging generator, evaluated on the
**combined** dataset (Norway 100 + Korea 30 + Pecan 10 = 140 exact selected
profiles, 28,159 sessions; built at `../dataset/combined/combined_sessions.csv`).

## Layout

```
benchmarking/
├── evsdg/                     Baseline #1 — EV-SDG (self-contained)
│   ├── run_baseline.py        one command: preprocess → fit → generate
│   ├── res/transactions.csv   combined profiles in EV-SDG input format
│   ├── results/               ← baseline outputs live here
│   │   ├── EVSDG_combined_generated.csv
│   │   └── baseline_metrics_combined.csv
│   ├── requirements.txt        (pandas < 2.0)
│   ├── README.md               method notes + audit of patches
│   └── <EV-SDG source code>
├── compare.py                 head-to-head: real vs every generator
└── README.md                  this file

   (baseline #2 and #3 to be added as sibling folders, e.g. evsdg/ → copula/, ctgan/)
```

## Where the results are

* **EV-SDG baseline** → `evsdg/results/` (synthetic sessions + metrics CSV).
* **Our method** → `../artifacts/combined/` (`sim_sessions.csv`,
  `kl_divergence.txt`, validation_l1–l4) and plots in `../plots/combined/`.
* **Head-to-head table** → printed by `compare.py` (not written to disk).

## What to run

```bash
# 1. our method on the combined set  (Julia)
julia ../SMC_framework_combined.jl
julia ../results_generation_combined.jl

# 2. EV-SDG baseline  (optional — output already in evsdg/results/)
pip install -r evsdg/requirements.txt
python evsdg/run_baseline.py

# 3. compare everything
python compare.py
```

`compare.py` scores each generator's arrival-hour, duration, and energy
distributions against the real combined data with identical binning
(KL / JS / KS / Wasserstein) and prints a mean-KL ranking.

## Current standings (mean KL, lower = better)

| generator | mean KL |
|-----------|---------|
| Ours (per-user SMC) | run step 1 to fill in (≈0.02–0.03 on individual sets) |
| EV-SDG (pooled)     | **1.533** |

## Adding baselines #2 and #3

Create a sibling folder with the baseline's code + a `results/<gen>.csv`, then add
one line to the `GENERATORS` list in `compare.py` (label, path, column-map). Missing
files are skipped automatically.
