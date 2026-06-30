# EV-SDG baseline (adapted for residential, combined dataset)

Self-contained EV-SDG baseline used in the paper. Original method:
Lahariya, Benoit, Develder — *Defining a Synthetic Data Generator for Realistic
EV Charging Sessions* (github.com/mlahariya/EV-SDG). Arrivals = exponential
inter-arrival process per time-slot; connection time and energy = Gaussian
mixtures. This copy is the **clean re-download plus the minimal audited patches**
needed to run it on residential / single-year data.

## Run it (already done — outputs are in `results/`)

```bash
pip install -r requirements.txt      # pandas MUST be < 2.0
python run_baseline.py               # preprocess -> fit -> generate -> results/
```

Input : `res/transactions.csv` (the 140 combined profiles, 4-column EV-SDG format,
dates remapped to a common year 2019 — the marginals we score are year-invariant).
Output: `results/EVSDG_combined_generated.csv` and `results/baseline_metrics_combined.csv`.

## Result on the combined set (140 users, 28,159 sessions)

| variable     | KL (real‖synth) | JS    | KS    | W1   |
|--------------|-----------------|-------|-------|------|
| arrival hour | 0.040           | 0.011 | 0.098 | 1.26 |
| plug-in dur. | 1.706           | 0.039 | 0.097 | 2.93 |
| energy       | 2.853           | 0.043 | 0.123 | 2.93 |
| **mean KL**  | **1.533**       |       |       |      |

Strong on aggregate arrival shape, weak on duration/energy — a single pooled GMM
cannot capture per-user heterogeneity. Two independent clean runs agree to 3 dp.

## Audit of the modifications (none favour our method)

EV-SDG was built for ELaadNL public charging (thousands of poles, multi-year).
Running it on residential cohorts needs:

* **Python/pandas compat** (behaviour-preserving): `sklearn.metrics.scorer →
  sklearn.metrics`, `np.in1d → np.isin`, `df.drop(col,1) → drop(col,axis=1)`,
  `as_matrix() → values`, `str.split("_",-1) → n=-1`, `astype(object)`.
* **Year bug-fix:** clean code hard-codes `y_train=[2015]` (ELaadNL's year) → on
  any other dataset it trains on a year with zero sessions. Fixed to `[year]`.
* **Residential clustering density:** `minpoints 440→15`, session DBSCAN
  `min_samples 6→4`; robust nearest-slot fallback in `mixturemodels`.
* **Single-year combined run:** pole selector `continous→topn` (continous needs
  adjacent years), dynamic session-cluster column names, and a non-finite guard
  on GMM `precisions_init` (single-observation slots → variance 0 → ∞).

EV-SDG's clustering **degenerates on the small individual cohorts** (Korea 30,
Pecan 10 homes) — too few homes to form behaviour clusters. It runs only on the
rich combined set, which is itself evidence for the per-user approach.
