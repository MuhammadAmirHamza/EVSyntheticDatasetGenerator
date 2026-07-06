# SPARC on ACN-Data (Caltech workplace charging)

Demonstrates SPARC beyond residential — on **workplace** charging, where each
recurring driver (`userID`) has a habitual routine (exactly SPARC's assumption).

## Why this must run on your machine
- ACN **session attributes** (connection/disconnect time, kWh, userID) are served
  only by the ACN-Data **API, which needs a free token** — account registration I
  can't do for you. (The static GitHub mirror only holds per-session power
  time-series, not the session table.)
- **SPARC is Julia** (`run_smc.jl`), which doesn't run in the assistant's sandbox.

Everything below is set up; it's ~4 commands.

## Steps

```bash
# 1. Get a free ACN token (instant): https://ev.caltech.edu/dataset  -> register

# 2. Preprocess ACN -> SPARC schema (recurring users, top-30 by session count)
pip install requests pandas
python dataset/acn_caltech/acn_preprocess.py --token YOUR_TOKEN --site caltech
#   (or, if you downloaded the sessions JSON yourself:)
#   python dataset/acn_caltech/acn_preprocess.py --json caltech_sessions.json
#   -> writes dataset/acn_caltech/acn_sessions.csv

# 3. Run SPARC on ACN  (registered as the 'acn' config in run_smc.jl)
julia run_smc.jl acn
#   -> artifacts/acn_caltech/  (sim_sessions.csv, kl_divergence.txt, validation_l1..l4)

# 4. (optional) baselines + comparison on ACN
python benchmarking/gmmnet/run_gmmnet.py acn --epochs 200
python benchmarking/copula/run_copula.py acn
python benchmarking/ctgan/run_ctgan.py  acn --model tvae --epochs 300
python benchmarking/plots_and_metrics.py acn      # after adding 'acn' to its REAL map
```

## Selection knobs
- `acn_preprocess.py`: `N_MIN` (min sessions/user, default 30) and `TOP_N`
  (number of users, default 30) — change TOP_N for a different "number of users".
- `run_smc.jl` `acn` config: `top_n=30, q_upper=1.0, upper_inclusive=true,
  num_bins=24, a_kmax=6` (small-cohort settings, like Pecan). Match `top_n` to TOP_N.

## Expected result
SPARC should reproduce per-user workplace routines and the idle gap between a
commuter's sessions — showing the method generalizes from residential to
workplace charging. (`plots_and_metrics.py` needs one line added to its `REAL`
map for `acn`; tell me and I'll wire it.)
