# SPARC — Project State & Handoff (for continuing next session)

> Working notes so we can resume without re-deriving context. Assistant has no
> cross-session memory; **this file is the anchor — point me here next time.**

## 1. What the project is
Revising a manuscript for **IEEE Transactions on Smart Grid (TSG)**, submit as a
**Research Paper** (not Application), **≤10 pages** (first-submission hard limit).

**Framework name: SPARC** = *Semi-markov Per-entity chARging Chains*
(originally "Synthetic Per-user chARging Chains"; broadened P: per-user → per-entity).

**Method:** per-entity semi-Markov generator of synthetic EV charging sessions.
Each entity = an identifiable charging point/user with a recurring, patterned routine.
Session-as-state encoding; per-bin BIC-selected distributions (arrival, duration,
idle-gap, energy); per-entity transition matrix; sojourn; Gaussian-copula correction
where duration–gap dependence detected; 3-level fallback for sparse data; weekday/
weekend two-stream simulation; four-level validation (L1 marginal / L2 transition /
L3 sojourn / L4 population KL).

## 2. KEY DECISION — scope (locked)
Reframe from "residential" to **per-entity, recurring-pattern charging**, treating
**residential (per-household)** and **workplace (per-station)** on **equal footing**
from the start. Scope boundary: **excludes transient public/fast charging**
(one-off drivers, no per-entity pattern).
- Title to become per-entity (drop "Residential"), e.g. *"Capturing Per-Entity
  Heterogeneity in Synthetic EV Charging Sessions via Semi-Markov Chains."*

## 3. Datasets (all in SPARC 16-col schema)
- **Norway** (Sørensen apartments, per-household) — residential. 100 users used.
- **Korea** (GIST apartments) — residential. 30 users.
- **Pecan Street** (US homes) — residential. 10 users. *License-restricted: never commit
  raw/derived Pecan data (see .gitignore).*
- **Combined** = Norway100+Korea30+Pecan10 = 140 users, 28,159 sessions.
  Rebuild: `python dataset/combined/build_combined.py`.
- **ACN Caltech** (workplace) — keyed by **charging station** (per-driver userID is only
  ~1% coverage → unusable; per-station is data-rich). 15 stations, 3,871 sessions,
  209–374/station. Built OFFLINE from the download checkpoint. `dataset/acn_caltech/`.

## 4. Baselines (in benchmarking/)
| Baseline | Class | Folder | Status |
|---|---|---|---|
| **EV-SDG** (Lahariya 2020) | parametric | `benchmarking/evsdg/` | done (has 24h session cap → poor residential dur/energy) |
| **GMMNet** (Li 2024, cond. density + Gibbs) | neural | `benchmarking/gmmnet/` | done |
| **Vine copula** (semiparametric) | statistical | `benchmarking/copula/` | done (marginals empirical → trivially low; judge on dependence) |
| **CTGAN / TVAE** (Xu 2019) | deep generative | `benchmarking/ctgan/` | code done; **TVAE compute-limited in sandbox** — run full 300 epochs on user machine |
SPARC itself is Julia (`run_smc.jl` / `smc_core.jl`) — runs on the user's machine only.

## 5. Headline results
**Combined (marginal mean KL, lower=better):** Ours 0.008 · GMMNet 0.012 · TVAE ~0.31(under-trained) · EV-SDG 1.53 · Vine 0.0007*(empirical).
**Full panel (combined):** Ours wins pop_KL + **idle-gap (0.036, unique)**; GMMNet best on dependence/C2ST/load; per-user KL was noise-dominated (use per-user MAE instead).
**Per-user heterogeneity (per-user median MAE):** Ours 0.50h/0.65h/0.66kWh vs GMMNet 1.7/4.7/6.2 → **3–9× better** (baselines assign one profile to all users).
**Norway:** Ours 0.010 · GMMNet 0.020 · EV-SDG 1.67. EV-SDG stays bad (24h cap, not heterogeneity).
**ACN workplace (per-station):** Ours **0.043** · GMMNet 0.094 · Vine 0.012*(empirical). SPARC generalizes AND beats GMMNet; idle-gap 0.02 (unique).

**Two-part story:** residential = SPARC *beats* baselines (per-user, low-data);
workplace = SPARC *generalizes* and still wins; idle-gap/sequential = SPARC-only everywhere.

## 6. Deliverables already produced (in D:\codes\SDG\SDG unless noted)
- PDFs: `EV-SDG_baseline_summary.pdf`, `GMMNet_summary.pdf`, `VineCopula_summary.pdf`,
  `CTGAN_TVAE_summary.pdf`, `Results_heterogeneity_sequence.pdf`, `metrics_explained.pdf`.
- `literature_to_add.md` (17 papers not yet cited, with venues/roles).
- `benchmarking/` : `compare.py`, `metrics.py`, `plots_and_metrics.py`, `paper_results.py`,
  `RUN_GUIDE.md`, `metrics_panel.csv`, per-baseline `run_*.py`.
- Plots: `plots/comparison/{combined,norway,acn}/` and `plots/paper/`.
- Paper: **`D:\codes\SDG\SPARC\`** (renamed from Synthetic_Residential_...).
  `main_paper.tex` converted **conference → journal IEEEtran** (`\documentclass[journal]`,
  journal author block + `\markboth`). Compiles on user's MiKTeX (sandbox lacks packages).

## 7. Open items / next steps
1. **Manuscript content rewrite** (journal format already done): global user→entity + scope
   pass; rewrite abstract + intro + contributions (TSG-grade, human English); datasets +
   master results tables; add ACN/workplace; keep ≤10 pp. *Plan agreed; not started.*
2. **Temporal-consistency test** (paper currently lacks one). Planned panel:
   (a) idle-gap autocorrelation, (b) day×hour (7×24) weekly-rhythm KL/heatmap,
   (c) weekly 168-h load-profile MAE. Mostly SPARC-only (baselines are i.i.d.). *Not built.*
3. **Citations to add:** CTGAN/TVAE (NeurIPS 2019), GMMNet (Comm. Transp. Res. 2024),
   Data-in-Brief CTGAN+KDE (2024), Razghandi VAE-GAN (IEEE TSG 2023), EV copula (Energy 2022;
   vine+CODINE 2026), survey (2025).
4. **Optional:** full-epoch TVAE/CTGAN on user machine; per-entity baseline experiment;
   ACN load-profile PDF; email extension to Prof. Wu (drafted, until weekend).

## 8. Gotchas / environment facts (save me from re-learning)
- **Julia not runnable in assistant sandbox** → SPARC runs on user's machine; assistant uses
  the user's existing `artifacts/<ds>/sim_sessions.csv`.
- `run_smc.jl` has `SUFFIX="_module"` (verification mode) → SPARC writes to
  `artifacts/<name>_module/`. Set `SUFFIX=""` for canonical dir.
- **Mount write truncation:** editing files on the mount sometimes truncates the tail;
  after any write, verify (`ast.parse` / check `\end{document}` / tail). Prefer full atomic
  rewrites for large files.
- **Assistant folder access:** Read/Write/Edit tools reach only `D:\codes\SDG\SDG`; the paper
  at `D:\codes\SDG\SPARC` must be edited via shell.
- **Deletion** on the mount needs the cowork delete-permission tool (rm otherwise "Operation
  not permitted"); directory **rename** at the mount root is blocked (copy+delete instead).
- ACN: session table is API-only (token needed; ~1% userID). Fetch script checkpoints to
  `dataset/acn_caltech/_acn_raw_caltech.jsonl` and resumes on re-run.
- TVAE/CTGAN and long fits get reaped past ~45s in sandbox → run on user machine.
