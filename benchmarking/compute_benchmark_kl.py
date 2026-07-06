#!/usr/bin/env python3
"""Uniform KL(A/D/E) benchmark: SPARC vs EV-SDG, GMMNet, Einolander copula.

For each dataset, all methods are scored against the SAME real reference
(raw_clean.csv from the SPARC artifact) using IDENTICAL histogram edges per
quantity. KL is the forward divergence KL(real || model) in nats.
"""
import os, glob
import numpy as np
import pandas as pd

ART = "paper_results/artifacts"
EVS = "benchmarking/evsdg/results"
GMM = "benchmarking/gmmnet/results"
COP = "benchmarking/copula/results"

# dataset -> (SPARC artifact prefix, seeds, baseline key)
DATASETS = {
    "Norway":    ("norway_sorensen_N100_b12", "norway"),
    "GIST-Res":  ("korea_gist_N030_b24",      "gist"),
    "GIST-Comm": ("kim_commercial_N025_b24",  "kim"),
    "ACN":       ("acn_caltech_N015_b12",     "acn"),
    "Combined":  ("combined_all_N200_b24",    "combined"),
}

def kl(real, model, edges):
    pr, _ = np.histogram(real, bins=edges); pm, _ = np.histogram(model, bins=edges)
    pr = pr.astype(float); pm = pm.astype(float)
    pr += 1e-9; pm += 1e-9
    pr /= pr.sum(); pm /= pm.sum()
    return float(np.sum(pr * np.log(pr / pm)))

def edges_for(x, lo=0.0, hi_pct=99.0, n=30, hard_hi=None):
    hi = hard_hi if hard_hi is not None else np.percentile(x, hi_pct)
    return np.linspace(lo, hi, n + 1)

rows = []
for name, (prefix, key) in DATASETS.items():
    # --- real reference ---
    real_path = sorted(glob.glob(f"{ART}/{prefix}_s01/raw_clean.csv"))
    if not real_path:
        print(f"[skip] no real for {name}"); continue
    r = pd.read_csv(real_path[0])
    rA = r["arrival_hour"].dropna().values
    rD = r["duration_hours"].dropna().values
    rE = r["energy"].dropna().values
    rD = rD[(rD > 0)]; rE = rE[(rE > 0)]

    # common edges from real
    eA = np.linspace(0, 24, 25)
    eD = edges_for(rD, 0, 99, 30)
    eE = edges_for(rE, 0, 99, 30)

    # --- SPARC: pool sim across seeds ---
    sim = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{ART}/{prefix}_s0*/sim_sessions.csv"))],
                    ignore_index=True)
    methods = {"SPARC": (sim["arrival_hour"], sim["duration_h"], sim["energy_kwh"])}

    # --- baselines ---
    for label, path in [("EV-SDG", f"{EVS}/EVSDG_{key}.csv"),
                        ("GMMNet", f"{GMM}/gmmnet_{key}.csv"),
                        ("Copula", f"{COP}/einolander_{key}.csv")]:
        if os.path.exists(path):
            m = pd.read_csv(path)
            methods[label] = (m["arrival_hour"], m["duration_h"], m["energy_kwh"])
        else:
            methods[label] = None

    for label, cols in methods.items():
        if cols is None:
            rows.append([name, label, np.nan, np.nan, np.nan]); continue
        A, D, E = [pd.to_numeric(c, errors="coerce").dropna().values for c in cols]
        D = D[D > 0]; E = E[E > 0]
        # arrival wrap into [0,24)
        A = np.mod(A, 24)
        rows.append([name, label,
                     kl(rA, A, eA), kl(rD, D, eD), kl(rE, E, eE)])

df = pd.DataFrame(rows, columns=["Dataset", "Method", "KL_A", "KL_D", "KL_E"])
df["KL_mean"] = df[["KL_A", "KL_D", "KL_E"]].mean(axis=1)
df.to_csv("benchmarking/benchmark_kl.csv", index=False)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(df.to_string(index=False))
