#!/usr/bin/env python3
"""Collate the low-data sensitivity sweep (reviewer #5).

Reads artifacts/lowdata_n*/ and reports, per n (= sessions/user used to FIT):

  internal (vs the truncated real input, produced by Stage 5):
    L1 pass-% per variable, L2 %, L3 %, L4 KL (nats)
  vs FULL history (the meaningful sparse-data test — fit on first n, compare
  against the user's complete real record):
    KLfull_A/D/E : population KL, same binning as the paper's L4
                   (24 clock-hour bins for A; 50 equal-width bins for D, E)
    MAE_dur/MAE_E/MAE_arr : per-user |mean_sim - mean_real_full|, median
                   across users (arrival uses circular distance, hours)

Caveat printed with the table: internal L1/L3 pass-% RISES as n shrinks
(KS power drops with sample size), so judge sparse-data robustness on the
vs-full columns, not on internal pass rates. Report both in the paper.

Usage: python collate_lowdata.py          (from paper_results/)
Writes: lowdata_sensitivity.csv, lowdata_table.tex
"""
import glob, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AD   = os.path.join(HERE, "artifacts")
FULL = os.path.join(HERE, "dataset", "combined", "combined_sessions.csv")
DELTA = 1e-9   # matches eq. (KL) in the paper

def tobool(s): return s.astype(str).str.lower().isin(["true", "1", "1.0"])

def rate(df, var=None):
    d = df if var is None else df[df.variable == var]
    return d.passed.mean() * 100 if len(d) else float("nan")

def kl(real, sim, bins):
    pr, edges = np.histogram(real, bins=bins)
    ps, _     = np.histogram(sim,  bins=edges)
    pr = pr / max(pr.sum(), 1); ps = ps / max(ps.sum(), 1)
    return float(np.sum(pr * np.log((pr + DELTA) / (ps + DELTA))))

def circ_dist(a, b):
    d = abs(a - b) % 24.0
    return min(d, 24.0 - d)

full = pd.read_csv(FULL)
full = full[(full.duration_hours > 0) & (full.energy > 0)].dropna(
    subset=["arrival_hour", "duration_hours", "energy"])

rows = []
for d in sorted(glob.glob(os.path.join(AD, "lowdata_n*"))):
    m = re.search(r"lowdata_n(\d+)", os.path.basename(d))
    if not m: continue
    n = int(m.group(1))
    def rd(p):
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.DataFrame(columns=["user_id", "variable", "passed"])
        return df
    try:
        l1 = rd(f"{d}/validation_l1.csv"); l2 = rd(f"{d}/validation_l2.csv")
        l3 = rd(f"{d}/validation_l3.csv"); l4 = pd.read_csv(f"{d}/validation_l4.csv")
        for x in (l1, l2, l3):
            x["passed"] = tobool(x["passed"]) if len(x) else x.get("passed", pd.Series(dtype=bool))
        if "p_value" in l1.columns:   # KS not computable below its sample minimum -> exclude, not fail
            l1 = l1[pd.to_numeric(l1["p_value"], errors="coerce").notna()]
        klint = dict(zip(l4["short"], l4["KL_nats"]))
        sim = pd.read_csv(f"{d}/sim_sessions.csv")
    except Exception as e:
        print(f"  n={n}: {e}"); continue

    users = sim.user_id.unique()
    fr = full[full.user_id.isin(users)]

    # population KL vs FULL history (paper L4 binning)
    edges_A = np.arange(25)
    klA = kl(fr.arrival_hour % 24, sim.arrival_hour % 24, edges_A)
    klD = kl(fr.duration_hours, sim.duration_h,
             np.linspace(0, max(fr.duration_hours.max(), sim.duration_h.max()), 51))
    klE = kl(fr.energy, sim.energy_kwh,
             np.linspace(0, max(fr.energy.max(), sim.energy_kwh.max()), 51))

    # per-user recovery of the user's own mean behavior, vs FULL history
    mae_d, mae_e, mae_a = [], [], []
    gs = sim.groupby("user_id"); gf = fr.groupby("user_id")
    for u in users:
        if u not in gf.groups: continue
        s, f = gs.get_group(u), gf.get_group(u)
        mae_d.append(abs(s.duration_h.mean()  - f.duration_hours.mean()))
        mae_e.append(abs(s.energy_kwh.mean()  - f.energy.mean()))
        # circular means for arrival
        def cmean(h):
            th = 2 * np.pi * np.asarray(h) / 24.0
            return (np.arctan2(np.sin(th).mean(), np.cos(th).mean()) % (2 * np.pi)) * 24 / (2 * np.pi)
        mae_a.append(circ_dist(cmean(s.arrival_hour), cmean(f.arrival_hour)))

    rows.append(dict(
        n=n, users=len(users),
        L1A=rate(l1, "A"), L1D=rate(l1, "D"), L1G=rate(l1, "G"), L1E=rate(l1, "E"),
        L2=rate(l2), L3=rate(l3),
        KLintA=klint.get("A"), KLintD=klint.get("D"), KLintG=klint.get("G"), KLintE=klint.get("E"),
        KLfullA=klA, KLfullD=klD, KLfullE=klE,
        MAEdur=float(np.median(mae_d)), MAEen=float(np.median(mae_e)), MAEarr=float(np.median(mae_a)),
    ))

if not rows:
    print("No artifacts. Run: for n in 10 20 30 50 100; do julia --project=. run_lowdata_N.jl $n; done")
    raise SystemExit

t = pd.DataFrame(rows).sort_values("n")
t.to_csv(os.path.join(HERE, "lowdata_sensitivity.csv"), index=False)
pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.3f}")
print(t.to_string(index=False))
print("\nNOTE: internal L1/L3 pass-% rises as n shrinks (KS loses power at small n);")
print("judge robustness on KLfull_* and MAE_* (fit on first n, tested on full history).")

# compact LaTeX table for the paper
with open(os.path.join(HERE, "lowdata_table.tex"), "w") as f:
    f.write("% auto-generated by collate_lowdata.py\n")
    f.write("\\begin{tabular}{r r ccc ccc}\n\\toprule\n")
    f.write("$n$ & Users & $\\mathrm{KL}_A$ & $\\mathrm{KL}_D$ & $\\mathrm{KL}_E$ & "
            "MAE$_{\\mathrm{dur}}$ (h) & MAE$_{E}$ (kWh) & MAE$_{\\mathrm{arr}}$ (h) \\\\\n")
    f.write("\\midrule\n")
    for _, r in t.iterrows():
        f.write(f"{int(r.n)} & {int(r.users)} & {r.KLfullA:.3f} & {r.KLfullD:.3f} & {r.KLfullE:.3f} & "
                f"{r.MAEdur:.2f} & {r.MAEen:.2f} & {r.MAEarr:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("wrote lowdata_sensitivity.csv, lowdata_table.tex")
