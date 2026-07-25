#!/usr/bin/env python3
"""Downstream smart-grid validation (reviewer #3): do synthetic sessions
preserve the OPERATIONAL characteristics practical studies need?

Three analyses on the combined cohort:

A. Peak demand (time series; Real vs SPARC). Hourly fleet charging load
   L(t) = sum_sessions (E/D) over [start, start+D); daily-peak distribution
   (median, P95, max).

B. Coincidence factor CF(K) = max_t sum_{u in K} l_u(t) / sum_u max_t l_u(t),
   bootstrap groups of K users.
   - time-series CF: Real vs SPARC (needs calendars; baselines have none)
   - average-profile CF: all generators, baselines via pseudo-users whose
     session counts and observation days are matched to real users
     (pooled bags carry no identity, cf. compute_heterogeneity.py).

C. Transformer pocket loading: K=10 users on a nominal 50-kVA transformer,
   charging-only load; P95 pocket peak and % of hours above 60/80/100% of
   rating (Real vs SPARC, time series).

Session power is E/D, clipped at 22 kW (L2 ceiling) for every generator.
Outputs: downstream_metrics.csv, downstream_table.tex, cf_curve.pdf
"""
import argparse, os
import numpy as np
import pandas as pd

RNG   = np.random.default_rng(0)
PCLIP = 22.0          # kW
RATING = 50.0         # kVA nominal pocket transformer
K_POCKET = 10
K_GRID = [2, 5, 10, 20, 50]
R_BOOT = 200          # bootstrap groups per K (time series)
R_PSEUDO = 30         # pseudo-user partitions (baselines)

ap = argparse.ArgumentParser()
ap.add_argument("--real"); ap.add_argument("--sparc")
ap.add_argument("--copula"); ap.add_argument("--gmmnet"); ap.add_argument("--evsdg")
ap.add_argument("--out", default=".")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------- load ------------------------------------------------------------
real = pd.read_csv(args.real)
real["start"] = pd.to_datetime(real["abs_start"])
real = real[(real.duration_hours > 0) & (real.energy > 0)]
real = real.rename(columns={"duration_hours": "D", "energy": "E"})

sparc = pd.read_csv(args.sparc)
sparc["start"] = pd.to_datetime(sparc["calendar_date"]) + pd.to_timedelta(sparc["arrival_hour"], unit="h")
sparc = sparc[(sparc.duration_h > 0) & (sparc.energy_kwh > 0)]
sparc = sparc.rename(columns={"duration_h": "D", "energy_kwh": "E"})

bags = {}
for lab, p in [("Copula", args.copula), ("GMMNet", args.gmmnet), ("EV-SDG", args.evsdg)]:
    if not p: continue
    m = pd.read_csv(p)
    m = m.apply(pd.to_numeric, errors="coerce").dropna()
    m = m[(m.duration_h > 0) & (m.energy_kwh > 0)]
    bags[lab] = m

# ---------- hourly series ----------------------------------------------------
def hourly_series(df):
    """dict user -> (t0_hourindex, np.array hourly kW). Hour index = hours since global epoch."""
    epoch = pd.Timestamp("2015-01-01")
    out = {}
    for u, g in df.groupby("user_id"):
        h0 = int(np.floor((g.start.min() - epoch).total_seconds() / 3600))
        h1 = int(np.ceil(((g.start + pd.to_timedelta(g.D, unit="h")).max() - epoch).total_seconds() / 3600)) + 1
        L = np.zeros(h1 - h0)
        for s, d, e in zip(g.start, g.D, g.E):
            p = min(e / d, PCLIP)
            a = (s - epoch).total_seconds() / 3600 - h0
            end = a + d
            h = int(np.floor(a))
            while h < end:
                lo, hi = max(a, h), min(end, h + 1)
                if hi > lo and 0 <= h < len(L): L[h] += p * (hi - lo)
                h += 1
        out[u] = (h0, L)
    return out

def fleet_series(user_series):
    h0 = min(v[0] for v in user_series.values())
    h1 = max(v[0] + len(v[1]) for v in user_series.values())
    F = np.zeros(h1 - h0)
    for o, L in user_series.values():
        F[o - h0 : o - h0 + len(L)] += L
    return h0, F

def daily_peaks(h0, F):
    n = (len(F) // 24) * 24
    return F[:n].reshape(-1, 24).max(axis=1)

def group_cf(user_series, users):
    off = [user_series[u] for u in users]
    h0 = min(o for o, _ in off); h1 = max(o + len(L) for o, L in off)
    S = np.zeros(h1 - h0); pk = 0.0
    for o, L in off:
        S[o - h0 : o - h0 + len(L)] += L
        pk += L.max() if len(L) else 0.0
    return S.max() / pk if pk > 0 else np.nan

print("building hourly series (real)..."); US_real = hourly_series(real)
print("building hourly series (sparc)..."); US_sparc = hourly_series(sparc)
users = sorted(set(US_real) & set(US_sparc))
print(f"users in both: {len(users)}")

rows = []
PKS = {}
# ---------- A. peak demand ----------------------------------------------------
for lab, US in [("Real", US_real), ("SPARC", US_sparc)]:
    h0, F = fleet_series({u: US[u] for u in users})
    pk = daily_peaks(h0, F)
    PKS[lab] = pk[pk > 0]
    rows.append(dict(metric="daily_peak_median_kW", gen=lab, value=float(np.median(pk))))
    rows.append(dict(metric="daily_peak_P95_kW",    gen=lab, value=float(np.quantile(pk, 0.95))))
    rows.append(dict(metric="daily_peak_max_kW",    gen=lab, value=float(pk.max())))

# ---------- B1. time-series CF(K) ----------------------------------------------
cf_ts = {}
for lab, US in [("Real", US_real), ("SPARC", US_sparc)]:
    cf_ts[lab] = {}
    for K in K_GRID + [len(users)]:
        if K > len(users): continue
        vals = [group_cf(US, RNG.choice(users, size=K, replace=False)) for _ in range(R_BOOT if K < len(users) else 1)]
        cf_ts[lab][K] = float(np.nanmean(vals))
        rows.append(dict(metric=f"CF_ts_K{K}", gen=lab, value=cf_ts[lab][K]))

# ---------- B2. average-profile CF (all generators) ---------------------------
def avg_profile_users(df):
    """per real user: average 24h profile (kW) = session loads on hour-of-day axis / observed days"""
    prof = {}
    for u, g in df.groupby("user_id"):
        days = max((g.start.max() - g.start.min()).days, 1)
        P24 = np.zeros(24)
        A = (g.start.dt.hour + g.start.dt.minute / 60).values
        for a, d, e in zip(A, g.D, g.E):
            p = min(e / d, PCLIP); end = a + d; h = int(np.floor(a))
            while h < end:
                lo, hi = max(a, h), min(end, h + 1)
                if hi > lo: P24[h % 24] += p * (hi - lo)
                h += 1
        prof[u] = P24 / days
    return prof

def cf_from_profiles(profs, K, R=R_BOOT):
    keys = list(profs)
    vals = []
    for _ in range(R):
        gs = RNG.choice(keys, size=K, replace=False)
        S = np.sum([profs[k] for k in gs], axis=0)
        pk = sum(profs[k].max() for k in gs)
        vals.append(S.max() / pk if pk > 0 else np.nan)
    return float(np.nanmean(vals))

def pseudo_profiles(bag, real_df):
    """assign pooled sessions to pseudo-users matching real (count, days) pairs"""
    meta = [(len(g), max((g.start.max() - g.start.min()).days, 1)) for _, g in real_df.groupby("user_id")]
    A = bag.arrival_hour.values % 24; D = bag.duration_h.values; E = bag.energy_kwh.values
    n = len(bag); profs_all = []
    for r in range(R_PSEUDO):
        idx = RNG.permutation(n); pos = 0; profs = {}
        for j, (c, days) in enumerate(meta):
            if pos + c > n: break
            sl = idx[pos:pos + c]; pos += c
            P24 = np.zeros(24)
            for a, d, e in zip(A[sl], D[sl], E[sl]):
                p = min(e / d, PCLIP); end = a + d; h = int(np.floor(a))
                while h < end:
                    lo, hi = max(a, h), min(end, h + 1)
                    if hi > lo: P24[int(h) % 24] += p * (hi - lo)
                    h += 1
            profs[j] = P24 / days
        profs_all.append(profs)
    return profs_all

real_prof  = avg_profile_users(real[real.user_id.isin(users)])
sparc_prof = avg_profile_users(sparc[sparc.user_id.isin(users)])
cf_prof = {"Real": {}, "SPARC": {}}
for K in K_GRID:
    cf_prof["Real"][K]  = cf_from_profiles(real_prof, K)
    cf_prof["SPARC"][K] = cf_from_profiles(sparc_prof, K)
    rows.append(dict(metric=f"CF_prof_K{K}", gen="Real",  value=cf_prof["Real"][K]))
    rows.append(dict(metric=f"CF_prof_K{K}", gen="SPARC", value=cf_prof["SPARC"][K]))
for lab, bag in bags.items():
    cf_prof[lab] = {}
    pp = pseudo_profiles(bag, real[real.user_id.isin(users)])
    for K in K_GRID:
        v = [cf_from_profiles(p, K, R=20) for p in pp if len(p) >= K]
        cf_prof[lab][K] = float(np.nanmean(v))
        rows.append(dict(metric=f"CF_prof_K{K}", gen=lab, value=cf_prof[lab][K]))

# ---------- C. transformer pocket ---------------------------------------------
POCKET = {}
for lab, US in [("Real", US_real), ("SPARC", US_sparc)]:
    pks, o60, o80, o100 = [], [], [], []
    for _ in range(R_BOOT):
        gs = RNG.choice(users, size=K_POCKET, replace=False)
        off = [US[u] for u in gs]
        h0 = min(o for o, _ in off); h1 = max(o + len(L) for o, L in off)
        S = np.zeros(h1 - h0)
        for o, L in off: S[o - h0 : o - h0 + len(L)] += L
        pks.append(np.quantile(S, 0.999))
        o60.append((S > 0.6 * RATING).mean() * 100)
        o80.append((S > 0.8 * RATING).mean() * 100)
        o100.append((S > RATING).mean() * 100)
    rows.append(dict(metric="pocket_P999_kW", gen=lab, value=float(np.mean(pks))))
    rows.append(dict(metric="pocket_hrs_gt60pct", gen=lab, value=float(np.mean(o60))))
    rows.append(dict(metric="pocket_hrs_gt80pct", gen=lab, value=float(np.mean(o80))))
    rows.append(dict(metric="pocket_hrs_gt100pct", gen=lab, value=float(np.mean(o100))))
    POCKET[lab] = dict(P999=float(np.mean(pks)), o60=float(np.mean(o60)),
                       o80=float(np.mean(o80)), o100=float(np.mean(o100)))

# ---------- outputs ------------------------------------------------------------
t = pd.DataFrame(rows)
t.to_csv(os.path.join(args.out, "downstream_metrics.csv"), index=False)
piv = t.pivot(index="metric", columns="gen", values="value")
pd.set_option("display.width", 180, "display.float_format", lambda v: f"{v:.3f}")
print(piv.to_string())

# CF table (LaTeX)
with open(os.path.join(args.out, "downstream_table.tex"), "w") as f:
    f.write("% auto-generated by downstream_validation.py\n\\begin{tabular}{l" + "c" * len(K_GRID) + "}\n\\toprule\n")
    f.write("CF (avg.\\ profile) & " + " & ".join(f"$K={k}$" for k in K_GRID) + " \\\\\n\\midrule\n")
    for lab in ["Real", "SPARC"] + list(bags):
        f.write(lab + " & " + " & ".join(f"{cf_prof[lab][k]:.3f}" for k in K_GRID) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# CF curve figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
Ks = K_GRID
styles = {"Real": ("k", "o", "-"), "SPARC": ("#1f77b4", "s", "-"),
          "Copula": ("#7f7f7f", "^", "--"), "GMMNet": ("#2ca02c", "v", "--"), "EV-SDG": ("#d62728", "d", "--")}
for lab in ["Real", "SPARC"] + list(bags):
    c, m, ls = styles.get(lab, ("gray", ".", ":"))
    ax.plot(Ks, [cf_prof[lab][k] for k in Ks], ls, marker=m, ms=4, lw=1.4, color=c, label=lab)
ax.set_xlabel("Group size $K$ (users)"); ax.set_ylabel("Coincidence factor")
ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "cf_curve.pdf")); fig.savefig(os.path.join(args.out, "cf_curve.png"))

# daily-peak CDF (Real vs SPARC)
fig2, ax2 = plt.subplots(figsize=(4.2, 3.0), dpi=200)
for lab, c in [("Real", "k"), ("SPARC", "#1f77b4")]:
    x = np.sort(PKS[lab]); y = np.arange(1, len(x) + 1) / len(x)
    ax2.plot(x, y, lw=1.5, color=c, label=lab)
ax2.set_xlabel("Daily fleet peak (kW)"); ax2.set_ylabel("CDF")
ax2.grid(alpha=0.3, lw=0.5); ax2.legend(fontsize=8, frameon=False)
fig2.tight_layout(); fig2.savefig(os.path.join(args.out, "peak_cdf.pdf")); fig2.savefig(os.path.join(args.out, "peak_cdf.png"))

# pocket exceedance bars (Real vs SPARC)
fig3, ax3 = plt.subplots(figsize=(4.2, 3.0), dpi=200)
cats = ["> 60% rating", "> 80% rating", "> 100% rating"]
x = np.arange(3); wdt = 0.35
ax3.bar(x - wdt/2, [POCKET["Real"][k] for k in ("o60", "o80", "o100")], wdt, color="k", alpha=0.8, label="Real")
ax3.bar(x + wdt/2, [POCKET["SPARC"][k] for k in ("o60", "o80", "o100")], wdt, color="#1f77b4", alpha=0.8, label="SPARC")
ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=8)
ax3.set_ylabel(f"% of hours ({K_POCKET} users, {RATING:.0f} kVA)")
ax3.grid(alpha=0.3, lw=0.5, axis="y"); ax3.legend(fontsize=8, frameon=False)
fig3.tight_layout(); fig3.savefig(os.path.join(args.out, "pocket_loading.pdf")); fig3.savefig(os.path.join(args.out, "pocket_loading.png"))
print("wrote downstream_metrics.csv, downstream_table.tex, cf_curve, peak_cdf, pocket_loading")
