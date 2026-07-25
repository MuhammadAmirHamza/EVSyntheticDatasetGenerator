#!/usr/bin/env python3
"""Theme-matched downstream figures (reviewer #3), combined cohort.
Style copied from paper_results/plotting/make_paper_figs.py (Times/STIX serif,
8pt, gray Real staircase + fill, blue dashed SPARC, no top/right spines).

Figures:
  fig_downstream_load.pdf    average weekday load per user (kW), all generators
  fig_downstream_cf.pdf      coincidence factor vs group size, all generators
  fig_downstream_peak.pdf    daily fleet-peak CDF, Real vs SPARC
  fig_downstream_pocket.pdf  10-user pocket: load-duration curve, Real vs SPARC
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.4, "axes.axisbelow": True,
})
REAL_C = "0.45"
C = {"SPARC": "#1f6fb4", "Copula": "#e49444", "GMMNet": "#009e8f", "EV-SDG": "#b0435a"}
LS = {"SPARC": "--", "Copula": ":", "GMMNet": "-.", "EV-SDG": (0, (3, 1, 1, 1))}
MK = {"Real": "o", "SPARC": "s", "Copula": "^", "GMMNet": "v", "EV-SDG": "d"}
COL1 = 3.5
PCLIP = 22.0
RNG = np.random.default_rng(0)

ap = argparse.ArgumentParser()
ap.add_argument("--real"); ap.add_argument("--sparc")
ap.add_argument("--copula"); ap.add_argument("--gmmnet"); ap.add_argument("--evsdg")
ap.add_argument("--metrics", help="downstream_metrics.csv from downstream_validation.py")
ap.add_argument("--out", default=".")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

real = pd.read_csv(a.real); real["start"] = pd.to_datetime(real.abs_start)
real = real[(real.duration_hours > 0) & (real.energy > 0)].rename(columns={"duration_hours": "D", "energy": "E"})
sparc = pd.read_csv(a.sparc)
sparc["start"] = pd.to_datetime(sparc.calendar_date) + pd.to_timedelta(sparc.arrival_hour, unit="h")
sparc = sparc[(sparc.duration_h > 0) & (sparc.energy_kwh > 0)].rename(columns={"duration_h": "D", "energy_kwh": "E"})
bags = {}
for lab, p in [("Copula", a.copula), ("GMMNet", a.gmmnet), ("EV-SDG", a.evsdg)]:
    if p:
        m = pd.read_csv(p).apply(pd.to_numeric, errors="coerce").dropna()
        bags[lab] = m[(m.duration_h > 0) & (m.energy_kwh > 0)]

# ---------------- fig 1: average weekday load per user -------------------------
def prof24(A, D, E):
    P = np.zeros(24)
    for t0, d, e in zip(np.asarray(A) % 24, D, E):
        p = min(e / d, PCLIP); t = float(t0); end = t0 + d
        while t < end - 1e-9:
            hb = np.floor(t) + 1.0; seg = min(hb, end) - t
            P[int(np.floor(t)) % 24] += p * seg; t += seg
    return P

rw = real[real.day_type == "Weekday"]; sw = sparc[sparc.day_type == "Weekday"]
n_users = real.user_id.nunique()
r_days = rw.start.dt.date.nunique(); s_days = pd.to_datetime(sw.start).dt.date.nunique()
R = prof24(rw.start.dt.hour + rw.start.dt.minute / 60, rw.D, rw.E) / r_days / n_users * n_users  # fleet avg per weekday
S = prof24(sw.start.dt.hour + sw.start.dt.minute / 60, sw.D, sw.E) / s_days
R = R  # fleet kW per weekday day
# per-user scale
Ru, Su = R / n_users, S / n_users
rate = len(rw) / r_days   # real weekday sessions per day (fleet)
B24 = {}
for lab, m in bags.items():
    P = prof24(m.arrival_hour, m.duration_h, m.energy_kwh) / len(m)   # kW per session
    B24[lab] = P * rate / n_users                                     # scaled to real session rate, per user

h = np.arange(24) + 0.5
fig, ax = plt.subplots(figsize=(COL1, 2.3))
ax.fill_between(h, Ru, color=REAL_C, alpha=0.35, step="mid", lw=0)
ax.plot(h, Ru, color=REAL_C, lw=1.4, drawstyle="steps-mid", label="Real")
ax.plot(h, Su, color=C["SPARC"], lw=1.3, ls="--", drawstyle="steps-mid", label="SPARC")
for lab in bags:
    ax.plot(h, B24[lab], color=C[lab], lw=1.0, ls=LS[lab], drawstyle="steps-mid", label=lab)
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(bottom=0)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Avg. load per user (kW)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.8)
fig.tight_layout(); fig.savefig(f"{a.out}/fig_downstream_load.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}/fig_downstream_load.png", bbox_inches="tight", dpi=300)

# ---------------- fig 2: CF vs K ------------------------------------------------
met = pd.read_csv(a.metrics)
cf = met[met.metric.str.startswith("CF_prof_K")].copy()
cf["K"] = cf.metric.str.extract(r"K(\d+)").astype(int)
fig, ax = plt.subplots(figsize=(COL1, 2.3))
for lab in ["Real", "SPARC", "Copula", "GMMNet", "EV-SDG"]:
    d = cf[(cf.gen == lab) & cf.value.notna()].sort_values("K")
    if not len(d): continue
    col = REAL_C if lab == "Real" else C[lab]
    ls = "-" if lab == "Real" else LS.get(lab, "--")
    lw = 1.5 if lab in ("Real", "SPARC") else 1.0
    ax.plot(d.K, d.value, linestyle=ls, marker=MK[lab], ms=3.5, lw=lw, color=col, label=lab,
            markerfacecolor=col, markeredgewidth=0)
ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels([2, 5, 10, 20, 50])
ax.set_xlabel("Group size $K$ (users)"); ax.set_ylabel("Coincidence factor")
ax.set_ylim(0.5, 1.0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, ncol=2, handlelength=1.8)
fig.tight_layout(); fig.savefig(f"{a.out}/fig_downstream_cf.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}/fig_downstream_cf.png", bbox_inches="tight", dpi=300)

# ---------------- hourly series for figs 3-4 ------------------------------------
def hourly_series(df):
    epoch = pd.Timestamp("2015-01-01"); out = {}
    for u, g in df.groupby("user_id"):
        h0 = int(np.floor((g.start.min() - epoch).total_seconds() / 3600))
        h1 = int(np.ceil(((g.start + pd.to_timedelta(g.D, unit="h")).max() - epoch).total_seconds() / 3600)) + 1
        L = np.zeros(h1 - h0)
        for s, d, e in zip(g.start, g.D, g.E):
            p = min(e / d, PCLIP); t0h = (s - epoch).total_seconds() / 3600 - h0
            end = t0h + d; hh = int(np.floor(t0h))
            while hh < end:
                lo, hi = max(t0h, hh), min(end, hh + 1)
                if hi > lo and 0 <= hh < len(L): L[hh] += p * (hi - lo)
                hh += 1
        out[u] = (h0, L)
    return out

US_real, US_sparc = hourly_series(real), hourly_series(sparc)
users = sorted(set(US_real) & set(US_sparc))

def fleet(US):
    h0 = min(v[0] for v in US.values()); h1 = max(v[0] + len(v[1]) for v in US.values())
    F = np.zeros(h1 - h0)
    for o, L in US.values(): F[o - h0: o - h0 + len(L)] += L
    return F

# fig 3: daily peak CDF
fig, ax = plt.subplots(figsize=(COL1, 2.3))
for lab, US, col, ls in [("Real", US_real, REAL_C, "-"), ("SPARC", US_sparc, C["SPARC"], "--")]:
    F = fleet({u: US[u] for u in users}); n = (len(F) // 24) * 24
    pk = F[:n].reshape(-1, 24).max(axis=1); pk = np.sort(pk[pk > 0])
    ax.plot(pk, np.arange(1, len(pk) + 1) / len(pk), color=col, ls=ls, lw=1.4, label=lab)
ax.set_xlabel("Daily fleet peak (kW)"); ax.set_ylabel("CDF")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{a.out}/fig_downstream_peak.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}/fig_downstream_peak.png", bbox_inches="tight", dpi=300)

# fig 4: pocket load-duration curve (10 users), mean over 200 bootstrap groups
fig, ax = plt.subplots(figsize=(COL1, 2.3))
for lab, US, col, ls in [("Real", US_real, REAL_C, "-"), ("SPARC", US_sparc, C["SPARC"], "--")]:
    curves = []
    for _ in range(200):
        gs = RNG.choice(users, size=10, replace=False)
        off = [US[u] for u in gs]
        h0 = min(o for o, _ in off); h1 = max(o + len(L) for o, L in off)
        Ssum = np.zeros(h1 - h0)
        for o, L in off: Ssum[o - h0: o - h0 + len(L)] += L
        q = np.quantile(Ssum, np.linspace(0, 1, 200))
        curves.append(q[::-1])       # load-duration: sorted descending
    m = np.mean(curves, axis=0)
    x = np.linspace(0, 100, 200)
    ax.plot(x, m, color=col, ls=ls, lw=1.4, label=lab)
ax.set_xlabel("Fraction of hours exceeded (%)"); ax.set_ylabel("10-user load (kW)")
ax.set_xlim(0, 20)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{a.out}/fig_downstream_pocket.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}/fig_downstream_pocket.png", bbox_inches="tight", dpi=300)
print("wrote fig_downstream_{load,cf,peak,pocket}.{pdf,png}")
