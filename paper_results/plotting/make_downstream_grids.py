#!/usr/bin/env python3
"""Downstream figure GRIDS (2x2 and 1x4), boxed axes, paper theme.
Panels: (a) avg weekday load per user  (b) CF vs K  (c) 10-user load-duration
        (d) grid parameters: LF, UF(50 kVA), CF@10  (Real vs SPARC)
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
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.4, "axes.axisbelow": True,
})
REAL_C = "0.45"
C = {"SPARC": "#1f6fb4", "Copula": "#e49444", "GMMNet": "#009e8f", "EV-SDG": "#b0435a"}
LS = {"SPARC": "--", "Copula": ":", "GMMNet": "-.", "EV-SDG": (0, (3, 1, 1, 1))}
MK = {"Real": "o", "SPARC": "s", "Copula": "^", "GMMNet": "v", "EV-SDG": "d"}
COL1, COL2 = 3.5, 7.16
PCLIP, RATING, KP = 22.0, 50.0, 10
RNG = np.random.default_rng(0)

ap = argparse.ArgumentParser()
for k in ("real", "sparc", "copula", "gmmnet", "evsdg", "metrics"): ap.add_argument("--" + k)
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

def prof24(A, D, E):
    P = np.zeros(24)
    for t0, d, e in zip(np.asarray(A) % 24, D, E):
        p = min(e / d, PCLIP); t = float(t0); end = t0 + d
        while t < end - 1e-9:
            hb = np.floor(t) + 1.0; seg = min(hb, end) - t
            P[int(np.floor(t)) % 24] += p * seg; t += seg
    return P

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

print("hourly series..."); US = {"Real": hourly_series(real), "SPARC": hourly_series(sparc)}
users = sorted(set(US["Real"]) & set(US["SPARC"]))

# panel data ---------------------------------------------------------------
rw = real[real.day_type == "Weekday"]; sw = sparc[sparc.day_type == "Weekday"]
n_users = real.user_id.nunique()
r_days = rw.start.dt.date.nunique(); s_days = sw.start.dt.date.nunique()
Ru = prof24(rw.start.dt.hour + rw.start.dt.minute / 60, rw.D, rw.E) / r_days / n_users
Su = prof24(sw.start.dt.hour + sw.start.dt.minute / 60, sw.D, sw.E) / s_days / n_users
rate = len(rw) / r_days
B24 = {lab: prof24(m.arrival_hour, m.duration_h, m.energy_kwh) / len(m) * rate / n_users
       for lab, m in bags.items()}

met = pd.read_csv(a.metrics)
cf = met[met.metric.str.startswith("CF_prof_K")].copy()
cf["K"] = cf.metric.str.extract(r"K(\d+)").astype(int)

LD, GP = {}, {}
for lab in ("Real", "SPARC"):
    curves, lf, uf = [], [], []
    for _ in range(200):
        gs = RNG.choice(users, size=KP, replace=False)
        off = [US[lab][u] for u in gs]
        h0 = min(o for o, _ in off); h1 = max(o + len(L) for o, L in off)
        S = np.zeros(h1 - h0)
        for o, L in off: S[o - h0: o - h0 + len(L)] += L
        q = np.quantile(S, np.linspace(0, 1, 200)); curves.append(q[::-1])
        pk = S.max()
        lf.append(S.mean() / pk if pk > 0 else np.nan)
        uf.append(pk / RATING)
    LD[lab] = np.mean(curves, axis=0)
    cf10 = float(cf[(cf.gen == lab) & (cf.K == 10)].value.iloc[0])
    GP[lab] = dict(LF=float(np.nanmean(lf)), UF=float(np.nanmean(uf)), CF=cf10)
print("grid params:", GP)

# panels --------------------------------------------------------------------
def panel_load(ax):
    h = np.arange(24) + 0.5
    ax.fill_between(h, Ru, color=REAL_C, alpha=0.35, step="mid", lw=0)
    ax.plot(h, Ru, color=REAL_C, lw=1.4, drawstyle="steps-mid", label="Real")
    ax.plot(h, Su, color=C["SPARC"], lw=1.2, ls="--", drawstyle="steps-mid", label="SPARC")
    for lab in bags:
        ax.plot(h, B24[lab], color=C[lab], lw=0.9, ls=LS[lab], drawstyle="steps-mid", label=lab)
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Avg. load per user (kW)")

def panel_cf(ax):
    for lab in ["Real", "SPARC", "Copula", "GMMNet", "EV-SDG"]:
        d = cf[(cf.gen == lab) & cf.value.notna()].sort_values("K")
        if not len(d): continue
        col = REAL_C if lab == "Real" else C[lab]
        ax.plot(d.K, d.value, linestyle="-" if lab == "Real" else LS.get(lab, "--"),
                marker=MK[lab], ms=3, lw=1.4 if lab in ("Real", "SPARC") else 0.9,
                color=col, label=lab, markerfacecolor=col, markeredgewidth=0)
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels([2, 5, 10, 20, 50])
    ax.set_xlabel("Group size $K$ (users)"); ax.set_ylabel("Coincidence factor")
    ax.set_ylim(0.5, 1.0)

def panel_ld(ax):
    x = np.linspace(0, 100, 200)
    ax.plot(x, LD["Real"], color=REAL_C, lw=1.4, label="Real")
    ax.plot(x, LD["SPARC"], color=C["SPARC"], lw=1.2, ls="--", label="SPARC")
    ax.set_xlim(0, 20); ax.set_ylim(bottom=0)
    ax.set_xlabel("Hours exceeded (%)"); ax.set_ylabel(f"{KP}-user load (kW)")

def panel_gp(ax):
    labels = ["Load factor", f"Utilization\n({RATING:.0f} kVA)", "CF ($K$=10)"]
    xs = np.arange(3); w = 0.36
    rv = [GP["Real"][k] for k in ("LF", "UF", "CF")]
    sv = [GP["SPARC"][k] for k in ("LF", "UF", "CF")]
    ax.bar(xs - w / 2, rv, w, color=REAL_C, alpha=0.75, label="Real")
    ax.bar(xs + w / 2, sv, w, color=C["SPARC"], alpha=0.9, label="SPARC")
    for x0, v in zip(xs - w / 2, rv): ax.text(x0, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
    for x0, v in zip(xs + w / 2, sv): ax.text(x0, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Value"); ax.set_ylim(0, max(rv + sv) * 1.25)
    ax.grid(axis="x", visible=False)

def panel_cfbar(ax):
    gens = ["Real", "SPARC", "Copula", "GMMNet", "EV-SDG"]
    vals, cols = [], []
    for g in gens:
        d = cf[(cf.gen == g) & (cf.K == 10)].value
        vals.append(float(d.iloc[0]) if len(d) and pd.notna(d.iloc[0]) else np.nan)
        cols.append(REAL_C if g == "Real" else C[g])
    xs = np.arange(len(gens))
    ax.bar(xs, vals, 0.62, color=cols, alpha=0.9)
    for x0, v in zip(xs, vals):
        if np.isfinite(v): ax.text(x0, v + 0.012, f"{v:.2f}", ha="center", fontsize=6)
    ax.axhline(vals[0], color=REAL_C, lw=0.8, ls=":", zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(gens, rotation=20, ha="right")
    ax.set_ylabel("CF ($K$=10)"); ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)

VARIANTS = {"": [("(a)", panel_load), ("(b)", panel_cf), ("(c)", panel_ld), ("(d)", panel_gp)],
            "_cfbar": [("(a)", panel_load), ("(b)", panel_cf), ("(c)", panel_ld), ("(d)", panel_cfbar)]}

for suf, PANELS in VARIANTS.items():
    for name, (nr, nc, size) in {f"fig_downstream_2x2{suf}": (2, 2, (COL1 * 2 * 0.52 + 1.4, 4.4)),
                                 f"fig_downstream_1x4{suf}": (1, 4, (COL2, 1.95))}.items():
        fig, axes = plt.subplots(nr, nc, figsize=size)
        for (tag, fn), ax in zip(PANELS, np.ravel(axes)):
            fn(ax)
            for sp in ax.spines.values(): sp.set_visible(True)     # boxed axes
            ax.set_title(tag, loc="left", fontsize=8, fontweight="bold", pad=3)
        handles = [Line2D([0], [0], color=REAL_C, lw=1.6, ls="-", marker="o", ms=3.5, label="Real"),
                   Line2D([0], [0], color=C["SPARC"], lw=1.4, ls="--", marker="s", ms=3.5, label="SPARC"),
                   Line2D([0], [0], color=C["Copula"], lw=1.1, ls=":", marker="^", ms=3.5, label="Copula"),
                   Line2D([0], [0], color=C["GMMNet"], lw=1.1, ls="-.", marker="v", ms=3.5, label="GMMNet"),
                   Line2D([0], [0], color=C["EV-SDG"], lw=1.1, ls=(0, (3, 1, 1, 1)), marker="d", ms=3.5, label="EV-SDG")]
        yanch = 1.10 if nr == 1 else 1.045
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, yanch), handlelength=2.2, columnspacing=1.2)
        fig.tight_layout(w_pad=1.0, h_pad=1.2)
        fig.savefig(f"{a.out}/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"{a.out}/{name}.png", bbox_inches="tight", dpi=300)
        print("wrote", name)


# ---------------- 1x5 variant: prepend sessions-per-active-day panel ------------
ART = "/mnt/user-data/uploads/SDG/SDG/paper_results/artifacts"
COHORTS = [("Norway", "norway_sorensen_N100_b12_s01"), ("GIST-Res", "korea_gist_N030_b24_s01"),
           ("GIST-Comm", "kim_commercial_N025_b24_s01"), ("ACN", "acn_caltech_N015_b12_s01"),
           ("Combined", "combined_all_N200_b24_s01")]
SES = []
for labc, d in COHORTS:
    rr = pd.read_csv(f"{ART}/{d}/raw_clean.csv"); ss = pd.read_csv(f"{ART}/{d}/sim_sessions.csv")
    rr["date"] = pd.to_datetime(rr.abs_start, errors="coerce").dt.date
    rc = rr.groupby(["user_id", "date"]).size(); sc = ss.groupby(["user_id", "calendar_date"]).size()
    SES.append((labc, rc.mean(), rc.std() / np.sqrt(len(rc)), sc.mean(), sc.std() / np.sqrt(len(sc))))

def panel_sessions(ax):
    labs = [t[0] for t in SES]
    rmu = [t[1] for t in SES]; rse = [t[2] for t in SES]
    smu = [t[3] for t in SES]; sse = [t[4] for t in SES]
    xs = np.arange(len(labs)); w = 0.38
    ax.bar(xs - w / 2, rmu, w, yerr=rse, color=REAL_C, alpha=0.75, error_kw=dict(lw=0.7, capsize=1.5))
    ax.bar(xs + w / 2, smu, w, yerr=sse, color=C["SPARC"], alpha=0.9, error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Sessions / active day")
    ax.set_ylim(0, max(np.add(rmu, rse).max(), np.add(smu, sse).max()) * 1.25)
    ax.grid(axis="x", visible=False)

P5 = [("(a)", panel_sessions), ("(b)", panel_load), ("(c)", panel_cf), ("(d)", panel_ld), ("(e)", panel_cfbar)]
fig, axes = plt.subplots(1, 5, figsize=(COL2, 1.9))
for (tag, fn), ax in zip(P5, axes):
    fn(ax)
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.set_title(tag, loc="left", fontsize=8, fontweight="bold", pad=3)
axes[1].set_ylim(bottom=0.02)                      # floor the load axis
axes[4].set_xticklabels(["Real", "SPARC", "Copula", "GMMNet", "EV-SDG"],
                        rotation=90, fontsize=6)
axes[4].tick_params(axis="x", length=0)
handles = [Line2D([0], [0], color=REAL_C, lw=1.6, ls="-", marker="o", ms=3.5, label="Real"),
           Line2D([0], [0], color=C["SPARC"], lw=1.4, ls="--", marker="s", ms=3.5, label="SPARC"),
           Line2D([0], [0], color=C["Copula"], lw=1.1, ls=":", marker="^", ms=3.5, label="Copula"),
           Line2D([0], [0], color=C["GMMNet"], lw=1.1, ls="-.", marker="v", ms=3.5, label="GMMNet"),
           Line2D([0], [0], color=C["EV-SDG"], lw=1.1, ls=(0, (3, 1, 1, 1)), marker="d", ms=3.5, label="EV-SDG")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.13), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(w_pad=0.25)
fig.savefig(f"{a.out}/fig_downstream_1x5_cfbar.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}/fig_downstream_1x5_cfbar.png", bbox_inches="tight", dpi=300)
print("wrote fig_downstream_1x5_cfbar")
