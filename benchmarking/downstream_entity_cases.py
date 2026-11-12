#!/usr/bin/env python3
"""Per-entity downstream cases (combined pool): does a generator reproduce what
each entity does, not only what the population does?

For every entity u the same quantity is computed on the real sessions and on the
synthetic sessions of that entity, then compared across the entities:
    rho  = Spearman rank correlation between real and synthetic values
    err  = median over entities of |synthetic - real| / real   (percent)

Cases
  occupancy      share of the observation window in which the entity is connected
  night_avail    share of the hours 22:00-06:00 in which the entity is connected
  peak_share     share of the entity's energy delivered in 17:00-21:00
  weekly_energy  mean energy per week (kWh)
  flexibility    mean idle share of a session, (D - E/P_u)/D, where P_u is the
                 entity's rated power (95th percentile of E/D in the real data,
                 clipped to [1.4, 22] kW)
  overlap        sessions that start before an earlier session of the same
                 entity has ended (percent of consecutive pairs)

SPARC keeps entity identities and its own calendar.  Pooled baselines carry
neither, so each real session keeps its entity and its date and receives a
pooled (A, D, E) draw: the baselines are given the real session dates and
counts (an advantage for them).  Results are averaged over --reps placements.

    python benchmarking/downstream_entity_cases.py \
        --real  paper_results/artifacts/combined_all_N200_b24_s01/raw_clean.csv \
        --sparc paper_results/artifacts/combined_all_N200_b24_s01/sim_sessions.csv \
        --copula benchmarking/copula/results/einolander_combined.csv \
        --gmmnet benchmarking/gmmnet/results/gmmnet_combined.csv \
        --evsdg  benchmarking/evsdg/results/EVSDG_combined_all.csv \
        --out benchmarking/downstream
Outputs: downstream_entity_cases.csv   rho / err per generator and case
         downstream_entity_table.tex   the same as a LaTeX table
         downstream_entity_real.csv    real values: median and quartiles, per source
         downstream_entity_values.csv  per-entity values, real and every generator
         downstream_load_profiles.csv  weekday load per entity by hour, per cohort and generator
         entity_cases_scatter.pdf/.png       real vs synthetic, all generators (two-column figure)
         entity_cases_scatter_1col.pdf/.png  SPARC and one baseline (one-column figure)
         fig_summary.pdf/.png          rho and error for every case and generator
         fig_overlap.pdf/.png          overlapping sessions per generator
         fig_peak_share.pdf/.png       distribution of the peak-window share over entities
         fig_weekly_energy.pdf/.png    distribution of the per-entity weekly-energy error
         fig_flexibility.pdf/.png      flexibility index per cohort
         fig_availability.pdf/.png     share of entities connected by hour of day, per cohort
         fig_load_cohorts.pdf/.png     average weekday load per entity by hour of day, per cohort and combined
         panels/load_<cohort>, panels/availability_<cohort> (.pdf and 600-dpi .png): single panels of
                                       equal size; panels/legend: the legend alone
         fig_load_week.pdf/.png        the same for all entities: weekday and weekend
         fig_timeline.pdf/.png         two weeks of one entity: sessions and overlaps per generator
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True); ap.add_argument("--sparc", required=True)
ap.add_argument("--copula"); ap.add_argument("--gmmnet"); ap.add_argument("--evsdg")
ap.add_argument("--out", default="."); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeline-entity", default=None, help="entity drawn in fig_timeline (default: 90th percentile of session rate)")
ap.add_argument("--fig-cases", default="occupancy,night_avail",
                help="cases drawn in the scatter figures (comma separated)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
EPOCH = pd.Timestamp("2015-01-05")        # a Monday, midnight
PCLIP = 22.0                              # kW
TOL = np.timedelta64(60, "s")             # back-to-back sessions are not overlaps

# ---------- load ---------------------------------------------------------------
real = pd.read_csv(args.real)
real["start"] = pd.to_datetime(real["abs_start"])
real = real.rename(columns={"duration_hours": "D", "energy": "E"})
real = real[(real.D > 0) & (real.E > 0)].copy()
real["end"] = real.start + pd.to_timedelta(real.D, unit="h")

sparc = pd.read_csv(args.sparc)
sparc["start"] = pd.to_datetime(sparc["calendar_date"]) + pd.to_timedelta(sparc["arrival_hour"], unit="h")
sparc = sparc.rename(columns={"duration_h": "D", "energy_kwh": "E"})
sparc = sparc[(sparc.D > 0) & (sparc.E > 0)].copy()
sparc["end"] = sparc.start + pd.to_timedelta(sparc.D, unit="h")

users = sorted(set(real.user_id) & set(sparc.user_id))
real = real[real.user_id.isin(users)]; sparc = sparc[sparc.user_id.isin(users)]

bags = {}
for lab, p in [("Copula", args.copula), ("GMMNet", args.gmmnet), ("EV-SDG", args.evsdg)]:
    if not p: continue
    m = pd.read_csv(p)[["arrival_hour", "duration_h", "energy_kwh"]].apply(pd.to_numeric, errors="coerce").dropna()
    m = m[(m.duration_h > 0) & (m.energy_kwh > 0)].rename(columns={"duration_h": "D", "energy_kwh": "E"})
    bags[lab] = m.reset_index(drop=True)

def place(bag):
    """pooled sessions on the real calendar: entity and date from the real session"""
    idx = rng.choice(len(bag), size=len(real), replace=len(bag) < len(real))
    d = pd.DataFrame({"user_id": real.user_id.values})
    d["start"] = real.start.dt.normalize().values + pd.to_timedelta(bag.arrival_hour.values[idx] % 24, unit="h")
    d["D"] = bag.D.values[idx]; d["E"] = bag.E.values[idx]
    d["end"] = d.start + pd.to_timedelta(d.D, unit="h")
    return d

# ---------- metrics ------------------------------------------------------------
span = {u: (g.start.min(), g.end.max()) for u, g in real.groupby("user_id")}
P_u = real.assign(p=real.E / real.D).groupby("user_id").p.quantile(0.95).clip(1.4, 22.0).to_dict()
hidx = lambda t: int((t - EPOCH).total_seconds() // 3600)
H = hidx(max(real.end.max(), sparc.end.max())) + 24 * 400

def overlap_rate(df):
    n = o = 0
    for _, g in df.groupby("user_id"):
        g = g.sort_values("start"); s = g.start.values; e = g.end.values
        prev = np.maximum.accumulate(e)[:-1] - TOL
        n += len(s) - 1; o += int((s[1:] < prev).sum())
    return 100.0 * o / max(n, 1)

def per_entity(df, prof=None):
    """per-entity case values; if `prof` is a dict it receives u -> 24 hourly availability shares"""
    rows = []
    for u, g in df.groupby("user_id"):
        C = np.zeros(H); L = np.zeros(H)
        a = ((g.start - EPOCH).dt.total_seconds() / 3600).values
        for ai, di, ei in zip(a, g.D.values, g.E.values):
            p = min(ei / di, PCLIP); end = ai + di; h = int(np.floor(ai))
            while h < end:
                lo, hi = max(ai, h), min(end, h + 1)
                if hi > lo and 0 <= h < H: C[h] += hi - lo; L[h] += p * (hi - lo)
                h += 1
        C = np.minimum(C, 1.0)
        h0, h1 = hidx(span[u][0]), hidx(span[u][1]) + 1
        c, l = C[h0:h1], L[h0:h1]; hod = np.arange(h0, h1) % 24
        wk = l[: (len(l) // 168) * 168].reshape(-1, 168).sum(1) if len(l) >= 336 else np.array([np.nan])
        if prof is not None:
            prof[u] = np.bincount(hod, weights=c, minlength=24) / np.maximum(np.bincount(hod, minlength=24), 1)
            we = ((np.arange(h0, h1) // 24) % 7) >= 5          # EPOCH is a Monday
            LOAD.setdefault(id(prof), {})[u] = tuple(
                np.bincount(hod[m], weights=l[m], minlength=24) / np.maximum(np.bincount(hod[m], minlength=24), 1)
                for m in (~we, we))
        lax = np.clip(g.D.values - g.E.values / P_u[u], 0, None) / g.D.values
        rows.append(dict(user_id=u,
                         occupancy=c.mean(),
                         night_avail=c[(hod >= 22) | (hod < 6)].mean(),
                         peak_share=l[(hod >= 17) & (hod < 21)].sum() / max(l.sum(), 1e-9),
                         weekly_energy=np.nanmean(wk),
                         flexibility=lax.mean()))
    return pd.DataFrame(rows).set_index("user_id").reindex(users)

CASES = ["occupancy", "night_avail", "peak_share", "weekly_energy", "flexibility"]
def compare(pe_real, pe):
    out = {}
    for c in CASES:
        a, b = pe_real[c], pe[c]; m = a.notna() & b.notna() & (a.abs() > 1e-9)
        out[c + "_rho"] = spearmanr(a[m], b[m])[0]
        out[c + "_err"] = 100 * np.median(np.abs(a[m] - b[m]) / a[m].abs())
    return out

PROF = {"Real": {}, "SPARC": {}}          # hour-of-day availability per entity
LOAD = {}                                 # id(PROF[g]) -> u -> (weekday, weekend) load by hour, kW
SESS = {"Real": real, "SPARC": sparc}     # sessions drawn in the timeline figure
print("real ..."); pe_real = per_entity(real, PROF["Real"])
res = {"Real": dict(overlap=overlap_rate(real))}
PE = {}                                   # per-entity values shown in the figures
print("SPARC ..."); PE["SPARC"] = per_entity(sparc, PROF["SPARC"])
res["SPARC"] = dict(overlap=overlap_rate(sparc), **compare(pe_real, PE["SPARC"]))
for lab, bag in bags.items():
    print(lab, "...")
    reps = []
    for r in range(args.reps):
        d = place(bag)
        if r == 0:
            PROF[lab] = {}; pe = per_entity(d, PROF[lab]); PE[lab] = pe   # first placement is the one drawn
            SESS[lab] = d
        else:
            pe = per_entity(d)
        reps.append(dict(overlap=overlap_rate(d), **compare(pe_real, pe)))
    res[lab] = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}

t = pd.DataFrame(res).T
t.index.name = "generator"
t.to_csv(os.path.join(args.out, "downstream_entity_cases.csv"))
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.3f}")
print(t.T.to_string())

names = {"occupancy": "Occupancy", "night_avail": "Night availability",
         "peak_share": "Peak-window energy share", "weekly_energy": "Weekly energy",
         "flexibility": "Flexibility index"}
gens = [g for g in t.index if g != "Real"]
with open(os.path.join(args.out, "downstream_entity_table.tex"), "w") as f:
    f.write("% auto-generated by downstream_entity_cases.py\n")
    f.write("\\begin{tabular}{l" + "cc" * len(gens) + "}\n\\toprule\n")
    f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{g}}}" for g in gens) + " \\\\\n")
    f.write("Case & " + " & ".join("$\\rho$ & err.\\,(\\%)" for _ in gens) + " \\\\\n\\midrule\n")
    for c in CASES:
        f.write(names[c] + " & " + " & ".join(f"{t.loc[g, c + '_rho']:.2f} & {t.loc[g, c + '_err']:.0f}" for g in gens) + " \\\\\n")
    f.write("\\midrule\nOverlapping sessions (\\%) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t.loc[g, 'overlap']:.1f}}}" for g in gens) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("real overlap rate: %.2f %%" % res["Real"]["overlap"])

# ---------- real values --------------------------------------------------------
src = pd.Series([u.split("|")[0] if "|" in u else "all" for u in users], index=users, name="source")
rows = []
for grp, idx in [("all", users)] + [(k, list(v.index)) for k, v in src.groupby(src)]:
    for c in CASES:
        v = pe_real.loc[idx, c].dropna()
        rows.append(dict(source=grp, case=c, n_entities=len(v), q25=v.quantile(.25), median=v.median(),
                         q75=v.quantile(.75), mean=v.mean(),
                         **{g + "_median": PE[g].loc[idx, c].median() for g in PE}))
real_tab = pd.DataFrame(rows)
real_tab.to_csv(os.path.join(args.out, "downstream_entity_real.csv"), index=False)
print("\nreal values (median [q25, q75]) and synthetic medians")
print(real_tab.to_string(index=False))

vals = pd.concat([pe_real.add_prefix("real_")] + [PE[g].add_prefix(g + "_") for g in PE], axis=1)
vals.insert(0, "source", src)
vals.to_csv(os.path.join(args.out, "downstream_entity_values.csv"))

# ---------- figures: real vs synthetic, one panel per generator ------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# paper theme: same fonts, colours, line styles and markers as make_downstream_grids.py
plt.rcParams.update({
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3,
})
INK, MUTED, GRID = "black", "black", "0.9"
REAL_C = "0.45"
COL = {"Real": REAL_C, "SPARC": "#1f6fb4", "Copula": "#e49444", "GMMNet": "#009e8f", "EV-SDG": "#b0435a"}
LSTY = {"Real": "-", "SPARC": "--", "Copula": ":", "GMMNet": "-.", "EV-SDG": (0, (3, 1, 1, 1))}
MARK = {"Real": "o", "SPARC": "s", "Copula": "^", "GMMNet": "v", "EV-SDG": "d"}
AXLAB = {"occupancy": "occupancy", "night_avail": "night availability",
         "peak_share": "peak-window share", "weekly_energy": "weekly energy (kWh)",
         "flexibility": "flexibility index"}

def scatter_fig(gens, cases, width, fname):
    fig, axes = plt.subplots(len(cases), len(gens), figsize=(width, 0.92 * width * len(cases) / len(gens) + 0.25),
                             squeeze=False, sharex="row", sharey="row")
    for i, c in enumerate(cases):
        hi = 1.05 * np.nanmax([pe_real[c].max()] + [PE[g][c].max() for g in gens])
        for j, g in enumerate(gens):
            ax = axes[i, j]
            ax.plot([0, hi], [0, hi], color="0.3", lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.scatter(pe_real[c], PE[g][c], s=9, color=COL[g], marker=MARK[g], edgecolor="white",
                       linewidth=0.3, zorder=2)
            ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
            ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
            ax.text(0.05, 0.93, f"$\\rho$ = {res[g][c + '_rho']:.2f}", transform=ax.transAxes,
                    ha="left", va="top", color=INK, fontsize=7)
            if i == 0: ax.set_title(g, fontsize=8, fontweight="bold", color=INK, pad=3)
            if j == 0: ax.set_ylabel("Synthetic " + AXLAB[c], color=INK)
            ax.set_xlabel("Real " + AXLAB[c], color=INK)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, fname + "." + ext), dpi=300)
    plt.close(fig)

fig_cases = [c for c in args.fig_cases.split(",") if c in CASES]
gens_all = [g for g in ["SPARC", "Copula", "GMMNet", "EV-SDG"] if g in PE]
scatter_fig(gens_all, fig_cases, 7.16, "entity_cases_scatter")
if len(gens_all) > 1:
    scatter_fig(gens_all[:2], fig_cases, 3.5, "entity_cases_scatter_1col")
print("figures ->", os.path.join(args.out, "entity_cases_scatter.pdf"))

# ---------- further figures: one form per case ----------------------------------
from matplotlib.lines import Line2D
STYLE = {g: dict(color=COL[g], ls=LSTY[g], lw=1.4 if g == "Real" else 1.2 if g == "SPARC" else 1.0, marker=MARK[g])
         for g in COL}
COHORT = {"norway": "Norway", "gist": "GIST-Res", "kim": "GIST-Comm", "acn": "ACN", "all": "All"}
cohorts = [k for k in ["norway", "gist", "kim", "acn", "all"] if (src == k).any()]
NAME = {"occupancy": "Occupancy", "night_avail": "Night availability", "peak_share": "Peak-window share",
        "weekly_energy": "Weekly energy", "flexibility": "Flexibility index"}

def style_ax(ax, grid="both"):
    ax.grid(color=GRID, lw=0.4, axis=grid); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(True)          # boxed axes, as in the paper figures

def save(fig, fname):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, fname + "." + ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def legend_handles(gens, line=True):
    return [Line2D([0], [0], color=STYLE[g]["color"], ls=STYLE[g]["ls"] if line else "none",
                   lw=STYLE[g]["lw"] + 0.2, marker=STYLE[g]["marker"], ms=3.5, markeredgewidth=0, label=g)
            for g in gens]

# (1) summary: rho and median error, every case and generator
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
ypos = np.arange(len(CASES))[::-1]
for ax, key, lab in [(axes[0], "_rho", "Rank correlation $\\rho$"), (axes[1], "_err", "Median error (%)")]:
    for y, c in zip(ypos, CASES):
        v = [res[g][c + key] for g in gens_all]
        ax.plot([min(v), max(v)], [y, y], color="0.8", lw=1.0, zorder=1)
    for g in gens_all:
        ax.scatter([res[g][c + key] for c in CASES], ypos, s=22 if g == "SPARC" else 14, color=STYLE[g]["color"],
                   marker=STYLE[g]["marker"], edgecolor="white", linewidth=0.4, zorder=3 if g == "SPARC" else 2)
    style_ax(ax, "x"); ax.set_xlabel(lab, color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels([NAME[c] for c in CASES], color=INK, fontsize=6.5)
    ax.set_ylim(-0.6, len(CASES) - 0.4)
axes[0].axvline(0, color="0.3", lw=0.5); axes[0].set_xlim(-0.45, 1.05); axes[1].set_xlim(0, None)
fig.legend(handles=legend_handles(gens_all, line=False), loc="upper center", ncol=len(gens_all), frameon=False,
           fontsize=6.5, handletextpad=0.1, columnspacing=0.9, bbox_to_anchor=(0.56, 1.13))
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "fig_summary")

# (2) overlapping sessions per generator
order = ["Real"] + gens_all
fig, ax = plt.subplots(figsize=(3.5, 1.35))
vals_ov = [res[g]["overlap"] for g in order]; yy = np.arange(len(order))[::-1]
ax.barh(yy, vals_ov, height=0.55, color=[STYLE[g]["color"] for g in order])
for y, v in zip(yy, vals_ov):
    ax.text(v + 0.5, y, f"{v:.1f}", va="center", ha="left", fontsize=6.5, color=INK)
style_ax(ax, "x"); ax.set_yticks(yy); ax.set_yticklabels(order, color=INK, fontsize=6.5)
ax.set_xlabel("Overlapping sessions (%)", color=INK); ax.set_xlim(0, max(vals_ov) * 1.15 + 1)
save(fig, "fig_overlap")

# (3) peak-window share: distribution over the entities
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for g in order:
    v = np.sort((pe_real if g == "Real" else PE[g])["peak_share"].dropna().values)
    ax.step(np.r_[0, v, 1], np.r_[0, np.arange(1, len(v) + 1) / len(v), 1], where="post",
            color=STYLE[g]["color"], ls=STYLE[g]["ls"], lw=STYLE[g]["lw"], zorder=3 if g in ("Real", "SPARC") else 2)
style_ax(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("Share of energy delivered in 17:00-21:00", color=INK); ax.set_ylabel("Share of entities", color=INK)
ax.legend(handles=legend_handles(order), loc="lower right", frameon=False, fontsize=6.5, handlelength=2.6)
save(fig, "fig_peak_share")

# (4) weekly energy: distribution of the per-entity error
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for g in gens_all:
    a, b = pe_real["weekly_energy"], PE[g]["weekly_energy"]; m = a.notna() & b.notna()
    e = np.sort(100 * np.abs(b[m] - a[m]) / a[m])
    ax.step(np.r_[0, e, 1e4], np.r_[0, np.arange(1, len(e) + 1) / len(e), 1], where="post",
            color=STYLE[g]["color"], ls=STYLE[g]["ls"], lw=STYLE[g]["lw"], zorder=3 if g == "SPARC" else 2)
style_ax(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 1.01)
ax.set_xlabel("Error in weekly energy of an entity (%)", color=INK); ax.set_ylabel("Share of entities", color=INK)
ax.legend(handles=legend_handles(gens_all), loc="lower right", frameon=False, fontsize=6.5, handlelength=2.6)
save(fig, "fig_weekly_energy")

# (5) flexibility index per cohort
fig, axes = plt.subplots(1, len(cohorts), figsize=(7.16, 1.45), sharex=True, sharey=True, squeeze=False)
for ax, k in zip(axes[0], cohorts):
    idx = src.index[src == k]
    data = [(pe_real if g == "Real" else PE[g]).loc[idx, "flexibility"].dropna().values for g in order]
    kw = dict(positions=yy, widths=0.55, patch_artist=True, showfliers=False,
              medianprops=dict(color="white", lw=1.0), whiskerprops=dict(color="0.3", lw=0.6),
              capprops=dict(color="0.3", lw=0.6), boxprops=dict(lw=0))
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)     # matplotlib >= 3.10
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kw)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(STYLE[g]["color"])
    style_ax(ax, "x"); ax.set_xlim(0, 1); ax.set_title(COHORT[k], fontsize=8, fontweight="bold", color=INK, pad=3)
    ax.set_yticks(yy); ax.set_yticklabels(order, color=INK, fontsize=6.5)
    ax.set_xlabel("Flexibility index", color=INK)
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "fig_flexibility")

# (6) availability by hour of day, per cohort
fig, axes = plt.subplots(1, len(cohorts), figsize=(7.16, 1.6), sharex=True, squeeze=False)
for ax, k in zip(axes[0], cohorts):
    idx = src.index[src == k]
    ax.axvspan(0, 6, color="0.94", lw=0); ax.axvspan(22, 24, color="0.94", lw=0)
    top = 0
    for g in order:
        y = np.mean([PROF[g][u] for u in idx if u in PROF[g]], axis=0); top = max(top, y.max())
        ax.plot(np.arange(24) + 0.5, y, color=STYLE[g]["color"], ls=STYLE[g]["ls"], lw=STYLE[g]["lw"],
                zorder=3 if g in ("Real", "SPARC") else 2)
    style_ax(ax); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(0, top * 1.12)
    ax.set_title(COHORT[k], fontsize=8, fontweight="bold", color=INK, pad=3); ax.set_xlabel("Hour of day", color=INK)
axes[0, 0].set_ylabel("Share of entities connected", color=INK)
fig.legend(handles=legend_handles(order), loc="upper center", ncol=len(order), frameon=False, fontsize=6.5,
           handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(pad=0.3, w_pad=0.9)
save(fig, "fig_availability")
print("figures ->", args.out)

# (7) average load per entity by hour of day
def load_curve(g, idx, k):
    d = LOAD[id(PROF[g])]
    return np.mean([d[u][k] for u in idx if u in d], axis=0)

def draw_load(ax, idx, k):
    h = np.arange(24) + 0.5; top = 0
    r = load_curve("Real", idx, k)
    ax.fill_between(h, r, color=REAL_C, alpha=0.35, lw=0, zorder=1)
    for g in order:
        y = load_curve(g, idx, k); top = max(top, y.max())
        ax.plot(h, y, color=STYLE[g]["color"], ls=STYLE[g]["ls"], lw=STYLE[g]["lw"],
                zorder=3 if g in ("Real", "SPARC") else 2)
    style_ax(ax); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("Hour of day", color=INK)

# one panel per cohort, plus the combined pool
load_groups = [(COHORT[k], src.index[src == k]) for k in cohorts if k != "all"]
if len(load_groups) > 1: load_groups.append(("Combined", pd.Index(users)))
fig, axes = plt.subplots(1, len(load_groups), figsize=(7.16, 1.55), sharex=True, squeeze=False)
for ax, (name, idx) in zip(axes[0], load_groups):
    draw_load(ax, idx, 0)
    ax.set_title(name, fontsize=8, fontweight="bold", color=INK, pad=3)
axes[0, 0].set_ylabel("Weekday load per entity (kW)", color=INK)
fig.legend(handles=legend_handles(order), loc="upper center", ncol=len(order), frameon=False, fontsize=6.5,
           handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(pad=0.3, w_pad=0.5)
save(fig, "fig_load_cohorts")

# load curves as a table (read by paper_results/plotting/make_downstream_grids.py)
pd.DataFrame([dict(group=name, n_entities=len(idx), generator=g, hour=h, weekday_kW=v)
              for name, idx in load_groups for g in order
              for h, v in enumerate(load_curve(g, idx, 0))]).to_csv(
    os.path.join(args.out, "downstream_load_profiles.csv"), index=False)

# the same panels as separate files of identical size (for a grid assembled in LaTeX), and the legend alone
pdir = os.path.join(args.out, "panels"); os.makedirs(pdir, exist_ok=True)
for name, idx in load_groups:
    fig, ax = plt.subplots(figsize=(1.43, 1.6))
    draw_load(ax, idx, 0)
    ax.set_title(name, fontsize=8, fontweight="bold", color=INK, pad=3)
    ax.set_ylabel("Weekday load (kW)", color=INK)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.25, top=0.87)      # fixed margins: panels align
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(pdir, "load_" + name.lower().replace("-", "_") + "." + ext), dpi=600)
    plt.close(fig)
    # availability by hour of day, same layout
    fig, ax = plt.subplots(figsize=(1.43, 1.6))
    ax.axvspan(0, 6, color="0.94", lw=0); ax.axvspan(22, 24, color="0.94", lw=0); top = 0
    for g in order:
        y = np.mean([PROF[g][u] for u in idx if u in PROF[g]], axis=0); top = max(top, y.max())
        ax.plot(np.arange(24) + 0.5, y, color=STYLE[g]["color"], ls=STYLE[g]["ls"], lw=STYLE[g]["lw"],
                zorder=3 if g in ("Real", "SPARC") else 2)
    style_ax(ax); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("Hour of day", color=INK); ax.set_ylabel("Share connected", color=INK)
    ax.set_title(name, fontsize=8, fontweight="bold", color=INK, pad=3)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.25, top=0.87)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(pdir, "availability_" + name.lower().replace("-", "_") + "." + ext), dpi=600)
    plt.close(fig)
fig = plt.figure(figsize=(7.16, 0.22))
fig.legend(handles=legend_handles(order), loc="center", ncol=len(order), frameon=False, fontsize=6.5,
           handlelength=2.6, columnspacing=1.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(pdir, "legend." + ext), dpi=600)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7), sharey=True)
for ax, k, ttl in zip(axes, (0, 1), ("Weekday", "Weekend")):
    draw_load(ax, users, k); ax.set_title(ttl, fontsize=8, fontweight="bold", color=INK, pad=3)
top = 1.12 * max(load_curve(g, users, k).max() for g in order for k in (0, 1)); axes[0].set_ylim(0, top)
axes[0].set_ylabel("Load per entity (kW)", color=INK)
fig.legend(handles=legend_handles(order), loc="upper center", ncol=len(order), frameon=False, fontsize=6,
           handlelength=2.0, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.54, 1.1))
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "fig_load_week")

# (8) timeline of one entity: sessions and overlaps
from matplotlib.patches import Rectangle
RED = "black"                             # overlap highlight (EV-SDG already uses a red)
# default entity: a frequent one (90th percentile of sessions per day over its window)
rate = real.groupby("user_id").apply(lambda g: len(g) / max((g.start.max() - g.start.min()).days, 1))
ent = args.timeline_entity or rate.index[np.argsort(np.abs(rate.values - rate.quantile(0.9)))[0]]
g0 = real[real.user_id == ent]
t_mid = g0.start.min() + (g0.start.max() - g0.start.min()) / 2
w0 = (t_mid - pd.Timedelta(days=int(t_mid.dayofweek))).normalize()     # Monday of the middle week
w1 = w0 + pd.Timedelta(days=14)
fig, ax = plt.subplots(figsize=(7.16, 1.55))
for y, g in zip(yy, order):
    d = SESS[g]; d = d[(d.user_id == ent) & (d.end > w0) & (d.start < w1)].sort_values("start")
    busy = pd.Timestamp("1900-01-01")
    for s0, e0 in zip(d.start, d.end):
        x0 = (s0 - w0).total_seconds() / 86400; x1 = (e0 - w0).total_seconds() / 86400
        ax.add_patch(Rectangle((x0, y - 0.3), x1 - x0, 0.6, facecolor=STYLE[g]["color"],
                               edgecolor="white", linewidth=0.5, zorder=2))
        if s0 < busy - pd.Timedelta(seconds=60):                         # overlap with an earlier session
            xo = (min(busy, e0) - w0).total_seconds() / 86400
            ax.add_patch(Rectangle((x0, y - 0.3), xo - x0, 0.6, facecolor=RED, edgecolor="white",
                                   linewidth=0.5, zorder=3))
        busy = max(busy, e0)
    ax.text(14.15, y, f"{res[g]['overlap']:.1f} %", va="center", ha="left", fontsize=6.5, color=INK)
ax.text(14.15, len(order) - 0.35, "overlapping", va="center", ha="left", fontsize=6.5, color=INK)
for dday in range(15):
    ax.axvline(dday, color=GRID, lw=0.4, zorder=0)
ax.axvspan(5, 7, color="0.94", lw=0, zorder=0); ax.axvspan(12, 14, color="0.94", lw=0, zorder=0)
ax.set_xlim(0, 14); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xticks(np.arange(14) + 0.5); ax.set_xticklabels(list("MTWTFSS") * 2, fontsize=6.5)
ax.tick_params(length=0); ax.set_yticks(yy); ax.set_yticklabels(order, color=INK, fontsize=6.5)
for k, sp in ax.spines.items(): sp.set_visible(False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=RED, label="starts before the previous session has ended")],
          loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=6.5, handlelength=1.2)
save(fig, "fig_timeline")
print("timeline entity:", ent, "| window", w0.date(), "to", w1.date())
