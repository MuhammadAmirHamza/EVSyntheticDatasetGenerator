# =============================================================================
# Dataset Description Plots — final version
# Run in a fresh Julia session (no Plots.jl loaded).
# =============================================================================

using CSV, DataFrames, Dates, Statistics, Printf
using CairoMakie

CairoMakie.activate!(type = "png")
Makie.inline!(true)   # render in IDE plot pane; prevents OS image viewer (Photos) popup
const MK = CairoMakie

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR     = "plots"
mkpath(PLOTS_DIR)

# ── Configuration ─────────────────────────────────────────────────────────────
RANK_RANGE  = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # rank positions to display (1 = most sessions); skips rank 5
N_USERS     = length(RANK_RANGE)
MARKER_SIZE = 6      # ← change to adjust scatter point size in timeline
DUR_MAX_H   = 48.0   # clip duration axis at this value for readability
ENERGY_MAX  = 80.0   # clip energy axis (kWh) for readability

# ── Figure styling (applied across all single-panel plots) ────────────────────
FIG_W        = 720       # single-panel figure width  (px)
FIG_H        = 460       # single-panel figure height (px)
FIG_FONT     = 16        # base fontsize
TITLE_SIZE   = 22
LABEL_SIZE   = 20
TICK_SIZE    = 16
LEGEND_SIZE  = 15
CB_LABEL_SIZE = 18       # colorbar label
CB_TICK_SIZE  = 16       # colorbar tick
# Heatmap-specific size (matrix fills more of the figure, colorbar closer)
HMAP_W   = 780
HMAP_H   = 540
CB_GAP   = 6        # spacing (px) between heatmap axis and colorbar column
# Multi-panel (2×2 validation overlay) styling
MFIG_W            = 1100
MFIG_H            = 820
PANEL_TITLE_SIZE  = 18
PANEL_LABEL_SIZE  = 18
PANEL_TICK_SIZE   = 14
PANEL_LEGEND_SIZE = 13
# Save resolution
SAVE_PXU = 4        # px_per_unit when saving PNGs (higher = sharper)

# Apply bold x/y axis labels everywhere via a global theme
MK.set_theme!(MK.Theme(
    Axis = (xlabelfont = :bold, ylabelfont = :bold),
))

# ── Load data ─────────────────────────────────────────────────────────────────
raw = CSV.read(joinpath(ARTIFACTS_DIR, "raw_clean.csv"), DataFrame)

if eltype(raw.abs_start) <: AbstractString
    raw.abs_start = DateTime.(raw.abs_start)
end
if eltype(raw.abs_end) <: AbstractString
    raw.abs_end = DateTime.(raw.abs_end)
end

raw.date            = Date.(raw.abs_start)
raw.arrival_hour    = [Dates.hour(t) + Dates.minute(t)/60.0 for t in raw.abs_start]
raw.departure_hour  = mod.(raw.arrival_hour .+ raw.duration_hours, 24.0)
raw.bin             = [floor(Int, h / 2) + 1 for h in raw.arrival_hour]
raw.day_type        = ifelse.(dayofweek.(raw.date) .>= 6, "Weekend", "Weekday")

# Sort users by session count descending, take the slice at RANK_RANGE
user_counts = sort(combine(groupby(raw, :user_id), nrow => :n), :n, rev = true)
top_ids     = user_counts.user_id[RANK_RANGE]
raw_top     = filter(r -> r.user_id in top_ids, raw)

# Assign rank labels using the actual position (rank 1 = most sessions overall)
ranks_list   = collect(RANK_RANGE)
uid_to_rank  = Dict(uid => ranks_list[i] for (i, uid) in enumerate(top_ids))
raw_top.rank = [uid_to_rank[u] for u in raw_top.user_id]

# ── Full dataset for population-level distributions (Plots 2, 3, 4) ───────────
data_full = CSV.read("data_filtered.csv", DataFrame)

_hour_of(t::Dates.Time)     = Dates.hour(t) + Dates.minute(t)/60.0
_hour_of(s::AbstractString) = (p = split(s, ":");
                               parse(Float64, p[1]) + parse(Float64, p[2])/60.0)

data_full.arrival_hour   = _hour_of.(data_full.start_time)
data_full.departure_hour = mod.(data_full.arrival_hour .+ data_full.duration_hours, 24.0)

date_to_f(d::Date) = Float64(Dates.value(d))

# ── Shared helper: adjacent paired barplot ────────────────────────────────────
# Weekday and weekend bars are side-by-side within each bin,
# bars of same day-type are touching across adjacent bins.
# `centers` is a range/vector of bin midpoints; bar geometry scales to its step.
function paired_barplot!(ax, centers, wk_vals, we_vals;
                          bw_frac  = 0.42,
                          gap_frac = 0.08,
                          wk_color = (:steelblue, 0.85),
                          we_color = (:tomato, 0.80))
    spacing = length(centers) > 1 ? Float64(centers[2] - centers[1]) : 1.0
    bw      = bw_frac  * spacing
    gap     = gap_frac * spacing
    offset  = bw/2 + gap/2
    MK.barplot!(ax, collect(centers) .- offset, wk_vals;
        width = bw, color = wk_color,
        strokecolor = :white, strokewidth = 0.2, label = "Weekday",
        gap = 0)
    MK.barplot!(ax, collect(centers) .+ offset, we_vals;
        width = bw, color = we_color,
        strokecolor = :white, strokewidth = 0.2, label = "Weekend",
        gap = 0)
end

function hour_counts(hours, n_bins = 24)
    edges = 0:1:n_bins
    counts = Float64[count(h -> edges[i] <= h < edges[i+1], hours)
                     for i in 1:n_bins]
    counts ./ sum(counts)
end

# =============================================================================
# Plot 1 — Session timeline (users in RANK_RANGE, sorted by session count)
# =============================================================================

# Reverse so the highest-ranked user (lowest rank number) sits at the top of the axis
show_ids    = reverse(top_ids)
show_labels = ["User $(uid_to_rank[u])" for u in show_ids]

fig1 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax1  = MK.Axis(fig1[1, 1],
    title          = "Session Timeline",
    xlabel         = "Date",
    ylabel         = "User",
    yticks         = (1:N_USERS, show_labels),
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

for (row_idx, uid) in enumerate(show_ids)
    sub = filter(r -> r.user_id == uid, raw_top)
    MK.scatter!(ax1, date_to_f.(sub.date), fill(Float32(row_idx), nrow(sub));
        markersize = MARKER_SIZE, color = (:steelblue, 0.7))
end

date_ticks = Date(2018,1,1):Month(6):Date(2022,1,1)
ax1.xticks = (date_to_f.(date_ticks), Dates.format.(date_ticks, "yyyy-mm"))
ax1.xticklabelrotation = π/4

MK.save(joinpath(PLOTS_DIR, "fig_dataset_timeline.png"), fig1, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_timeline.png")

# =============================================================================
# Plot 2 — Arrival hour distribution (weekday vs weekend, adjacent bars)
# =============================================================================

wk_arr = hour_counts(data_full.arrival_hour[data_full.day_type .== "Weekday"])
we_arr = hour_counts(data_full.arrival_hour[data_full.day_type .== "Weekend"])

fig2 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax2  = MK.Axis(fig2[1, 1],
    title          = "Arrival Hour Distribution",
    xlabel         = "Hour of day",
    ylabel         = "Proportion of sessions",
    xticks         = 0:2:24,
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

paired_barplot!(ax2, 0.5:1.0:23.5, wk_arr, we_arr)
MK.axislegend(ax2, position = :lt, framevisible = false, labelsize = LEGEND_SIZE)
MK.save(joinpath(PLOTS_DIR, "fig_dataset_arrival_hour.png"), fig2, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_arrival_hour.png")

# =============================================================================
# Plot 3 — Departure hour distribution (weekday vs weekend, adjacent bars)
# =============================================================================

wk_dep = hour_counts(data_full.departure_hour[data_full.day_type .== "Weekday"])
we_dep = hour_counts(data_full.departure_hour[data_full.day_type .== "Weekend"])

fig3 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax3  = MK.Axis(fig3[1, 1],
    title          = "Departure Hour Distribution",
    xlabel         = "Hour of day",
    ylabel         = "Proportion of sessions",
    xticks         = 0:2:24,
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

paired_barplot!(ax3, 0.5:1.0:23.5, wk_dep, we_dep)
MK.axislegend(ax3, position = :lt, framevisible = false, labelsize = LEGEND_SIZE)
MK.save(joinpath(PLOTS_DIR, "fig_dataset_departure_hour.png"), fig3, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_departure_hour.png")

# =============================================================================
# Plot 4 — Plug-in duration distribution (weekday vs weekend)
# =============================================================================

wk_dur = data_full.duration_hours[data_full.day_type .== "Weekday"]
we_dur = data_full.duration_hours[data_full.day_type .== "Weekend"]
wk_dur = wk_dur[wk_dur .<= DUR_MAX_H]
we_dur = we_dur[we_dur .<= DUR_MAX_H]

n_dur       = 24
bin_w_dur   = DUR_MAX_H / n_dur                       # 2.0-hour bins
dur_edges   = 0.0:bin_w_dur:DUR_MAX_H
dur_centers = (bin_w_dur/2):bin_w_dur:(DUR_MAX_H - bin_w_dur/2)

dur_props(vals) = begin
    c = Float64[count(v -> dur_edges[i] <= v < dur_edges[i+1], vals)
                for i in 1:n_dur]
    length(vals) > 0 ? c ./ length(vals) : c
end

wk_dur_p = dur_props(Float64.(wk_dur))
we_dur_p = dur_props(Float64.(we_dur))

fig4 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax4  = MK.Axis(fig4[1, 1],
    title          = "Plug-in Duration Distribution",
    xlabel         = "Duration (hours)",
    ylabel         = "Proportion of sessions",
    xticks         = 0:6:Int(DUR_MAX_H),
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

paired_barplot!(ax4, dur_centers, wk_dur_p, we_dur_p)

MK.axislegend(ax4, position = :rt, framevisible = false, labelsize = LEGEND_SIZE)
MK.save(joinpath(PLOTS_DIR, "fig_dataset_duration.png"), fig4, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_duration.png")

# =============================================================================
# Plot 5 — Heatmap: jump (transition) probability between time-of-day bins
# Reads pairs.csv (one row per consecutive session pair):
#   bin_i    = arrival bin of the current session
#   bin_next = arrival bin of the next session
# P[i, j] = P(next bin = j | current bin = i)
# =============================================================================

NUM_BINS_T = 12   # 2-hour bins, matches NUM_BINS in the SMC framework
pairs_df   = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"), DataFrame)

N_trans = zeros(Float64, NUM_BINS_T, NUM_BINS_T)
for r in eachrow(pairs_df)
    N_trans[r.bin_i, r.bin_next] += 1.0
end
# Row-normalise; empty rows stay all-zero (no outgoing transitions observed)
row_totals = sum(N_trans, dims = 2)
P_trans    = N_trans ./ ifelse.(row_totals .> 0, row_totals, 1.0)

bin_labels = [@sprintf("%02d:00", 2*(b-1)) for b in 1:NUM_BINS_T]

fig5 = MK.Figure(resolution = (HMAP_W, HMAP_H), fontsize = FIG_FONT)
ax5  = MK.Axis(fig5[1, 1],
    title              = "Jump Probability Distribution",
    xlabel             = "Next bin (arrival hour)",
    ylabel             = "Current bin (arrival hour)",
    xticks             = (1:NUM_BINS_T, bin_labels),
    yticks             = (1:NUM_BINS_T, bin_labels),
    xticklabelrotation = π/4,
    titlesize          = TITLE_SIZE,
    xlabelsize         = LABEL_SIZE,
    ylabelsize         = LABEL_SIZE,
    xticklabelsize     = TICK_SIZE,
    yticklabelsize     = TICK_SIZE,
    aspect             = MK.DataAspect())

hm = MK.heatmap!(ax5, 1:NUM_BINS_T, 1:NUM_BINS_T, P_trans';
    colormap   = Reverse(:viridis),
    colorrange = (0.0, maximum(P_trans)))

MK.Colorbar(fig5[1, 2], hm;
    label         = "P(next | current)",
    labelsize     = CB_LABEL_SIZE,
    ticklabelsize = CB_TICK_SIZE,
    width         = 16)
MK.colgap!(fig5.layout, 1, CB_GAP)

MK.save(joinpath(PLOTS_DIR, "fig_dataset_heatmap.png"), fig5, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_heatmap.png")

# =============================================================================
# Plot 6 — Scatter: duration vs energy (full dataset)
# =============================================================================

dur_xy = Float64.(data_full.duration_hours)
e_xy   = Float64.(data_full.energy)
keep   = (dur_xy .<= DUR_MAX_H) .& (e_xy .<= ENERGY_MAX)

fig6 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax6  = MK.Axis(fig6[1, 1],
    title          = "Plug-in Duration vs Energy",
    xlabel         = "Plug-in duration (hours)",
    ylabel         = "Energy delivered (kWh)",
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

MK.scatter!(ax6, dur_xy[keep], e_xy[keep];
    markersize = 3, color = (:steelblue, 0.35))

MK.save(joinpath(PLOTS_DIR, "fig_dataset_duration_energy.png"), fig6, px_per_unit = SAVE_PXU)
println("Saved → plots/fig_dataset_duration_energy.png")

println("\nAll dataset plots saved to plots/")
# =============================================================================
# Coefficient of Variation of τ_eff
# Justifies SMC over CTMC: CV ≠ 1 invalidates the exponential assumption
# =============================================================================
 
pairs = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"), DataFrame)
 
tau_all = Float64.(pairs.tau_eff)
cv_pop  = std(tau_all) / mean(tau_all)
 
println("\n", "─"^55)
println("Coefficient of Variation of τ_eff")
println("─"^55)
@printf("Population  mean  : %.2f h\n", mean(tau_all))
@printf("Population  std   : %.2f h\n", std(tau_all))
@printf("Population  CV    : %.3f  (CTMC requires CV = 1.0)\n\n", cv_pop)
 
for dt in ["Weekday", "Weekend"]
    tau_dt = Float64.(pairs.tau_eff[pairs.day_type .== dt])
    @printf("%s  CV = %.3f  (mean = %.1f h,  std = %.1f h)\n",
        dt, std(tau_dt)/mean(tau_dt), mean(tau_dt), std(tau_dt))
end
 
# Per-user CVs (all users, not just top 10)
user_cvs = Float64[]
for sub in groupby(pairs, :user_id)
    tau_u = Float64.(sub.tau_eff)
    length(tau_u) < 10 && continue
    push!(user_cvs, std(tau_u) / mean(tau_u))
end
 
println()
@printf("Per-user CV (%d users):\n", length(user_cvs))
@printf("  Mean   : %.3f\n", mean(user_cvs))
@printf("  Median : %.3f\n", median(user_cvs))
@printf("  Min    : %.3f\n", minimum(user_cvs))
@printf("  Max    : %.3f\n\n", maximum(user_cvs))
 
# CV ranges with physical interpretations
ranges = [
    (0.0,  0.3,  "CV < 0.3",          "Near-deterministic — user charges at near-fixed intervals. Exponential assumption severely over-estimates variability."),
    (0.3,  0.9,  "0.3 ≤ CV < 0.9",   "Sub-exponential — more regular than Poisson; user has consistent habits. CTMC over-estimates idle gap dispersion."),
    (0.9,  1.1,  "CV ≈ 1.0",          "Consistent with exponential — CTMC adequate; memoryless gaps."),
    (1.1,  2.0,  "1.1 ≤ CV < 2.0",   "Over-dispersed — mixture of short and long gaps; bursty charging. Exponential under-estimates tail probability."),
    (2.0,  Inf,  "CV ≥ 2.0",          "Highly over-dispersed — heavy-tailed idle gaps; long inactive periods dominate. CTMC drastically under-estimates zero-session days."),
]
 
println("CV range breakdown:")
println("─"^55)
for (lo, hi, label, interp) in ranges
    n   = count(c -> lo <= c < hi, user_cvs)
    pct = 100 * n / length(user_cvs)
    @printf("  %-22s : %3d users (%5.1f%%)\n    → %s\n\n",
        label, n, pct, interp)
end



##
# =============================================================================
# Implementation Plots (Section V-B)
#   Plot 1 — BIC family selection grouped bar chart
#   Plot 2 — Copula rho distribution histogram
#   Plot 3 — Fallback level heatmap (bin x profile)
#
# Reads  : artifacts/profiles_summary.csv
#          artifacts/convolution_diagnostic.csv
# Saves  : plots/fig_impl_bic_families.png
#          plots/fig_impl_copula_rho.png
#          plots/fig_impl_fallback_heatmap.png
# =============================================================================

using CSV, DataFrames, Statistics, Printf
using CairoMakie

CairoMakie.activate!(type = "png")
const MK = CairoMakie

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR     = "plots"
mkpath(PLOTS_DIR)

profiles = CSV.read(joinpath(ARTIFACTS_DIR, "profiles_summary.csv"), DataFrame)
diag     = CSV.read(joinpath(ARTIFACTS_DIR, "convolution_diagnostic.csv"), DataFrame)

# =============================================================================
# Plot 1 — BIC family selection: grouped bar chart
# =============================================================================

FAMILIES = ["LogNormal", "Weibull", "Exponential",
            "TruncGMM(K=1)", "TruncGMM(K=2)", "TruncGMM(K=3)", "none"]

LEGEND_LABELS = ["LogNormal", "Weibull", "Exponential",
                 "GMM(K=1)", "GMM(K=2)", "GMM(K=3)", "none"]

# Light blue/orange palette: parametric families in blues, GMM variants in oranges
FAM_COLORS = [
    "#C6DBEF",   # LogNormal     — very light blue
    "#9ECAE1",   # Weibull       — light blue
    "#6BAED6",   # Exponential   — medium-light blue
    "#FEE6CE",   # GMM(K=1)      — very light orange
    "#FDD0A2",   # GMM(K=2)      — light orange
    "#FDAE6B",   # GMM(K=3)      — medium-light orange
    "#D9D9D9",   # none          — light gray (drawn but excluded from legend)
]

QUANTITIES = ["D", "G", "E", "A"]
QTY_COLS   = [:D_family, :G_family, :E_family, :A_family]

n_qty = length(QUANTITIES)
n_fam = length(FAMILIES)

counts = zeros(Int, n_qty, n_fam)
for (qi, col) in enumerate(QTY_COLS)
    for (fi, fam) in enumerate(FAMILIES)
        counts[qi, fi] = count(==(fam), skipmissing(profiles[!, col]))
    end
end

bar_w   = 0.11
offsets = range(-(n_fam-1)/2, (n_fam-1)/2, length=n_fam) .* bar_w

fig1 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax1  = MK.Axis(fig1[1, 1],
    title          = "BIC Family Selection by Variable",
    xlabel         = "Variable",
    ylabel         = "Number of bins",
    xticks         = (1:n_qty, QUANTITIES),
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

plot_handles = Any[]
for (fi, fam) in enumerate(FAMILIES)
    xs = collect(1:n_qty) .+ offsets[fi]
    ys = Float64.(counts[:, fi])
    p = MK.barplot!(ax1, xs, ys;
        width       = bar_w,
        color       = FAM_COLORS[fi],
        strokecolor = :white,
        strokewidth = 0.3)
    push!(plot_handles, p)
end

# Legend below the axis, "none" entry excluded
legend_idx = findall(!=("none"), FAMILIES)
MK.Legend(fig1[2, 1],
    plot_handles[legend_idx],
    LEGEND_LABELS[legend_idx];
    orientation     = :horizontal,
    nbanks          = 1,
    framevisible    = false,
    labelsize       = LEGEND_SIZE,
    tellheight      = true,
    tellwidth       = false,
    padding         = (4, 4, 4, 4))

MK.save(joinpath(PLOTS_DIR, "fig_impl_bic_families.png"), fig1, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_impl_bic_families.png")

# =============================================================================
# Plot 2 — Copula rho distribution (failing bins only)
# NOTE: use L"..." for math rendering in Makie axis labels
# =============================================================================

rho_vals = Float64.(collect(skipmissing(diag.rho_copula)))

fig2 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax2  = MK.Axis(fig2[1, 1],
    title          = L"Gaussian Copula Correlation $\hat{\rho}_b$",
    xlabel         = L"Estimated correlation $\hat{\rho}_b$",
    ylabel         = "Number of bins",
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

MK.hist!(ax2, rho_vals;
    bins        = 15,
    color       = (:steelblue, 0.80),
    strokecolor = :white,
    strokewidth = 0.4)

MK.vlines!(ax2, [0.0];
    color     = :black,
    linestyle = :dash,
    linewidth = 1.2,
    label     = L"\rho = 0")

MK.vlines!(ax2, [mean(rho_vals)];
    color     = :tomato,
    linestyle = :dash,
    linewidth = 1.5,
    label     = @sprintf("Mean = %.2f", mean(rho_vals)))

MK.axislegend(ax2, position = :lt, framevisible = false, labelsize = LEGEND_SIZE)

@printf("  rho: n=%d, mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n",
    length(rho_vals), mean(rho_vals), std(rho_vals),
    minimum(rho_vals), maximum(rho_vals))

MK.save(joinpath(PLOTS_DIR, "fig_impl_copula_rho.png"), fig2, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_impl_copula_rho.png")

# =============================================================================
# Plot 3 — Fallback level heatmap: bin (x) x profile (y)
# =============================================================================

NUM_BINS    = 12
MAX_USERS_H = 50   # cap on number of users shown in the fallback heatmap

# Restrict to the first MAX_USERS_H users (stable, alphabetical by user_id)
all_users      = sort(unique(profiles.user_id))
selected_users = all_users[1:min(MAX_USERS_H, length(all_users))]
profiles_h     = filter(r -> r.user_id in selected_users, profiles)
sort!(profiles_h, [:user_id, :day_type])

# Get unique (user_id, day_type) pairs in stable sorted order
prof_df   = unique(profiles_h[!, [:user_id, :day_type]])
n_prof    = nrow(prof_df)
heat_fall = zeros(Int, n_prof, NUM_BINS)

for (pi, row) in enumerate(eachrow(prof_df))
    sub = filter(r -> r.user_id == row.user_id &&
                      r.day_type == row.day_type, profiles_h)
    for r in eachrow(sub)
        b = r.bin
        1 <= b <= NUM_BINS || continue
        heat_fall[pi, b] = r.fallback
    end
end

bin_labels = [@sprintf("%02d:00", 2*(b-1)) for b in 1:NUM_BINS]

fig3 = MK.Figure(resolution = (HMAP_W, HMAP_H), fontsize = FIG_FONT)
ax3  = MK.Axis(fig3[1, 1],
    title              = "Fallback Level per Bin and Profile",
    xlabel             = "Time-of-day bin",
    ylabel             = "Profile index (user x day-type)",
    xticks             = (1:NUM_BINS, bin_labels),
    xticklabelrotation = pi/4,
    titlesize          = TITLE_SIZE,
    xlabelsize         = LABEL_SIZE,
    ylabelsize         = LABEL_SIZE,
    xticklabelsize     = TICK_SIZE,
    yticklabelsize     = TICK_SIZE)

hm = MK.heatmap!(ax3, 1:NUM_BINS, 1:n_prof, heat_fall';
    colormap   = Reverse(:viridis),
    colorrange = (0, 3))

MK.Colorbar(fig3[1, 2], hm;
    ticks         = ([0, 1, 2, 3], ["0", "L1", "L2", "L3"]),
    label         = "Fallback level",
    labelsize     = CB_LABEL_SIZE,
    ticklabelsize = CB_TICK_SIZE,
    width         = 16)
MK.colgap!(fig3.layout, 1, CB_GAP)

n_l1 = count(==(1), heat_fall)
n_l2 = count(==(2), heat_fall)
n_l3 = count(==(3), heat_fall)
@printf("  Fallback: L1=%d, L2=%d, L3=%d\n", n_l1, n_l2, n_l3)

MK.save(joinpath(PLOTS_DIR, "fig_impl_fallback_heatmap.png"), fig3, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_impl_fallback_heatmap.png")

println("\nAll implementation plots saved to plots/")

##

# =============================================================================
# Validation Results Plots (Section V-C/D combined)
#   Plot 1 — L1-L4 summary bar chart
#   Plot 2 — Real vs simulated overlays (4-panel: A, D, G, E)
#
# Reads  : artifacts/raw_clean.csv
#          artifacts/pairs.csv
#          artifacts/sim_sessions.csv
#          artifacts/validation_l1.csv
#          artifacts/validation_l2.csv
#          artifacts/validation_l3.csv
#          artifacts/kl_divergence.txt
# Saves  : plots/fig_val_summary.png
#          plots/fig_val_overlay.png
# =============================================================================

using CSV, DataFrames, Dates, Statistics, Printf
using CairoMakie

CairoMakie.activate!(type = "png")
const MK = CairoMakie

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR     = "plots"
mkpath(PLOTS_DIR)

# ── Load validation results ───────────────────────────────────────────────────
l1   = CSV.read(joinpath(ARTIFACTS_DIR, "validation_l1.csv"), DataFrame)
l2   = CSV.read(joinpath(ARTIFACTS_DIR, "validation_l2.csv"), DataFrame)
l3   = CSV.read(joinpath(ARTIFACTS_DIR, "validation_l3.csv"), DataFrame)
raw  = CSV.read(joinpath(ARTIFACTS_DIR, "raw_clean.csv"),     DataFrame)
pairs_df = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"),     DataFrame)
sim  = CSV.read(joinpath(ARTIFACTS_DIR, "sim_sessions.csv"),  DataFrame)

if eltype(raw.abs_start) <: AbstractString
    raw.abs_start = DateTime.(raw.abs_start)
end
raw.arrival_hour = [Dates.hour(t) + Dates.minute(t)/60.0 for t in raw.abs_start]

# Read KL divergence
kl_line = readlines(joinpath(ARTIFACTS_DIR, "kl_divergence.txt"))[1]
kl_val  = parse(Float64, first(split(strip(split(kl_line, "=")[2]))))

# =============================================================================
# Plot 1 — Validation summary: pass rates for L1(A/D/G/E), L2, L3
# =============================================================================

metrics = ["L1-A", "L1-D", "L1-G", "L1-E", "L2", "L3"]

function pass_rate(df, var=nothing)
    sub = var === nothing ? df :
          filter(r -> r.variable == var && !isnan(r.ks_stat), df)
    100 * sum(sub.passed) / max(nrow(sub), 1)
end

rates = [
    pass_rate(l1, "A"),
    pass_rate(l1, "D"),
    pass_rate(l1, "G"),
    pass_rate(l1, "E"),
    pass_rate(l2),
    pass_rate(l3),
]

colors = [
    (:steelblue, 0.85),
    (:steelblue, 0.85),
    (:steelblue, 0.85),
    (:steelblue, 0.85),
    (:tomato,    0.85),
    (:seagreen,  0.85),
]

fig1 = MK.Figure(resolution = (FIG_W, FIG_H), fontsize = FIG_FONT)
ax1  = MK.Axis(fig1[1, 1],
    title          = "Validation Pass Rates (L1--L3)",
    xlabel         = "Metric",
    ylabel         = "Pass rate (%)",
    xticks         = (1:length(metrics), metrics),
    limits         = (nothing, nothing, 0, 105),
    xgridvisible   = false,
    titlesize      = TITLE_SIZE,
    xlabelsize     = LABEL_SIZE,
    ylabelsize     = LABEL_SIZE,
    xticklabelsize = TICK_SIZE,
    yticklabelsize = TICK_SIZE)

MK.barplot!(ax1, 1:length(metrics), rates;
    color       = colors,
    strokecolor = :white,
    strokewidth = 0.4,
    width       = 0.6)

# Threshold line at 80%
MK.hlines!(ax1, [80.0];
    color     = :black,
    linestyle = :dash,
    linewidth = 1.2,
    label     = "80% threshold")

# Value labels on bars
for (i, r) in enumerate(rates)
    MK.text!(ax1, i, r + 1.5;
        text      = @sprintf("%.1f%%", r),
        align     = (:center, :bottom),
        fontsize  = LEGEND_SIZE,
        color     = :black)
end

# L4 annotation box
MK.text!(ax1, 3.5, 96.0;
    text     = @sprintf("L4: KL = %.4f nats (excellent)", kl_val),
    align    = (:center, :bottom),
    fontsize = LEGEND_SIZE,
    color    = :darkgreen)

MK.axislegend(ax1, position = :lb, framevisible = false, labelsize = LEGEND_SIZE)
MK.save(joinpath(PLOTS_DIR, "fig_val_summary.png"), fig1, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_val_summary.png")

# =============================================================================
# Plot 2 — Real vs simulated overlays: A, D, G, E
# Each panel: KDE or histogram overlay, population-level
# =============================================================================

# Clip values for readable axes
D_MAX  = 48.0
G_MAX  = 72.0
TAU_MAX = 120.0

real_A = Float64.(raw.arrival_hour)
real_D = Float64.(raw.duration_hours[raw.duration_hours .<= D_MAX])
real_G = Float64.(pairs_df.G_eff[pairs_df.G_eff .<= G_MAX])
real_E = Float64.(raw.energy[raw.energy .<= 80.0])

sim_A  = Float64.(sim.arrival_hour)
sim_D  = Float64.(sim.duration_h[sim.duration_h .<= D_MAX])
sim_G  = Float64.(sim.gap_h[sim.gap_h .<= G_MAX])
sim_E  = Float64.(sim.energy_kwh[sim.energy_kwh .<= 80.0])

real_color = (:steelblue, 0.70)
sim_color  = (:tomato,    0.60)

function overlay_hist!(ax, real_vals, sim_vals, bins;
                        real_lbl="Real", sim_lbl="Simulated")
    MK.hist!(ax, real_vals;
        bins        = bins,
        normalization = :probability,
        color       = real_color,
        strokecolor = :white, strokewidth = 0.2,
        label       = real_lbl)
    MK.hist!(ax, sim_vals;
        bins        = bins,
        normalization = :probability,
        color       = sim_color,
        strokecolor = :white, strokewidth = 0.2,
        label       = sim_lbl)
end

fig2 = MK.Figure(resolution = (MFIG_W, MFIG_H), fontsize = FIG_FONT)

# Panel (a): Arrival hour
ax_A = MK.Axis(fig2[1, 1],
    title          = "(a) Arrival hour",
    xlabel         = "Hour of day",
    ylabel         = "Proportion",
    xticks         = 0:4:24,
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_hist!(ax_A, real_A, sim_A, collect(range(0, 24, length=49)))
MK.axislegend(ax_A, position = :lt, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

# Panel (b): Plug-in duration
ax_D = MK.Axis(fig2[1, 2],
    title          = "(b) Plug-in duration",
    xlabel         = "Duration (h)",
    ylabel         = "Proportion",
    xticks         = 0:6:Int(D_MAX),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_hist!(ax_D, real_D, sim_D, collect(range(0, D_MAX, length=25)))
MK.axislegend(ax_D, position = :rt, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

# Panel (c): Idle gap G_eff
ax_G = MK.Axis(fig2[2, 1],
    title          = "(c) Effective idle gap Gᵉᶠᶠ",
    xlabel         = "Gap (h)",
    ylabel         = "Proportion",
    xticks         = 0:12:Int(G_MAX),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_hist!(ax_G, real_G, sim_G, collect(range(0, G_MAX, length=25)))
MK.axislegend(ax_G, position = :rt, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

# Panel (d): Energy
ax_E = MK.Axis(fig2[2, 2],
    title          = "(d) Energy delivered",
    xlabel         = "Energy (kWh)",
    ylabel         = "Proportion",
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_hist!(ax_E, real_E, sim_E, collect(range(0, 80, length=30)))
MK.axislegend(ax_E, position = :rt, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

MK.save(joinpath(PLOTS_DIR, "fig_val_overlay.png"), fig2, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_val_overlay.png")

# =============================================================================
# Plot 3 — Real vs simulated CDFs: A, D, G, E
# Empirical CDF complement to the histogram overlay above. Population-level.
# Reuses real_A/D/G/E and sim_A/D/G/E from Plot 2 (already clipped).
# =============================================================================

function empirical_cdf(x::AbstractVector{<:Real})
    s = sort(x)
    n = length(s)
    return s, collect(1:n) ./ n
end

function overlay_cdf!(ax, real_vals, sim_vals; real_lbl="Real", sim_lbl="Simulated")
    rx, ry = empirical_cdf(real_vals)
    sx, sy = empirical_cdf(sim_vals)
    MK.lines!(ax, rx, ry;
        color     = real_color,
        linewidth = 3.5,
        label     = real_lbl)
    MK.lines!(ax, sx, sy;
        color     = sim_color,
        linewidth = 3.5,
        linestyle = :dash,
        label     = sim_lbl)
end

fig3 = MK.Figure(resolution = (MFIG_W, MFIG_H), fontsize = FIG_FONT)

ax_Ac = MK.Axis(fig3[1, 1],
    title          = "(a) Arrival hour",
    xlabel         = "Hour of day",
    ylabel         = "Cumulative probability",
    xticks         = 0:4:24,
    limits         = (0, 24, 0, 1.02),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_cdf!(ax_Ac, real_A, sim_A)
MK.axislegend(ax_Ac, position = :lt, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

ax_Dc = MK.Axis(fig3[1, 2],
    title          = "(b) Plug-in duration",
    xlabel         = "Duration (h)",
    ylabel         = "Cumulative probability",
    xticks         = 0:6:Int(D_MAX),
    limits         = (0, D_MAX, 0, 1.02),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_cdf!(ax_Dc, real_D, sim_D)
MK.axislegend(ax_Dc, position = :rb, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

ax_Gc = MK.Axis(fig3[2, 1],
    title          = "(c) Effective idle gap Gᵉᶠᶠ",
    xlabel         = "Gap (h)",
    ylabel         = "Cumulative probability",
    xticks         = 0:12:Int(G_MAX),
    limits         = (0, G_MAX, 0, 1.02),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_cdf!(ax_Gc, real_G, sim_G)
MK.axislegend(ax_Gc, position = :rb, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

ax_Ec = MK.Axis(fig3[2, 2],
    title          = "(d) Energy delivered",
    xlabel         = "Energy (kWh)",
    ylabel         = "Cumulative probability",
    limits         = (0, 80, 0, 1.02),
    xgridvisible   = false,
    titlesize      = PANEL_TITLE_SIZE, xlabelsize = PANEL_LABEL_SIZE, ylabelsize = PANEL_LABEL_SIZE,
    xticklabelsize = PANEL_TICK_SIZE, yticklabelsize = PANEL_TICK_SIZE)
overlay_cdf!(ax_Ec, real_E, sim_E)
MK.axislegend(ax_Ec, position = :rb, framevisible = false, labelsize = PANEL_LEGEND_SIZE)

# Per-panel KS statistics for quick numerical comparison
function ks_stat(x, y)
    sx = sort(Float64.(x)); sy = sort(Float64.(y))
    all_pts = sort(unique(vcat(sx, sy)))
    cdf_x(t) = searchsortedlast(sx, t) / length(sx)
    cdf_y(t) = searchsortedlast(sy, t) / length(sy)
    maximum(abs(cdf_x(t) - cdf_y(t)) for t in all_pts)
end
@printf("  KS(real, sim): A=%.3f  D=%.3f  G=%.3f  E=%.3f\n",
    ks_stat(real_A, sim_A), ks_stat(real_D, sim_D),
    ks_stat(real_G, sim_G), ks_stat(real_E, sim_E))

MK.save(joinpath(PLOTS_DIR, "fig_val_cdf.png"), fig3, px_per_unit = SAVE_PXU)
println("Saved -> plots/fig_val_cdf.png")

println("\nAll validation plots saved to plots/")