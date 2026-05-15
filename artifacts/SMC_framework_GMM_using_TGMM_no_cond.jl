# =============================================================================
# Stage 1: Data Preparation
# =============================================================================
#
# What this stage does:
#   1. Load the filtered CSV and apply dataset-level quality filters
#   2. Select the top N users by session count
#   3. For each user and day-type, form consecutive same-day-type pairs and
#      compute the effective idle gap G_eff (eq. 20) and τ_eff = D + G_eff
#
# What we DO compute:
#   - G_i^eff  : same-type idle gap (integral of 1{dt(s)=d} from T_end to T_next_start)
#   - τ_i^eff  : D_i + G_i^eff
#
# What we do NOT compute:
#   - D_i^eff  : not needed. D_i is directly observed; the session start
#                determines which day-type profile it belongs to. The plug-in
#                duration is used in full regardless of day-type boundaries.
#
# Outputs (saved to artifacts/):
#   raw_clean.csv   — session-level table after all filters
#   pairs.csv       — enriched pair-level table (one row per consecutive pair)
# =============================================================================

using CSV, DataFrames, Dates, Printf, Distributions, Statistics

# ----- Configuration ----------------------------------------------------------

const DATA_PATH   = "data_filtered.csv"
const TOP_N_USERS = 100
const N_MIN       = 30           # minimum sessions to keep a user
const Q_UPPER     = 0.98         # upper percentile cap for session count
const KS_ALPHA       = 0.05         # significance level for KS and χ² tests
const CORR_THRESHOLD = 0.3          # |r| flag for correlated bins
const N_OBS_VAL      = 10           # min observations for validation tests
const KL_EPS         = 1e-9         # epsilon for KL divergence

const ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")
isdir(ARTIFACTS_DIR) || mkdir(ARTIFACTS_DIR)

# ----- Load and filter --------------------------------------------------------

println("─"^60)
println("Stage 1: Data Preparation")
println("─"^60)

raw = CSV.read(DATA_PATH, DataFrame)
@printf("  Loaded                : %d sessions, %d users\n",
         nrow(raw), length(unique(raw.user_id)))

# Build absolute datetimes from the date + time string columns
raw.abs_start = DateTime.(string.(raw.start_date) .* "T" .* string.(raw.start_time))
raw.abs_end   = DateTime.(string.(raw.end_date)   .* "T" .* string.(raw.end_time))

# Recompute arrival_hour cleanly
raw.arrival_hour = [Hour(t).value + Minute(t).value / 60.0 for t in raw.abs_start]

# Dataset-level filters (Section IV-A-1)
const DURATION_MAX_HOURS = 10 * 24.0   # drop sessions with duration > 3 days

raw = raw[raw.energy        .>  0.5,     :]   # aborted sessions
raw = raw[raw.duration_hours .>= 2/60,   :]   # spurious connections (< 2 min)
n_before_dur = nrow(raw)
raw = raw[raw.duration_hours .< DURATION_MAX_HOURS, :]   # pathological outliers
@printf("  After quality filters : %d sessions\n", nrow(raw))
@printf("  Dropped (D ≥ %.0fd)    : %d\n",
        DURATION_MAX_HOURS / 24, n_before_dur - nrow(raw))

# ----- User-level filter (eq. 19) --------------------------------------------

counts = combine(groupby(raw, :user_id), nrow => :n_u)
upper  = quantile(counts.n_u, Q_UPPER)
counts = counts[(counts.n_u .>= N_MIN) .& (counts.n_u .< upper), :]
sort!(counts, :n_u, rev=true)
top_uids = first(counts.user_id, TOP_N_USERS)
# raw = raw[in.(raw.user_id, Ref(Set(top_uids))), :]
raw = filter(n -> n.:user_id in top_uids, raw)
@printf("  Top %d users selected : %d sessions\n", TOP_N_USERS, nrow(raw))
@printf("  Session count range   : %d – %d (median %d)\n",
         minimum(counts.n_u[1:TOP_N_USERS]),
         maximum(counts.n_u[1:TOP_N_USERS]),
         floor(median(counts.n_u[1:TOP_N_USERS])))

# Save cleaned session table
CSV.write(joinpath(ARTIFACTS_DIR, "raw_clean.csv"), raw)
println("  Saved → artifacts/raw_clean.csv")

# ----- Pair construction (eq. 22) --------------------------------------------
#
# For each user and day-type, sort sessions chronologically and form
# consecutive same-day-type pairs (S_i, S_{i+1}).
#
# G_i^eff  = ∫_{T_end_i}^{T_start_{i+1}} 1{dt(s) = d} ds   [hours]
#
# We walk day boundaries between T_end and T_start_next, accumulating
# only the hours that belong to day-type d.  No need to do this for D_i
# because D_i is directly observed.
# -----------------------------------------------------------------------------

function compute_g_eff(t_end::DateTime, t_start_next::DateTime,
                        day_type::AbstractString)::Float64
    t_start_next <= t_end && return 0.0
    target_wknd = (day_type == "Weekend")
    acc = 0.0
    cur = t_end
    while cur < t_start_next
        next_midnight = DateTime(Date(cur)) + Day(1)
        boundary      = min(next_midnight, t_start_next)
        Δ             = (boundary - cur).value / 3_600_000.0   # ms → hours
        (dayofweek(cur) >= 6) == target_wknd && (acc += Δ)
        cur = boundary
    end
    return acc
end

NUM_BINS    = 12
BIN_WIDTH_H = 24.0 / NUM_BINS
bin_of(a)   = clamp(floor(Int, a / BIN_WIDTH_H) + 1, 1, NUM_BINS)

const G_EFF_MAX_HOURS = 10 * 24.0   # drop pairs with idle gap > 5 days

pair_rows         = NamedTuple[]
n_dropped_gap_max = 0

for sub in groupby(raw, [:user_id, :day_type])
    s = sort(sub, :abs_start)
    nrow(s) < 2 && continue

    for i in 1:(nrow(s) - 1)
        t_start_i    = s.abs_start[i]
        t_end_i      = s.abs_end[i]
        t_start_next = s.abs_start[i+1]

        # Skip overlapping sessions (end >= next start)
        t_end_i >= t_start_next && continue

        d_i   = Float64(s.duration_hours[i])
        g_eff = compute_g_eff(t_end_i, t_start_next, s.day_type[i])

        # Skip pairs where the effective idle gap is zero or negative
        # (can happen if session spans the full same-type window)
        g_eff < 0 && continue

        # Drop pairs whose same-type idle gap exceeds the cap
        if g_eff > G_EFF_MAX_HOURS
        global    n_dropped_gap_max += 1
            continue
        end

        tau_eff = d_i + g_eff

        push!(pair_rows, (
            user_id      = s.user_id[i],
            day_type     = s.day_type[i],
            bin_i        = bin_of(s.arrival_hour[i]),
            bin_next     = bin_of(s.arrival_hour[i+1]),
            D            = d_i,
            G_eff        = g_eff,
            tau_eff      = tau_eff,
            A            = Float64(s.arrival_hour[i]),
            E            = Float64(s.energy[i]),
            t_start_i    = t_start_i,
        ))
    end
end

pairs = DataFrame(pair_rows)

@printf("  Pairs constructed     : %d\n", nrow(pairs))
@printf("    Weekday pairs       : %d\n", sum(pairs.day_type .== "Weekday"))
@printf("    Weekend pairs       : %d\n", sum(pairs.day_type .== "Weekend"))
@printf("  Dropped (G_eff > %.0fd) : %d\n", G_EFF_MAX_HOURS / 24, n_dropped_gap_max)
@printf("  Median D  (hours)     : %.2f\n", median(pairs.D))
@printf("  Median G_eff (hours)  : %.2f\n", median(pairs.G_eff))
@printf("  Median τ_eff (hours)  : %.2f\n", median(pairs.tau_eff))

CSV.write(joinpath(ARTIFACTS_DIR, "pairs.csv"), pairs)
println("  Saved → artifacts/pairs.csv")

println("\nStage 1 complete.")
println("Inspect artifacts/raw_clean.csv and artifacts/pairs.csv before proceeding.")










# =============================================================================
# Stage 2: Profile Estimation
# =============================================================================
#
# Reads    : artifacts/pairs.csv  (from Stage 1)
# Produces : artifacts/profiles.jls           (binary, for Stage 3/4)
#            artifacts/profiles_summary.csv    (human-readable inspection)
#
# What this stage does:
#   1. Pre-compute dataset-global fallback fits ONCE (3 fits total)
#   2. Pre-compute user-global fallback fits ONCE per user
#   3. For each profile (user, day_type):
#      a. Estimate π₀ and P with Laplace smoothing (eq. 23-24)
#      b. For each bin, decide the fallback level:
#         - Level 1 (per-bin):       n_bin >= N_BIN_NOMINAL → fit per-bin
#         - Level 2 (user-global):   reuse pre-computed user fit
#         - Level 3 (dataset-global): reuse pre-computed global fit
#      c. Arrival hours: empirical pool per bin
#
# Key performance trick:
#   Fallback fits are computed ONCE and referenced by pointer.  A bin that
#   falls back to user-global or dataset-global does ZERO additional fitting.
# =============================================================================

using CSV, DataFrames, Dates, Distributions, Printf, Random, Serialization, Statistics
using Plots
# ----- Configuration ----------------------------------------------------------


const ALPHA_LAP     = 1e-4
const N_BIN_NOMINAL = 15       # min per-bin observations for Level 1
const N_POOL        = 5        # min user observations for Level 2

const ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")

Random.seed!(24)

# ----- Load pairs from Stage 1 ------------------------------------------------

println("─"^60)
println("Stage 2: Profile Estimation")
println("─"^60)

pairs = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"), DataFrame)
@printf("  Loaded pairs          : %d\n", nrow(pairs))

# ----- BIC-selected MLE fit (eq. 25-26) --------------------------------------

## ---- Truncated Gaussian Mixture Model via TruncatedGaussianMixtures.jl ------
#
# Uses the TruncatedGaussianMixtures.jl package for proper truncated EM.
# For 1D data:
#   - D, G, E: truncated to [0, upper_bound]  (upper_bound = max(data)*2)
#   - A:       truncated to [0, 24]
# The package returns a Distributions.MixtureModel, so rand(), pdf(), cdf()
# work natively — no custom struct needed.
#
# Install: using Pkg; Pkg.add("TruncatedGaussianMixtures")

using TruncatedGaussianMixtures

"""
    fit_trunc_gmm_pkg(x, K; lower, upper)

Fit a K-component truncated GMM to 1D data x using TruncatedGaussianMixtures.jl.
Returns (MixtureModel, BIC) or (nothing, Inf) on failure.
"""
function fit_trunc_gmm_pkg(x::Vector{Float64}, K::Int;
                            lower::Float64 = 0.0,
                            upper::Float64 = NaN)
    n = length(x)
    n < 2*K && return (nothing, Inf)

    # Default upper bound: 2× the observed max (generous but finite)
    if isnan(upper)
        upper = max(maximum(x) * 2.0, 1.0)
    end

    # TruncatedGaussianMixtures expects a 1×N matrix for 1D data
    X = reshape(x, 1, n)
    a = [lower]
    b = [upper]

    try
        model = fit_gmm(X, K, a, b;
                          cov      = :diag,
                          tol      = 1e-3,
                          MAX_REPS = 100,
                          verbose  = false,
                          progress = false)

        # model is a Distributions.MixtureModel — compute BIC
        # log-likelihood
        ll = sum(logpdf(model, X))
        # 3K - 1 free parameters (K weights - 1, K means, K variances)
        p  = 3*K - 1
        bic = p * log(n) - 2 * ll

        return (model, bic)
    catch
        return (nothing, Inf)
    end
end

"""
    sample_1d(model)

Draw a scalar sample from a 1D MixtureModel returned by TruncatedGaussianMixtures.
The package returns multivariate distributions (1D MvNormal components), so
rand() returns a 1-element vector — this helper unwraps it.
"""
function sample_1d(model)
    x = rand(model)
    return x isa AbstractArray ? x[1] : Float64(x)
end

## ---- Unified fit function: parametric families + truncated GMM ---------------

function fit_best(x::AbstractVector{<:Real};
                   gmm_lower::Float64 = 0.0,
                   gmm_upper::Float64 = NaN)
    xx = filter(>(0), Float64.(x))
    length(xx) < N_BIN_NOMINAL && return nothing
    n = length(xx)
    best, best_bic = nothing, Inf

    # Standard parametric families
    for (Fam, k) in [(LogNormal, 2), (Exponential, 1), (Weibull, 2)]
        try
            f   = fit_mle(Fam, xx)
            ll  = sum(logpdf.(f, xx))
            bic = k * log(n) - 2 * ll
            if bic < best_bic
                best, best_bic = f, bic
            end
        catch; end
    end

    # Truncated GMM with K ∈ {1, 2, 3} — always attempted, n < 2*K guard inside
    ub = isnan(gmm_upper) ? max(maximum(xx) * 2.0, 1.0) : gmm_upper
    for K in 1:3
        try
            (d, bic) = fit_trunc_gmm_pkg(xx, K;
                                          lower = gmm_lower,
                                          upper = ub)
            if d !== nothing && bic < best_bic
                best, best_bic = d, bic
            end
        catch; end
    end

    return best
end

# ----- Step 1: Pre-compute dataset-global fallbacks (3 fits per day_type) ----

println("  Fitting dataset-global fallbacks (Level 3) ...")
global_fits = Dict{String, NamedTuple}()
for sub in groupby(pairs, :day_type)
    dt = sub.day_type[1]
    global_fits[dt] = (
        D = fit_best(sub.D),
        G = fit_best(sub.G_eff),
        E = fit_best(sub.E),
        A = fit_best(sub.A; gmm_lower=0.0, gmm_upper=24.0),
    )
    fmt(f) = f === nothing ? "failed" :
             f isa MixtureModel ? "TruncGMM(K=$(ncomponents(f)))" :
             string(typeof(f).name.name)
    println("    [$dt]  D → ", fmt(global_fits[dt].D),
                    "  G → ", fmt(global_fits[dt].G),
                    "  E → ", fmt(global_fits[dt].E),
                    "  A → ", fmt(global_fits[dt].A))
end

using Plots, Distributions, StatsBase

# # Histogram as a density (so it's comparable to a PDF)
# plt = histogram(
#     pairs.E;
#     bins        = 100,
#     normalize   = :pdf,
#     alpha       = 0.5,
#     label       = "Empirical D",
#     xlabel      = "D (hours)",
#     ylabel      = "density",
#     title       = "Duration: empirical vs $(typeof(global_fit_D).name.name) fit",
# )

# # PDF curve over the empirical range
# xgrid = range(max(1e-6, minimum(pairs.E)), maximum(pairs.E); length = 400)
# plot!(plt, xgrid, pdf.(global_fit_E, xgrid);
#       lw    = 2,
#       color = :red,
#       label = "global_fit_D")

# display(plt)
# savefig(plt, joinpath(ARTIFACTS_DIR, "global_fit_G.png"))


# ----- Step 2: Pre-compute (user, day_type)-global fallbacks (1 fit per profile × 3 qty) ----

println("  Fitting (user, day_type)-global fallbacks (Level 2) ...")
user_fits = Dict{Tuple{String,String}, NamedTuple}()
for sub in groupby(pairs, [:user_id, :day_type])
    uid = sub.user_id[1]
    dt  = sub.day_type[1]
    user_fits[(uid, dt)] = (
        D = fit_best(sub.D),
        G = fit_best(sub.G_eff),
        E = fit_best(sub.E),
        A = fit_best(sub.A; gmm_lower=0.0, gmm_upper=24.0),
    )
end
@printf("    Cached %d (user, day_type)-level fits\n", length(user_fits))

# ----- Step 3: Per-profile estimation -----------------------------------------

mutable struct UserDayProfile
    user_id  :: AbstractString
    day_type :: AbstractString
    pi0      :: Vector{Float64}           # length B
    P        :: Matrix{Float64}           # B × B row-stochastic
    dist_D   :: Vector{Any}               # per-bin duration CDF
    dist_G   :: Vector{Any}               # per-bin idle gap CDF
    dist_E   :: Vector{Any}               # per-bin energy CDF
    dist_A   :: Vector{Any}               # per-bin arrival-hour CDF (truncated to [0, 24])
    pool_A   :: Vector{Vector{Float64}}   # per-bin arrival hour empirical pool (fallback)
    rho      :: Vector{Float64}           # copula ρ (filled in Stage 3)
    bin_pass :: Vector{Bool}              # convolution pass/fail (filled in Stage 3)
    n_bin    :: Vector{Int}               # per-bin pair counts
    fallback :: Vector{Int}               # fallback level used (1/2/3) per bin
end

profiles = Dict{Tuple{String,String}, UserDayProfile}()
n_level1 = 0; n_level2 = 0; n_level3 = 0

println("  Estimating per-profile parameters ...")
for (uid, dt) in unique(zip(pairs.user_id, pairs.day_type))
    prof_pairs = pairs[(pairs.user_id .== uid) .& (pairs.day_type .== dt), :]
    n_user_dt  = nrow(prof_pairs)

    # --- π₀ with Laplace smoothing (eq. 24) ---
    pi0 = fill(ALPHA_LAP, NUM_BINS)
    for b in prof_pairs.bin_i; pi0[b] += 1.0; end
    pi0 ./= sum(pi0)

    # --- P with Laplace smoothing (eq. 23) ---
    P = fill(ALPHA_LAP, NUM_BINS, NUM_BINS)
    for r in eachrow(prof_pairs); P[r.bin_i, r.bin_next] += 1.0; end
    P ./= sum(P, dims=2)

    # --- Per-bin distribution fitting with pre-computed fallbacks ---
    dist_D   = Vector{Any}(undef, NUM_BINS)
    dist_G   = Vector{Any}(undef, NUM_BINS)
    dist_E   = Vector{Any}(undef, NUM_BINS)
    dist_A   = Vector{Any}(undef, NUM_BINS)
    pool_A   = Vector{Vector{Float64}}(undef, NUM_BINS)
    n_bin    = zeros(Int, NUM_BINS)
    fallback = zeros(Int, NUM_BINS)

    # Per-quantity chained fallback: take the highest-resolution fit available
    # among (Level 1 per-bin, Level 2 per-(user, day_type), Level 3 per-day_type).
    # A nothing at any level transparently falls through to the next.
    pick3(l1, l2, l3) = l1 !== nothing ? (l1, 1) :
                        l2 !== nothing ? (l2, 2) :
                        l3 !== nothing ? (l3, 3) : (nothing, 3)

    for b in 1:NUM_BINS
        bin_rows = prof_pairs[prof_pairs.bin_i .== b, :]
        n_bin[b] = nrow(bin_rows)

        # Level 1: try per-bin only when there's enough data
        d1 = nrow(bin_rows) >= N_BIN_NOMINAL ? fit_best(bin_rows.D) : nothing
        g1 = nrow(bin_rows) >= N_BIN_NOMINAL ? fit_best(bin_rows.G_eff) : nothing
        e1 = nrow(bin_rows) >= N_BIN_NOMINAL ? fit_best(bin_rows.E) : nothing
        a1 = nrow(bin_rows) >= N_BIN_NOMINAL ? fit_best(bin_rows.A; gmm_lower=0.0, gmm_upper=24.0) : nothing

        # Level 2: per-(user, day_type) cache (only consult if pool is large enough)
        local l2 = n_user_dt >= N_POOL ? user_fits[(uid, dt)] :
                   (D = nothing, G = nothing, E = nothing, A = nothing)

        # Level 3: per-day_type cache
        local l3 = global_fits[dt]

        (dist_D[b], lD) = pick3(d1, l2.D, l3.D)
        (dist_G[b], lG) = pick3(g1, l2.G, l3.G)
        (dist_E[b], lE) = pick3(e1, l2.E, l3.E)
        (dist_A[b], lA) = pick3(a1, l2.A, l3.A)

        # Track the worst (highest-numbered) level used across quantities
        fallback[b] = max(lD, lG, lE, lA)
        fallback[b] == 1 ? (global n_level1 += 1) :
        fallback[b] == 2 ? (global n_level2 += 1) : (global n_level3 += 1)

        # Arrival hour pool: keep as empirical fallback if dist_A fails
        a_bin = Float64.(bin_rows.A)
        pool_A[b] = length(a_bin) >= N_POOL ?
                      a_bin :
                      collect(BIN_WIDTH_H*(b-1) .+ BIN_WIDTH_H .* rand(20))
    end

    profiles[(uid, dt)] = UserDayProfile(
        uid, dt, pi0, P, dist_D, dist_G, dist_E, dist_A, pool_A,
        fill(NaN, NUM_BINS),     # rho: filled in Stage 3
        fill(true, NUM_BINS),    # bin_pass: all true until Stage 3
        n_bin, fallback,
    )
end

n_profiles  = length(profiles)
n_bins_total = n_level1 + n_level2 + n_level3
@printf("  Profiles estimated    : %d\n", n_profiles)
@printf("  Total bin-slots       : %d\n", n_bins_total)
@printf("    Level 1 (per-bin)   : %d (%5.1f%%)\n",
         n_level1, 100*n_level1/n_bins_total)
@printf("    Level 2 (user)      : %d (%5.1f%%)\n",
         n_level2, 100*n_level2/n_bins_total)
@printf("    Level 3 (global)    : %d (%5.1f%%)\n",
         n_level3, 100*n_level3/n_bins_total)

# ----- Save profiles (binary for Stage 3/4) -----------------------------------

open(joinpath(ARTIFACTS_DIR, "profiles.jls"), "w") do io
    serialize(io, profiles)
end
println("  Saved → artifacts/profiles.jls")

# ----- Save human-readable summary -------------------------------------------

summary_rows = NamedTuple[]
for ((uid, dt), p) in profiles
    for b in 1:NUM_BINS
        fname(d) = d === nothing ? "none" :
                   d isa MixtureModel ? "TruncGMM(K=$(ncomponents(d)))" :
                   string(typeof(d).name.name)
        push!(summary_rows, (
            user_id   = uid,
            day_type  = dt,
            bin       = b,
            n_bin     = p.n_bin[b],
            fallback  = p.fallback[b],
            pi0       = round(p.pi0[b], digits=4),
            D_family  = fname(p.dist_D[b]),
            G_family  = fname(p.dist_G[b]),
            E_family  = fname(p.dist_E[b]),
            A_family  = fname(p.dist_A[b]),
            n_pool_A  = length(p.pool_A[b]),
        ))
    end
end
CSV.write(joinpath(ARTIFACTS_DIR, "profiles_summary.csv"), DataFrame(summary_rows))
println("  Saved → artifacts/profiles_summary.csv")

println("\nStage 2 complete.")
# println("Inspect artifacts/profiles_summary.csv before proceeding.")
println("Check: which bins use Level 1 vs 2 vs 3?  Which families won?")






# =============================================================================
# Stage 3: Convolution Diagnostic & Distribution Update
# =============================================================================
#
# Reads    : artifacts/profiles.jls   (from Stage 2)
#            artifacts/pairs.csv      (from Stage 1)
#
# Produces : artifacts/profiles_corrected.jls   (updated profiles with copula ρ)
#            artifacts/convolution_diagnostic.csv (per-bin pass/fail + correlations)
#
# What this stage does:
#   For each profile and each bin with enough data:
#   1. Draw N_sim samples from fitted D_b and G_b, form τ_sim = d + g
#   2. KS test: τ_sim vs empirical τ_eff — tests D ⊥ G | b
#   3. If pass → keep marginals as-is (independence holds)
#   4. If fail → fit a Gaussian copula on (D, G) via probability integral
#      transform and store ρ_b.  Simulation will draw (D, G) jointly.
#   5. Bins with < N_BIN_NOMINAL pairs are skipped (carried forward unchanged)
#
# Note: Only bins at fallback Level 1 (per-bin fit) are meaningfully testable.
#   Level 2/3 bins use pooled distributions that were not fitted to this
#   specific bin's data, so the convolution test is not informative for them.
#   We test them anyway for completeness and flag the fallback level.
# =============================================================================

using CSV, DataFrames, Distributions, HypothesisTests, Printf, Random
using Serialization, Statistics, StatsBase

const N_SIM_CONV = 10000
# ----- Load from previous stages ---------------------------------------------

println("─"^60)
println("Stage 3: Convolution Diagnostic & Distribution Update")
println("─"^60)

pairs = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"), DataFrame)
profiles = open(deserialize, joinpath(ARTIFACTS_DIR, "profiles.jls"))
@printf("  Loaded %d profiles, %d pairs\n", length(profiles), nrow(pairs))

# ----- Run convolution test per profile per bin -------------------------------

diag_rows = NamedTuple[]
n_tested = 0; n_pass = 0; n_fail = 0; n_fail_corr = 0; n_skipped = 0

for ((uid, dt), prof) in profiles
    prof_pairs = pairs[(pairs.user_id .== uid) .& (pairs.day_type .== dt), :]

    for b in 1:NUM_BINS
        bin_rows = prof_pairs[prof_pairs.bin_i .== b, :]

        # Skip bins with insufficient data
        if nrow(bin_rows) < N_BIN_NOMINAL
            global n_skipped += 1
            continue
        end

        d_dist = prof.dist_D[b]
        g_dist = prof.dist_G[b]

        # Skip if either distribution failed to fit
        if d_dist === nothing || g_dist === nothing
            global n_skipped += 1
            continue
        end

        # --- KS test on convolution (eq. 28-29) ---
        d_sim   = [sample_1d(d_dist) for _ in 1:N_SIM_CONV]
        g_sim   = [sample_1d(g_dist) for _ in 1:N_SIM_CONV]
        tau_sim = d_sim .+ g_sim
        tau_emp = Float64.(bin_rows.tau_eff)

        ks     = ApproximateTwoSampleKSTest(tau_sim, tau_emp)
        p_val  = pvalue(ks)
        passed = p_val > KS_ALPHA

        # --- D-G correlation ---
        d_emp = Float64.(bin_rows.D)
        g_emp = Float64.(bin_rows.G_eff)
        r_pear  = length(d_emp) > 2 ? cor(d_emp, g_emp)         : NaN
        r_spear = length(d_emp) > 2 ? corspearman(d_emp, g_emp) : NaN
        correlated = (!isnan(r_pear)  && abs(r_pear)  > CORR_THRESHOLD) ||
                     (!isnan(r_spear) && abs(r_spear) > CORR_THRESHOLD)

        # --- Update profile ---
        prof.bin_pass[b] = passed
        ρ = NaN

        if !passed
            # Fit Gaussian copula (eq. 30-32)
            try
                u_d = clamp.(cdf.(d_dist, d_emp), 1e-6, 1 - 1e-6)
                u_g = clamp.(cdf.(g_dist, g_emp), 1e-6, 1 - 1e-6)
                z_d = quantile.(Normal(), u_d)
                z_g = quantile.(Normal(), u_g)
                ρ   = clamp(cor(z_d, z_g), -0.99, 0.99)
            catch
                ρ = NaN
            end
            prof.rho[b] = ρ
        end

        # --- Counters ---
        global n_tested += 1
        if passed
            global n_pass += 1
        else
            global n_fail += 1
            correlated && (global n_fail_corr += 1)
        end

        # --- Log row ---
        push!(diag_rows, (
            user_id    = uid,
            day_type   = dt,
            bin        = b,
            n_pairs    = nrow(bin_rows),
            fallback   = prof.fallback[b],
            ks_stat    = round(ks.δ, digits=4),
            p_value    = round(p_val, digits=4),
            passed     = passed,
            r_pearson  = round(r_pear, digits=4),
            r_spearman = round(r_spear, digits=4),
            correlated = correlated,
            rho_copula = isnan(ρ) ? missing : round(ρ, digits=4),
        ))
    end
end

# ----- Summary ----------------------------------------------------------------

@printf("\n  Bins tested           : %d\n", n_tested)
@printf("  Bins skipped (sparse) : %d\n", n_skipped)
@printf("  Pass                  : %d (%5.1f%%)\n",
         n_pass, 100*n_pass/max(n_tested,1))
@printf("  Fail                  : %d (%5.1f%%)\n",
         n_fail, 100*n_fail/max(n_tested,1))
@printf("    of which correlated : %d\n", n_fail_corr)

# Among Level-1 bins only (per-bin fits — most informative test)
diag_df = DataFrame(diag_rows)
l1_diag = filter(r -> r.fallback == 1, diag_df)
if nrow(l1_diag) > 0
    n1_pass = sum(l1_diag.passed)
    n1_tot  = nrow(l1_diag)
    @printf("\n  Level-1 bins only     : %d tested, %d pass (%.1f%%)\n",
             n1_tot, n1_pass, 100*n1_pass/n1_tot)
end

# ----- Save outputs -----------------------------------------------------------

open(joinpath(ARTIFACTS_DIR, "profiles_corrected.jls"), "w") do io
    serialize(io, profiles)
end
println("\n  Saved → artifacts/profiles_corrected.jls")

CSV.write(joinpath(ARTIFACTS_DIR, "convolution_diagnostic.csv"), diag_df)
println("  Saved → artifacts/convolution_diagnostic.csv")

println("\nStage 3 complete. but solution is yet to implement")
# println("Inspect artifacts/convolution_diagnostic.csv before proceeding.")
# println("Check: which bins fail?  Are failures concentrated in specific bins or users?")
# println("Check: for failing bins, is ρ_copula positive (SoC-driven) or negative?")


##

# =============================================================================
# Stage 4: Simulation
# =============================================================================
#
# Reads    : artifacts/profiles_corrected.jls   (from Stage 3)
#            artifacts/raw_clean.csv             (from Stage 1, for date ranges)
#
# Produces : artifacts/sim_sessions.csv
#
# What this stage does:
#   For each user, extract [t0_u, t1_u] from their real session dates.
#   Run two independent renewal processes (weekday + weekend) over that
#   user's own calendar window.  Merge and sort by calendar date.
#
# Simulation per step:
#   1. Draw D_n ~ D_b  and  G_n ~ G_b  (independent if bin passed Stage 3,
#      joint Gaussian copula if bin failed)
#   2. τ_n = D_n + G_n
#   3. Advance effective-time clock by τ_n
#   4. Map back to calendar via Φ_d(t_eff, t0_u)
#   5. Draw E_n ~ E_b,  A_n ~ pool_A_b
#   6. Draw next bin b_{n+1} ~ P(b_n, ·)
#
# Note: For bins that failed the convolution test in Stage 3, (D, G) are
#   drawn jointly via a Gaussian copula using the stored ρ_b.  For passing
#   bins, D and G are drawn independently from their marginals.
# =============================================================================

using CSV, DataFrames, Dates, Distributions, Printf, Random, Serialization, Statistics

# ----- Configuration ----------------------------------------------------------

# const NUM_BINS    = 12
# const BIN_WIDTH_H = 24.0 / NUM_BINS

# const ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")

# Random.seed!(20260506)

# ----- Load from previous stages ---------------------------------------------

println("─"^60)
println("Stage 4: Simulation")
println("─"^60)

profiles = open(deserialize, joinpath(ARTIFACTS_DIR, "profiles_corrected.jls"))
raw      = CSV.read(joinpath(ARTIFACTS_DIR, "raw_clean.csv"), DataFrame)

# Parse abs_start if it came back as string from CSV
if eltype(raw.abs_start) <: AbstractString
    raw.abs_start = DateTime.(raw.abs_start)
end

@printf("  Loaded %d profiles, %d sessions\n", length(profiles), nrow(raw))

# ----- Compute per-user date ranges from the real data ------------------------

user_ranges = Dict{String, Tuple{Date, Date, Int}}()
for sub in groupby(raw, :user_id)
    uid   = sub.user_id[1]
    dates = Date.(sub.abs_start)
    t0    = minimum(dates)
    t1    = maximum(dates)
    H     = max((t1 - t0).value, 30)   # at least 30 days
    user_ranges[uid] = (t0, t1, H)
end
@printf("  Per-user date ranges  : %d users\n", length(user_ranges))

# Quick summary of horizons
horizons = [v[3] for v in values(user_ranges)]
@printf("  Horizon range (days)  : %d – %d (median %d)\n",
         minimum(horizons), maximum(horizons), floor(median(horizons)))

# ----- Simulation loop --------------------------------------------------------

sim_rows = NamedTuple[]

for uid in unique(raw.user_id)
    haskey(user_ranges, uid) || continue
    t0_u, t1_u, H_days = user_ranges[uid]
    t0_dt = DateTime(t0_u)

    for dt in ["Weekday", "Weekend"]
        haskey(profiles, (uid, dt)) || continue
        prof = profiles[(uid, dt)]

        # Same-type budget over this user's horizon
        full_weeks = div(H_days, 7)
        rem_days   = H_days - 7 * full_weeks
        budget = dt == "Weekday" ?
                   5 * 24.0 * full_weeks + min(rem_days, 5) * 24.0 :
                   2 * 24.0 * full_weeks + max(0, rem_days - 5) * 24.0

        b     = rand(Categorical(prof.pi0))
        t_eff = 0.0
        target_wknd = (dt == "Weekend")

        while true
            d_dist = prof.dist_D[b]
            g_dist = prof.dist_G[b]
            e_dist = prof.dist_E[b]

            # --- Sample (D, G) ---
            if d_dist === nothing || g_dist === nothing
                d_n, g_n = 1.0, 1.0
            elseif prof.bin_pass[b] || isnan(prof.rho[b])
                # Independent draws (bin passed convolution test)
                d_n = max(sample_1d(d_dist), 0.01)
                g_n = max(sample_1d(g_dist), 0.0)
            else
                # Gaussian copula draws (bin failed convolution test)
                ρ = prof.rho[b]
                z_d = randn()
                z_g = ρ * z_d + sqrt(1 - ρ^2) * randn()
                u_d = cdf(Normal(), z_d)
                u_g = cdf(Normal(), z_g)
                d_n = max(quantile(d_dist, clamp(u_d, 1e-6, 1-1e-6)), 0.01)
                g_n = max(quantile(g_dist, clamp(u_g, 1e-6, 1-1e-6)), 0.0)
            end

            τ = d_n + g_n
            τ <= 0 && (τ = 1.0)

            t_eff_next = t_eff + τ
            t_eff_next >= budget && break

            # --- Effective-to-calendar map Φ_d(t_eff, t0_u) ---
            cur, acc, cal_dt = t0_dt, 0.0, t0_dt
            while acc < t_eff_next
                next_midnight = DateTime(Date(cur)) + Day(1)
                Δ = (next_midnight - cur).value / 3_600_000.0
                if (dayofweek(cur) >= 6) == target_wknd
                    if acc + Δ >= t_eff_next
                        ms = round(Int, (t_eff_next - acc) * 3_600_000.0)
                        cal_dt = cur + Millisecond(ms)
                        break
                    end
                    acc += Δ
                end
                cur = next_midnight
            end

            # --- Sample arrival hour and energy from fitted distributions ---
            a_dist = prof.dist_A[b]
            if a_dist !== nothing
                a = sample_1d(a_dist)
                a = clamp(a, 0.0, 23.99)
            elseif !isempty(prof.pool_A[b])
                a = rand(prof.pool_A[b])
            else
                a = BIN_WIDTH_H * (b - 1) + BIN_WIDTH_H * rand()
            end
            e_n = e_dist === nothing ? 0.01 : max(sample_1d(e_dist), 0.01)

            push!(sim_rows, (
                user_id       = uid,
                day_type      = dt,
                bin           = b,
                calendar_date = Date(cal_dt),
                arrival_hour  = round(a,   digits=4),
                duration_h    = round(d_n, digits=4),
                gap_h         = round(g_n, digits=4),
                tau_eff_h     = round(τ,   digits=4),
                energy_kwh    = round(e_n, digits=4),
            ))

            # --- Next bin ---
            b     = rand(Categorical(prof.P[b, :]))
            t_eff = t_eff_next
        end
    end
end

sim_df = DataFrame(sim_rows)

# ----- Summary ----------------------------------------------------------------

@printf("\n  Synthetic sessions    : %d\n", nrow(sim_df))
@printf("  Users simulated       : %d\n", length(unique(sim_df.user_id)))
@printf("    Weekday sessions    : %d\n", sum(sim_df.day_type .== "Weekday"))
@printf("    Weekend sessions    : %d\n", sum(sim_df.day_type .== "Weekend"))
@printf("  Median D  (hours)     : %.2f\n", median(sim_df.duration_h))
@printf("  Median G  (hours)     : %.2f\n", median(sim_df.gap_h))
@printf("  Median τ  (hours)     : %.2f\n", median(sim_df.tau_eff_h))
@printf("  Median E  (kWh)       : %.2f\n", median(sim_df.energy_kwh))

# Compare real vs simulated session counts per user
real_counts = combine(groupby(raw, :user_id), nrow => :n_real)
sim_counts  = combine(groupby(sim_df, :user_id), nrow => :n_sim)
compare = innerjoin(real_counts, sim_counts, on=:user_id)
compare.ratio = compare.n_sim ./ compare.n_real
@printf("\n  Session count ratio (sim/real):\n")
@printf("    Mean   : %.2f\n", mean(compare.ratio))
@printf("    Median : %.2f\n", median(compare.ratio))
@printf("    Min    : %.2f\n", minimum(compare.ratio))
@printf("    Max    : %.2f\n", maximum(compare.ratio))

# ----- Save -------------------------------------------------------------------

CSV.write(joinpath(ARTIFACTS_DIR, "sim_sessions.csv"), sim_df)
println("\n  Saved → artifacts/sim_sessions.csv")

println("\nStage 4 complete.")
println("Inspect artifacts/sim_sessions.csv before proceeding to validation.")
# println("Check: does the sim/real session count ratio cluster around 1.0?")
# println("Check: do the median D, G, E look reasonable vs Stage 1 output?")


##


# =============================================================================
# Stage 5: Validation
# =============================================================================
#
# Reads    : artifacts/raw_clean.csv        (from Stage 1)
#            artifacts/pairs.csv            (from Stage 1)
#            artifacts/sim_sessions.csv     (from Stage 4)
#
# Produces : artifacts/validation_l1.csv    (per-profile, per-variable KS)
#            artifacts/validation_l2.csv    (per-profile transition χ²)
#            artifacts/validation_l3.csv    (per-user sessions-per-day χ²)
#            artifacts/kl_divergence.txt    (single L4 number)
#
# Four-level hierarchical validation:
#   L1 — Marginal KS on {A, D, G_eff, E} per profile
#   L2 — Transition matrix χ² per profile
#   L3 — Sessions-per-day χ² per user (using user's own date range)
#   L4 — Population-aggregate load curve KL divergence
# =============================================================================

using CSV, DataFrames, Dates, Distributions, HypothesisTests, Printf, Statistics

# ----- Load data from previous stages ----------------------------------------

println("─"^60)
println("Stage 5: Validation")
println("─"^60)

raw    = CSV.read(joinpath(ARTIFACTS_DIR, "raw_clean.csv"), DataFrame)
pairs  = CSV.read(joinpath(ARTIFACTS_DIR, "pairs.csv"), DataFrame)
sim_df = CSV.read(joinpath(ARTIFACTS_DIR, "sim_sessions.csv"), DataFrame)

# Parse dates if needed
if eltype(raw.abs_start) <: AbstractString
    raw.abs_start = DateTime.(raw.abs_start)
end

@printf("  Real sessions         : %d\n", nrow(raw))
@printf("  Real pairs            : %d\n", nrow(pairs))
@printf("  Simulated sessions    : %d\n", nrow(sim_df))


# =============================================================================
# L1 — Marginal distributions (KS test per profile, eq. 40)
# =============================================================================
# Variables tested: A (arrival hour), D (duration), G (idle gap), E (energy)
# These are the four quantities we actually fit — τ is derived as D + G.
# =============================================================================
println("\n  Running L1 (marginal KS) ...")

l1_rows = NamedTuple[]

for sub_real in groupby(pairs, [:user_id, :day_type])
    uid = sub_real.user_id[1]
    dt  = sub_real.day_type[1]

    sub_sim = filter(r -> r.user_id == uid && r.day_type == dt, sim_df)
    nrow(sub_sim) < N_OBS_VAL && continue

    for (var, real_col, sim_col) in [
        ("A", :A,     :arrival_hour),
        ("D", :D,     :duration_h),
        ("G", :G_eff, :gap_h),
        ("E", :E,     :energy_kwh),
    ]
        x = Float64.(sub_real[!, real_col])
        y = Float64.(sub_sim[!, sim_col])

        if length(x) < N_OBS_VAL || length(y) < N_OBS_VAL
            push!(l1_rows, (user_id=uid, day_type=dt, variable=var,
                             n_real=length(x), n_sim=length(y),
                             ks_stat=NaN, p_value=NaN, passed=false))
        else
            r = ApproximateTwoSampleKSTest(x, y)
            push!(l1_rows, (user_id=uid, day_type=dt, variable=var,
                             n_real=length(x), n_sim=length(y),
                             ks_stat=round(r.δ, digits=4),
                             p_value=round(pvalue(r), digits=4),
                             passed=pvalue(r) > KS_ALPHA))
        end
    end
end

l1 = DataFrame(l1_rows)
CSV.write(joinpath(ARTIFACTS_DIR, "validation_l1.csv"), l1)


# =============================================================================
# L2 — Transition matrix χ² (per profile, eq. 41)
# =============================================================================
println("  Running L2 (transition χ²) ...")

l2_rows = NamedTuple[]

for sub_real in groupby(pairs, [:user_id, :day_type])
    uid = sub_real.user_id[1]
    dt  = sub_real.day_type[1]
    nrow(sub_real) < N_OBS_VAL && continue

    sub_sim = filter(r -> r.user_id == uid && r.day_type == dt, sim_df)
    nrow(sub_sim) < 2 && continue

    # Real transition counts (Laplace smoothed)
    N_real = fill(ALPHA_LAP, NUM_BINS, NUM_BINS)
    for r in eachrow(sub_real)
        N_real[r.bin_i, r.bin_next] += 1.0
    end

    # Simulated transition counts (Laplace smoothed)
    sim_sorted = sort(sub_sim, :calendar_date)
    N_sim = fill(ALPHA_LAP, NUM_BINS, NUM_BINS)
    for i in 1:(nrow(sim_sorted) - 1)
        N_sim[sim_sorted.bin[i], sim_sorted.bin[i+1]] += 1.0
    end

    # Row-by-row χ², skipping cells with expected < 0.5
    χ2 = 0.0; dof = 0; rows_used = 0
    for b in 1:NUM_BINS
        rt = sum(N_real[b, :])
        st = sum(N_sim[b, :])
        rt <= ALPHA_LAP * NUM_BINS && continue
        st <= ALPHA_LAP * NUM_BINS && continue

        p_real   = N_real[b, :] ./ rt
        expected = p_real .* st
        for b2 in 1:NUM_BINS
            expected[b2] < 0.5 && continue
            χ2 += (N_sim[b, b2] - expected[b2])^2 / expected[b2]
            dof += 1
        end
        rows_used += 1
    end
    dof  = max(dof - rows_used, 1)
    pval = 1.0 - cdf(Chisq(dof), χ2)

    push!(l2_rows, (user_id=uid, day_type=dt,
                     n_real=nrow(sub_real), n_sim=nrow(sub_sim),
                     chi2=round(χ2, digits=4), dof=dof,
                     p_value=round(pval, digits=4),
                     passed=pval > KS_ALPHA))
end

l2 = DataFrame(l2_rows)
CSV.write(joinpath(ARTIFACTS_DIR, "validation_l2.csv"), l2)


# =============================================================================
# L3 — Sessions-per-day χ² at population level
# =============================================================================
# Pool real and simulated day counts across ALL users, then run a single
# goodness-of-fit χ² test on the population totals.
#
# Per-user histograms are saved to validation_l3_per_user.csv as diagnostics.
# The population-level result (single pass/fail) is saved to validation_l3.csv.
#
# Categories : {0, 1, ≥2} sessions per calendar day
# Test       : ChisqTest(h_pop_sim, h_pop_real ./ sum(h_pop_real))
# =============================================================================

println("  Running L3 (sessions-per-day χ², population level) ...")

# --- Accumulators for population totals ---
h_pop_real = zeros(Float64, 3)
h_pop_sim  = zeros(Int,     3)

per_user_rows = NamedTuple[]

for sub_real in groupby(raw, :user_id)
    uid     = sub_real.user_id[1]
    sub_sim = filter(r -> r.user_id == uid, sim_df)
    isempty(sub_sim) && continue

    # --- Real histogram over user's own observation window ---
    real_dates  = Date.(sub_real.abs_start)
    real_t0     = minimum(real_dates)
    real_t1     = maximum(real_dates)
    real_n_days = max((real_t1 - real_t0).value + 1, 1)

    spd_real = combine(groupby(DataFrame(date = real_dates), :date), nrow => :n)

    h_real = Float64[
        max(real_n_days - nrow(spd_real), 0),
        sum(spd_real.n .== 1),
        sum(spd_real.n .>= 2),
    ]

    # --- Simulated histogram over simulated date range ---
    sim_t0     = minimum(sub_sim.calendar_date)
    sim_t1     = maximum(sub_sim.calendar_date)
    sim_n_days = max((sim_t1 - sim_t0).value + 1, 1)

    spd_sim = combine(groupby(sub_sim, :calendar_date), nrow => :n)

    h_sim = Int[
        max(sim_n_days - nrow(spd_sim), 0),
        sum(spd_sim.n .== 1),
        sum(spd_sim.n .>= 2),
    ]

    sum(h_real) == 0 && continue
    sum(h_sim)  == 0 && continue

    # --- Accumulate into population totals ---
    h_pop_real .+= h_real
    h_pop_sim  .+= h_sim

    # --- Save per-user proportions as diagnostics ---
    p_real = h_real ./ sum(h_real)
    p_sim  = h_sim  ./ sum(h_sim)
    push!(per_user_rows, (
        user_id      = uid,
        real_n_days  = real_n_days,
        sim_n_days   = sim_n_days,
        real_0       = Int(h_real[1]),
        real_1       = Int(h_real[2]),
        real_2p      = Int(h_real[3]),
        sim_0        = h_sim[1],
        sim_1        = h_sim[2],
        sim_2p       = h_sim[3],
        prop_real_0  = round(p_real[1], digits = 4),
        prop_real_1  = round(p_real[2], digits = 4),
        prop_real_2p = round(p_real[3], digits = 4),
        prop_sim_0   = round(p_sim[1],  digits = 4),
        prop_sim_1   = round(p_sim[2],  digits = 4),
        prop_sim_2p  = round(p_sim[3],  digits = 4),
    ))
end

# --- Population-level χ² test ---
pop_probs = h_pop_real ./ sum(h_pop_real)
l3_test   = ChisqTest(h_pop_sim, pop_probs)
l3_pval   = pvalue(l3_test)
l3_passed = l3_pval > KS_ALPHA

p_pop_real = h_pop_real ./ sum(h_pop_real) .* 100
p_pop_sim  = h_pop_sim  ./ sum(h_pop_sim)  .* 100

# --- Save per-user diagnostics ---
l3_per_user = DataFrame(per_user_rows)
CSV.write(joinpath(ARTIFACTS_DIR, "validation_l3_per_user.csv"), l3_per_user)

# --- Save population-level result ---
l3 = DataFrame([(
    real_0_days  = Int(h_pop_real[1]),
    real_1_days  = Int(h_pop_real[2]),
    real_2p_days = Int(h_pop_real[3]),
    sim_0_days   = h_pop_sim[1],
    sim_1_days   = h_pop_sim[2],
    sim_2p_days  = h_pop_sim[3],
    prop_real_0  = round(p_pop_real[1], digits = 2),
    prop_real_1  = round(p_pop_real[2], digits = 2),
    prop_real_2p = round(p_pop_real[3], digits = 2),
    prop_sim_0   = round(p_pop_sim[1],  digits = 2),
    prop_sim_1   = round(p_pop_sim[2],  digits = 2),
    prop_sim_2p  = round(p_pop_sim[3],  digits = 2),
    chi2         = round(l3_test.stat,  digits = 4),
    dof          = round(Int, l3_test.df),
    p_value      = round(l3_pval,       digits = 4),
    passed       = l3_passed,
)])
CSV.write(joinpath(ARTIFACTS_DIR, "validation_l3.csv"), l3)

@printf("\n  L3 population χ²  : chi2 = %.4f, p = %.4f  →  %s\n",
    l3_test.stat, l3_pval, l3_passed ? "PASS" : "FAIL")
@printf("  Real  : 0-sess = %.1f%%,  1-sess = %.1f%%,  ≥2-sess = %.1f%%\n",
    p_pop_real[1], p_pop_real[2], p_pop_real[3])
@printf("  Sim   : 0-sess = %.1f%%,  1-sess = %.1f%%,  ≥2-sess = %.1f%%\n",
    p_pop_sim[1],  p_pop_sim[2],  p_pop_sim[3])
@printf("  Diff  : 0-sess = %+.1f pp, 1-sess = %+.1f pp, ≥2-sess = %+.1f pp\n",
    p_pop_sim[1]-p_pop_real[1],
    p_pop_sim[2]-p_pop_real[2],
    p_pop_sim[3]-p_pop_real[3])

# =============================================================================
# L4 — Population load curve KL divergence (eq. 42)
# =============================================================================
println("  Running L4 (population KL) ...")

real_hours = [floor(Int, h) % 24 for h in raw.arrival_hour]
sim_hours  = [floor(Int, h) % 24 for h in sim_df.arrival_hour]
real_hist  = Float64[count(==(h), real_hours) for h in 0:23]
sim_hist   = Float64[count(==(h), sim_hours)  for h in 0:23]
real_dist  = real_hist ./ sum(real_hist)
sim_dist   = sim_hist  ./ sum(sim_hist)
kl_div     = sum(real_dist .* log.((real_dist .+ KL_EPS) ./ (sim_dist .+ KL_EPS)))

open(joinpath(ARTIFACTS_DIR, "kl_divergence.txt"), "w") do io
    @printf(io, "KL(real || sim) = %.6f nats\n", kl_div)
    @printf(io, "\nHour  Real     Sim\n")
    for h in 0:23
        @printf(io, "  %02d   %.4f   %.4f\n", h, real_dist[h+1], sim_dist[h+1])
    end
end


# =============================================================================
# Summary
# =============================================================================
println("\n  ─────────────────────────────────")
println("  Summary")
println("  ─────────────────────────────────")

for v in ["A", "D", "G", "E"]
    sub = filter(r -> r.variable == v && !isnan(r.ks_stat), l1)
    nt  = nrow(sub)
    np  = sum(sub.passed)
    @printf("  L1 %-3s : %3d / %3d pass (%5.1f%%)\n",
             v, np, nt, 100 * np / max(nt, 1))
end

@printf("  L2     : %3d / %3d pass (%5.1f%%)\n",
         sum(l2.passed), nrow(l2),
         100 * sum(l2.passed) / max(nrow(l2), 1))

@printf("  L3     : KL = population χ² p = %.4f  →  %s\n",
         l3_pval, l3_passed ? "PASS" : "FAIL")

@printf("  L4     : KL = %.4f nats", kl_div)
if kl_div < 0.05
    println("  (excellent)")
elseif kl_div < 0.15
    println("  (acceptable)")
else
    println("  (attention needed)")
end

println("\n  Saved → artifacts/validation_l1.csv")
println("  Saved → artifacts/validation_l2.csv")
println("  Saved → artifacts/validation_l3.csv")
println("  Saved → artifacts/kl_divergence.txt")

println("\nStage 5 complete.")
println("Check: L1 D and G pass rates — low rates suggest parametric families")
println("  cannot capture multimodal shapes (consider GMM extension).")
println("Check: validation_l3_per_user.csv for per-user proportion diagnostics.")
println("Check: validation_l3.csv for the population-level χ² result.")
println("Check: kl_divergence.txt for the per-hour real vs sim breakdown.")