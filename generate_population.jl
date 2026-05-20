# =============================================================================
# Population Synthetic Data Generator
# =============================================================================
#
# Uses the (user, day_type) profiles produced by Stage 3 of SMC_framework.jl
# (artifacts/profiles_corrected.jls) to synthesize a population of N users
# over an arbitrary calendar window [start_date, end_date].
#
# A "source user" is one for which BOTH Weekday and Weekend profiles exist
# in the saved profile dictionary. To simulate one synthetic user we sample
# one such source user and run two independent renewal processes (weekday +
# weekend) on the requested calendar window, exactly as in Stage 4.
#
# If n_users exceeds the number of available source users, the remaining
# synthetic users are sampled WITH REPLACEMENT from the same pool, so each
# extra synthetic user reuses an existing profile pair (different RNG path).
#
# Public entry point:
#   generate_population(start_date, end_date, n_users; ...) -> DataFrame
# =============================================================================

using CSV, DataFrames, Dates, Distributions, Printf, Random, Serialization, Statistics
using TruncatedGaussianMixtures   # required so deserialize can resolve GMM types in profiles_corrected.jls

const ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")
const PROFILES_PATH = joinpath(ARTIFACTS_DIR, "profiles_corrected.jls")

const NUM_BINS    = 12
const BIN_WIDTH_H = 24.0 / NUM_BINS

# ----- Struct needed for deserialization --------------------------------------
# Must match the struct defined in SMC_framework.jl Stage 2 byte-for-byte.

mutable struct UserDayProfile
    user_id  :: AbstractString
    day_type :: AbstractString
    pi0      :: Vector{Float64}
    P        :: Matrix{Float64}
    dist_D   :: Vector{Any}
    dist_G   :: Vector{Any}
    dist_E   :: Vector{Any}
    dist_A   :: Vector{Any}
    pool_A   :: Vector{Vector{Float64}}
    rho      :: Vector{Float64}
    bin_pass :: Vector{Bool}
    n_bin    :: Vector{Int}
    fallback :: Vector{Int}
end

# ----- Helpers ----------------------------------------------------------------

function sample_1d(model)
    x = rand(model)
    return x isa AbstractArray ? x[1] : Float64(x)
end

"""
    load_profiles(path = PROFILES_PATH) -> Dict{Tuple{String,String}, UserDayProfile}

Load the profile dictionary written by Stage 3.
"""
function load_profiles(path::AbstractString = PROFILES_PATH)
    isfile(path) || error("profiles file not found: $path — run SMC_framework.jl first")
    return open(deserialize, path)
end

"""
    source_users_with_both_profiles(profiles) -> Vector{String}

Return source user_ids that have BOTH a Weekday and Weekend profile.
Only these users are valid sampling units for a synthetic user.
"""
function source_users_with_both_profiles(profiles)
    wkday = Set{String}()
    wkend = Set{String}()
    for (uid, dt) in keys(profiles)
        dt == "Weekday" && push!(wkday, String(uid))
        dt == "Weekend" && push!(wkend, String(uid))
    end
    return sort(collect(intersect(wkday, wkend)))
end

"""
    assign_source_users(pool, n_users; rng) -> Vector{String}

Assign a source user id to each of the n_users synthetic users.
If n_users ≤ length(pool) → sample without replacement (each source used once).
If n_users >  length(pool) → use every source once, then top up with replacement.
"""
function assign_source_users(pool::Vector{String}, n_users::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    isempty(pool) && error("no source users have both Weekday and Weekend profiles")
    if n_users <= length(pool)
        return shuffle(rng, pool)[1:n_users]
    end
    extra = rand(rng, pool, n_users - length(pool))
    return vcat(shuffle(rng, pool), extra)
end

# ----- Same-type calendar budget over a date range ----------------------------
#
# Count exact hours that fall on weekdays vs weekends within [start_date, end_date]
# (inclusive). Each whole calendar day in range contributes 24 h to its bucket.
# This replaces Stage 4's "5 * full_weeks + min(rem, 5)" approximation, which
# assumes the window starts on a Monday.

function same_type_budget_hours(start_date::Date, end_date::Date)
    end_date >= start_date || error("end_date must be ≥ start_date")
    wkday_h, wkend_h = 0.0, 0.0
    d = start_date
    while d <= end_date
        if dayofweek(d) >= 6
            wkend_h += 24.0
        else
            wkday_h += 24.0
        end
        d += Day(1)
    end
    return wkday_h, wkend_h
end

# ----- Single-user simulation (one day_type leg) ------------------------------
#
# Adapted from Stage 4. Runs the renewal process for one day_type over the
# requested calendar window, using the given profile. Writes session rows into
# `sim_rows` tagged with `synth_user_id`.

function simulate_one_leg!(sim_rows::Vector,
                            synth_user_id::AbstractString,
                            source_uid::AbstractString,
                            prof::UserDayProfile,
                            start_date::Date,
                            budget_hours::Float64;
                            rng::AbstractRNG = Random.GLOBAL_RNG)
    budget_hours <= 0 && return
    dt          = String(prof.day_type)
    target_wknd = (dt == "Weekend")
    t0_dt       = DateTime(start_date)

    b     = rand(rng, Distributions.Categorical(prof.pi0))
    t_eff = 0.0

    while true
        d_dist = prof.dist_D[b]
        g_dist = prof.dist_G[b]
        e_dist = prof.dist_E[b]

        # --- Sample (D, G) — independent or via Gaussian copula ---
        if d_dist === nothing || g_dist === nothing
            d_n, g_n = 1.0, 1.0
        elseif prof.bin_pass[b] || isnan(prof.rho[b])
            d_n = max(sample_1d(d_dist), 0.01)
            g_n = max(sample_1d(g_dist), 0.0)
        else
            ρ   = prof.rho[b]
            z_d = randn(rng)
            z_g = ρ * z_d + sqrt(1 - ρ^2) * randn(rng)
            u_d = cdf(Normal(), z_d)
            u_g = cdf(Normal(), z_g)
            d_n = max(quantile(d_dist, clamp(u_d, 1e-6, 1-1e-6)), 0.01)
            g_n = max(quantile(g_dist, clamp(u_g, 1e-6, 1-1e-6)), 0.0)
        end

        τ = d_n + g_n
        τ <= 0 && (τ = 1.0)

        t_eff_next = t_eff + τ
        t_eff_next >= budget_hours && break

        # --- Effective-time → calendar map Φ_d(t_eff, t0) ---
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

        # --- Arrival hour ---
        a_dist = prof.dist_A[b]
        a = if a_dist !== nothing
                clamp(sample_1d(a_dist), 0.0, 23.99)
            elseif !isempty(prof.pool_A[b])
                rand(rng, prof.pool_A[b])
            else
                BIN_WIDTH_H * (b - 1) + BIN_WIDTH_H * rand(rng)
            end

        e_n = e_dist === nothing ? 0.01 : max(sample_1d(e_dist), 0.01)

        push!(sim_rows, (
            user_id        = synth_user_id,
            source_user_id = source_uid,
            day_type       = dt,
            bin            = b,
            start_dt       = cal_dt,
            calendar_date  = Date(cal_dt),
            arrival_hour   = round(a,   digits=4),
            duration_h     = round(d_n, digits=4),
            gap_h          = round(g_n, digits=4),
            tau_eff_h      = round(τ,   digits=4),
            energy_kwh     = round(e_n, digits=4),
        ))

        # --- Next bin ---
        b     = rand(rng, Distributions.Categorical(prof.P[b, :]))
        t_eff = t_eff_next
    end
end

# ----- Public entry point -----------------------------------------------------

"""
    generate_population(start_date, end_date, n_users;
                        profiles = nothing,
                        rng      = Random.GLOBAL_RNG,
                        save_csv = nothing,
                        verbose  = true)

Generate synthetic charging-session data for `n_users` users over the calendar
window `[start_date, end_date]` (inclusive). Each synthetic user is backed by
one source user from the profile dictionary; its Weekday and Weekend legs are
simulated independently.

Arguments
- `start_date`, `end_date` : `Date`s defining the inclusive window.
- `n_users`                : number of synthetic users to generate (≥ 1).

Keyword arguments
- `profiles`  : pre-loaded profile dict; if `nothing`, loaded from disk.
- `rng`       : RNG for reproducibility.
- `save_csv`  : if a path string, write the resulting DataFrame as CSV.
- `verbose`   : print a short summary.

Returns a `DataFrame` of synthetic sessions.
"""
function generate_population(start_date::Date, end_date::Date, n_users::Int;
                              profiles                                  = nothing,
                              rng::AbstractRNG                          = Random.GLOBAL_RNG,
                              save_csv::Union{Nothing,AbstractString}   = nothing,
                              verbose::Bool                             = true)
    n_users >= 1     || error("n_users must be ≥ 1")
    end_date >= start_date || error("end_date must be ≥ start_date")

    profiles === nothing && (profiles = load_profiles())
    pool       = source_users_with_both_profiles(profiles)
    assignment = assign_source_users(pool, n_users; rng=rng)

    wkday_h, wkend_h = same_type_budget_hours(start_date, end_date)

    if verbose
        println("─"^60)
        println("Population synthetic data generator")
        println("─"^60)
        @printf("  Window               : %s → %s  (%d days)\n",
                 start_date, end_date, (end_date - start_date).value + 1)
        @printf("  Weekday budget       : %.0f h\n", wkday_h)
        @printf("  Weekend budget       : %.0f h\n", wkend_h)
        @printf("  Source pool size     : %d users (both day-types)\n", length(pool))
        @printf("  Synthetic users      : %d\n", n_users)
        if n_users > length(pool)
            @printf("  Reused source users  : %d (sampled with replacement)\n",
                     n_users - length(pool))
        end
    end

    sim_rows = NamedTuple[]
    pad      = ndigits(n_users)
    for (i, source_uid) in enumerate(assignment)
        synth_uid = "sim_user_" * lpad(i, pad, '0')
        prof_wd   = profiles[(source_uid, "Weekday")]
        prof_we   = profiles[(source_uid, "Weekend")]

        simulate_one_leg!(sim_rows, synth_uid, source_uid, prof_wd,
                           start_date, wkday_h; rng=rng)
        simulate_one_leg!(sim_rows, synth_uid, source_uid, prof_we,
                           start_date, wkend_h; rng=rng)
    end

    sim_df = DataFrame(sim_rows)
    isempty(sim_df) || sort!(sim_df, [:user_id, :start_dt])

    if verbose
        @printf("\n  Synthetic sessions   : %d\n", nrow(sim_df))
        if nrow(sim_df) > 0
            @printf("    Weekday            : %d\n", sum(sim_df.day_type .== "Weekday"))
            @printf("    Weekend            : %d\n", sum(sim_df.day_type .== "Weekend"))
            @printf("  Median D (hours)     : %.2f\n", median(sim_df.duration_h))
            @printf("  Median G (hours)     : %.2f\n", median(sim_df.gap_h))
            @printf("  Median E (kWh)       : %.2f\n", median(sim_df.energy_kwh))
        end
    end

    if save_csv !== nothing
        CSV.write(save_csv, sim_df)
        verbose && println("\n  Saved → ", save_csv)
    end

    return sim_df
end

# ----- Script-mode example ---------------------------------------------------
# Run only when this file is executed directly (`julia generate_population.jl`),
# not when it is `include`d from another file.

if abspath(PROGRAM_FILE) == @__FILE__
    df = generate_population(
        Date(2026, 1, 1), Date(2026, 3, 31), 50;
        save_csv = joinpath(ARTIFACTS_DIR, "population_sim.csv"),
    )
    println("datasaved")
end
