# =============================================================================
# run_smc.jl — single entry point for the shared SMC framework (smc_core.jl)
# -----------------------------------------------------------------------------
# Usage:   julia run_smc.jl <dataset> [num_bins] [a_kmax] [gap_max_days]
#   <dataset> ∈  norway | korea | pecan | combined | acn | georgia | kim
#   The 3 optional positional args override the config defaults (see below).
#
# This injects the per-dataset config values, then `include`s smc_core.jl —
# the byte-identical shared body of the four SMC_framework_*.jl scripts.
# With Random.seed!(24) inside the core, outputs are deterministic.
#
# VERIFICATION MODE (default): SUFFIX = "_module" writes to
#   artifacts/<name>_module/   — so you can diff against the originals in
#   artifacts/<name>/   WITHOUT overwriting them. Set SUFFIX = "" to make this
#   the canonical runner.
# =============================================================================

const SUFFIX = "_module"          # "" → write to the real artifacts/<name>/

# gap_days = idle-gap cap in DAYS (pairs with a larger same-type idle gap are
# dropped). 10 for the original datasets (unchanged results); the workplace/
# commercial sets use a tighter cap because their long (vacation-like) gaps are a
# sparse tail that smears the idle-gap fit.
const CONFIGS = Dict(
    "norway"   => (name="norway_sorensen",
                   data=("dataset","norway_sorensen","norway_sorensen_sessions.csv"),
                   top_n=100, q_upper=0.98, upper_inclusive=false, num_bins=12, a_kmax=3, gap_days=10),
    "korea"    => (name="korea_gist",
                   data=("dataset","korea_gist","korea_gist_sessions.csv"),
                   top_n=30,  q_upper=0.98, upper_inclusive=false, num_bins=12, a_kmax=3, gap_days=10),
    "pecan"    => (name="pecanstreet",
                   data=("dataset","pecanstreet","pecanstreet_sessions.csv"),
                   top_n=10,  q_upper=1.0,  upper_inclusive=true,  num_bins=24, a_kmax=6, gap_days=10),
    # Combined = the 5 selected cohorts pooled: norway 100 + korea 30 + pecan 10
    # + georgia 30 + kim 30 = 200 users. Built by dataset/combined/build_combined.py
    # from each dataset's SPARC-selected raw_clean.csv (population = selected N only).
    "combined" => (name="combined",
                   data=("dataset","combined","combined_sessions.csv"),
                   top_n=200, q_upper=1.0,  upper_inclusive=true,  num_bins=24, a_kmax=6, gap_days=10),
    "acn"      => (name="acn_caltech",
                   data=("dataset","acn_caltech","acn_sessions.csv"),
                   top_n=30,  q_upper=1.0,  upper_inclusive=true,  num_bins=24, a_kmax=6, gap_days=10),
    # Georgia Tech workplace (per-DRIVER). georgia_sessions.csv is WEEKDAY-ONLY,
    # so the two-stream sim naturally learns/simulates weekdays only (no Weekend
    # profile is ever created -> the "Weekend" branch is skipped per user).
    "georgia"  => (name="georgia",
                   data=("dataset","georgia","georgia_sessions.csv"),
                   top_n=30,  q_upper=1.0,  upper_inclusive=true,  num_bins=12, a_kmax=6, gap_days=4),
    # Kim et al. 2024 commercial network (per-DRIVER RF-card UserID). Location-
    # loyal drivers only (each charges at exactly ONE of the 14 location types;
    # facility column tags which). top-30 drivers by count (apartment 16, company 6, public area 5, public institution 3), both weekday & weekend (~20%
    # weekend). 5-day idle-gap cap. top_n=30 = top-30 drivers by session count.
    "kim"      => (name="kim_commercial",
                   data=("dataset","kim-commercial","kim_commercial_sessions.csv"),
                   top_n=30,  q_upper=1.0,  upper_inclusive=true,  num_bins=24, a_kmax=6, gap_days=5),
)

(length(ARGS) >= 1 && haskey(CONFIGS, ARGS[1])) ||
    error("usage: julia run_smc.jl <norway|korea|pecan|combined|acn|georgia|kim> [num_bins] [a_kmax] [gap_max_days]\n" *
          "  num_bins     : arrival-hour bins over 24h (24 -> 1h bins, 12 -> 2h bins). default = config\n" *
          "  a_kmax       : arrival-fit GMM component cap.                            default = config\n" *
          "  gap_max_days : drop session pairs whose same-type idle gap exceeds this. default = config")
cfg = CONFIGS[ARGS[1]]

# ----- optional CLI overrides (positional) -----------------------------------
#   ARGS[2] = num_bins, ARGS[3] = a_kmax, ARGS[4] = gap_max_days
#   julia run_smc.jl kim            -> config defaults (24 bins, a_kmax 6, 5-day cap)
#   julia run_smc.jl kim 24 6 4     -> 24 bins, a_kmax 6, 4-day idle-gap cap
num_bins_run = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : cfg.num_bins
a_kmax_run   = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : cfg.a_kmax
gap_days_run = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : Float64(cfg.gap_days)

# ----- inject the per-dataset configuration (consumed by smc_core.jl) --------
const DATA_PATH         = joinpath(@__DIR__, cfg.data...)
const TOP_N_USERS       = cfg.top_n
const Q_UPPER           = cfg.q_upper
const UPPER_INCLUSIVE   = cfg.upper_inclusive     # false → n_u < upper ; true → n_u <= upper
const A_KMAX            = a_kmax_run               # arrival-fit GMM cap (norway/korea default = 3)
const G_EFF_MAX_HOURS_RUN = gap_days_run * 24.0   # idle-gap cap (hours) consumed by smc_core.jl
ARTIFACTS_DIR           = joinpath(@__DIR__, "artifacts", cfg.name * SUFFIX)
NUM_BINS                = num_bins_run

println("▶ SMC framework | dataset=$(ARGS[1]) | users ≤ $(TOP_N_USERS) | " *
        "NUM_BINS=$(NUM_BINS) | A_KMAX=$(A_KMAX) | GAP_MAX=$(gap_days_run)d | " *
        "UPPER_INCLUSIVE=$(UPPER_INCLUSIVE)")
println("  data      : $(DATA_PATH)")
println("  artifacts : $(ARTIFACTS_DIR)")

include(joinpath(@__DIR__, "smc_core.jl"))
