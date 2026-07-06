# run_combined_N.jl — pooled 3-cohort combined SMC (Norway+GIST+Kim, 155 users).
#   Usage: julia --project=. run_combined_N.jl [N] [num_bins] [seed]   defaults 200 24 24
using Printf
const N        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
const BINS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
const SEED_RUN = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 24
const DATA_PATH           = joinpath(@__DIR__, "dataset", "combined", "combined_sessions.csv")
const TOP_N_USERS         = N
const Q_UPPER             = 1.0
const UPPER_INCLUSIVE     = true
const A_KMAX              = 6
const G_EFF_MAX_HOURS_RUN = 10 * 24.0
NUM_BINS                  = BINS
ARTIFACTS_DIR             = joinpath(@__DIR__, "artifacts", @sprintf("combined_N%03d_b%02d_s%02d", N, BINS, SEED_RUN))
mkpath(ARTIFACTS_DIR)
@printf("▶ combined | N=%d bins=%d seed=%d\n  artifacts: %s\n", N, BINS, SEED_RUN, ARTIFACTS_DIR)
include(joinpath(@__DIR__, "smc_core.jl"))
