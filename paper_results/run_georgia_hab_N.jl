# run_georgia_hab_N.jl — Georgia HABITUAL subset (arr_std<=2h, 12 users, all facilities).
#   Usage: julia --project=. run_georgia_hab_N.jl [N] [num_bins] [seed]   defaults 12 24 24
using Printf
const N        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const BINS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
const SEED_RUN = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 24
const DATA_PATH           = joinpath(@__DIR__, "dataset", "georgia", "georgia_habitual_sessions.csv")
const TOP_N_USERS         = N
const Q_UPPER             = 1.0
const UPPER_INCLUSIVE     = true
const A_KMAX              = 6
const G_EFF_MAX_HOURS_RUN = 4 * 24.0
NUM_BINS                  = BINS
ARTIFACTS_DIR             = joinpath(@__DIR__, "artifacts", @sprintf("georgia_hab_N%03d_b%02d_s%02d", N, BINS, SEED_RUN))
mkpath(ARTIFACTS_DIR)
@printf("▶ georgia_habitual | N=%d bins=%d seed=%d\n  artifacts: %s\n", N, BINS, SEED_RUN, ARTIFACTS_DIR)
include(joinpath(@__DIR__, "smc_core.jl"))
