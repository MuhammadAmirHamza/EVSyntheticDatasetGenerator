# run_norway_N.jl — norway_sorensen SMC. Usage: julia --project=. run_norway_N.jl <N> [num_bins] [seed]
#   defaults: N=100, bins=12, seed=24. Artifacts -> artifacts/norway_sorensen_N<NNN>_b<BB>_s<SS>/
using Printf
const N        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const BINS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
const SEED_RUN = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 24
const DATA_PATH           = joinpath(@__DIR__, "dataset", "norway_sorensen", "norway_sorensen_sessions.csv")
const TOP_N_USERS         = N
const Q_UPPER             = 0.98
const UPPER_INCLUSIVE     = false
const A_KMAX              = 3
const G_EFF_MAX_HOURS_RUN = 10 * 24.0
NUM_BINS                  = BINS
ARTIFACTS_DIR             = joinpath(@__DIR__, "artifacts", @sprintf("norway_sorensen_N%03d_b%02d_s%02d", N, BINS, SEED_RUN))
mkpath(ARTIFACTS_DIR)
@printf("▶ norway_sorensen | N=%d bins=%d seed=%d\n  artifacts: %s\n", N, BINS, SEED_RUN, ARTIFACTS_DIR)
include(joinpath(@__DIR__, "smc_core.jl"))
