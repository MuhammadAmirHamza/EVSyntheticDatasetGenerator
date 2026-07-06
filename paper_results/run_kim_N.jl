# run_kim_N.jl — kim_commercial SMC. Usage: julia --project=. run_kim_N.jl <N> [num_bins] [seed]
#   defaults: N=25, bins=24, seed=24. Artifacts -> artifacts/kim_commercial_N<NNN>_b<BB>_s<SS>/
using Printf
const N        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
const BINS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
const SEED_RUN = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 24
const DATA_PATH           = joinpath(@__DIR__, "dataset", "kim-commercial", "kim_commercial_comm_wd_sessions.csv")
const TOP_N_USERS         = N
const Q_UPPER             = 1.0
const UPPER_INCLUSIVE     = true
const A_KMAX              = 6
const G_EFF_MAX_HOURS_RUN = 5 * 24.0
NUM_BINS                  = BINS
ARTIFACTS_DIR             = joinpath(@__DIR__, "artifacts", @sprintf("kim_commercial_N%03d_b%02d_s%02d", N, BINS, SEED_RUN))
mkpath(ARTIFACTS_DIR)
@printf("▶ kim_commercial | N=%d bins=%d seed=%d\n  artifacts: %s\n", N, BINS, SEED_RUN, ARTIFACTS_DIR)
include(joinpath(@__DIR__, "smc_core.jl"))
