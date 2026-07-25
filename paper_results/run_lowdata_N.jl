# run_lowdata_N.jl — sparse-data sensitivity (reviewer #5): combined cohort with
# each user truncated to the chronologically FIRST n sessions.
#   Prereq: python make_lowdata_subsets.py   (writes dataset/lowdata/combined_first*.csv)
#   Usage:  julia --project=. run_lowdata_N.jl <n_sessions> [num_bins] [seed]
#           n_sessions ∈ {10, 20, 30, 50, 100};  defaults bins=24 seed=24
#   Sweep:  for n in 10 20 30 50 100; do julia --project=. run_lowdata_N.jl $n; done
using Printf
length(ARGS) >= 1 || error("usage: julia --project=. run_lowdata_N.jl <n_sessions> [num_bins] [seed]")
const NSESS    = parse(Int, ARGS[1])
const N_MIN_RUN = NSESS   # relax the >=30-session inclusion rule to n for the sparse-data study
const BINS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
const SEED_RUN = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 24
const DATA_PATH = joinpath(@__DIR__, "dataset", "lowdata", @sprintf("combined_first%03d.csv", NSESS))
isfile(DATA_PATH) || error("subset not found: $(DATA_PATH)\n  run: python make_lowdata_subsets.py")
const TOP_N_USERS         = 200           # keep every user (as in run_combined_N.jl)
const Q_UPPER             = 1.0
const UPPER_INCLUSIVE     = true
const A_KMAX              = 6
const G_EFF_MAX_HOURS_RUN = 10 * 24.0
NUM_BINS                  = BINS
ARTIFACTS_DIR             = joinpath(@__DIR__, "artifacts", @sprintf("lowdata_n%03d_b%02d_s%02d", NSESS, BINS, SEED_RUN))
mkpath(ARTIFACTS_DIR)
@printf("▶ lowdata | first n=%d sessions/user | bins=%d seed=%d\n  data: %s\n  artifacts: %s\n",
        NSESS, BINS, SEED_RUN, DATA_PATH, ARTIFACTS_DIR)
include(joinpath(@__DIR__, "smc_core.jl"))
