# run_de_N.jl — run a cohort with the duration–energy dependence (smc_core_de.jl).
#
#   julia --project=. run_de_N.jl <cohort> <N> <bins> <seed>
#   cohort: norway | korea | kim | acn | combined_all      (same arguments as run_<cohort>_N.jl)
#   e.g.    julia --project=. run_de_N.jl combined_all 200 24 1
#
# It reuses run_<cohort>_N.jl unchanged, with two substitutions made in memory:
# the core file (smc_core.jl -> smc_core_de.jl) and the output folder
# (artifacts/ -> artifacts_de/).  Nothing in artifacts/ is read or written.
length(ARGS) >= 1 || error("usage: julia --project=. run_de_N.jl <cohort> <N> <bins> <seed>")
const DE_COHORT = popfirst!(ARGS)                     # the cohort script reads N, bins, seed from ARGS
const DE_SCRIPT = joinpath(@__DIR__, "run_$(DE_COHORT)_N.jl")
isfile(DE_SCRIPT) || error("no such cohort script: $DE_SCRIPT")
de_src = read(DE_SCRIPT, String)
occursin("\"smc_core.jl\"", de_src) && occursin("\"artifacts\"", de_src) ||
    error("unexpected layout of $DE_SCRIPT")
de_src = replace(de_src, "\"smc_core.jl\"" => "\"smc_core_de.jl\"", "\"artifacts\"" => "\"artifacts_de\"")
include_string(Main, de_src, DE_SCRIPT)
