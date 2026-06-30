# =============================================================================
# Preprocess: Korea GIST charging transactions  →  residential session table
# =============================================================================
# Extracts the RESIDENTIAL (apartment) subset of the Korea GIST dataset,
# removes physically-implausible outliers, and writes a clean session table in
# the SAME 16-column schema as the Norwegian dataset
# (dataset/norway_sorensen/norway_sorensen_sessions.csv) so the SMC framework
# ingests it with no code changes.
#
#   Input  : dataset/korea_gist/ChargingRecords.csv      (raw, 72,856 sessions)
#   Output : dataset/korea_gist/korea_gist_sessions.csv
#
# Duration is recomputed from EndDatetime − StartDatetime (the raw `Duration`
# column has ambiguous units), consistent with the Norwegian pipeline.
# =============================================================================

using CSV, DataFrames, Dates, Printf, Statistics

const HERE        = @__DIR__
const INPUT_PATH  = joinpath(HERE,  "ChargingRecords.csv")
const OUTPUT_PATH = joinpath(HERE,  "korea_gist_sessions.csv")

# ----- Filters (mirror the Norwegian dataset-level cleaning) ------------------
const RESIDENTIAL_SET = Set(["apartment"])  # habitual home charging only
const ENERGY_MIN_KWH  = 0.5      # drop aborted / near-zero sessions
const ENERGY_MAX_KWH  = 150.0    # sanity cap on implausible energy
const DUR_MIN_HOURS   = 2 / 60   # 2 minutes  (spurious connections)
const DUR_MAX_HOURS   = 120.0    # 5 days     (pathological outliers)

# ----- Helpers ---------------------------------------------------------------
const DT_FORMATS = (dateformat"yyyy-mm-dd HH:MM:SS", dateformat"yyyy-mm-dd HH:MM")

function parse_dt(x)::Union{DateTime,Missing}
    x === missing && return missing
    s = strip(String(x)); isempty(s) && return missing
    for f in DT_FORMATS
        try; return DateTime(s, f); catch; end
    end
    return missing
end

tonum(x)::Union{Float64,Missing} =
    x === missing ? missing :
    x isa Number  ? Float64(x) :
    something(tryparse(Float64, strip(String(x))), missing)

# ----- Load ------------------------------------------------------------------
println("─"^64); println("Korea GIST preprocessing"); println("─"^64)
raw = CSV.read(INPUT_PATH, DataFrame;
               types = Dict(:StartDatetime => String, :EndDatetime => String,
                            :Location => String, :Demand => String))
@printf("  Loaded                  : %d sessions, %d users, %d chargers\n",
        nrow(raw), length(unique(raw.UserID)), length(unique(raw.ChargerID)))

# ----- Residential + named-user filter ---------------------------------------
keep = [ (lowercase(strip(coalesce(l, ""))) in RESIDENTIAL_SET) for l in raw.Location ] .&
       (raw.UserID .!= 0)
df = raw[keep, :]
@printf("  Residential (apartment) : %d sessions, %d users\n",
        nrow(df), length(unique(df.UserID)))

# ----- Parse datetimes & energy, derive duration -----------------------------
df.t_start = parse_dt.(df.StartDatetime)
df.t_end   = parse_dt.(df.EndDatetime)
df.energy  = tonum.(df.Demand)
df.dur_h   = [ (s === missing || e === missing) ? missing : (e - s).value / 3.6e6
               for (s, e) in zip(df.t_start, df.t_end) ]   # ms → hours

n0 = nrow(df)
df = df[.!ismissing.(df.t_start) .& .!ismissing.(df.t_end) .&
        .!ismissing.(df.energy)  .& .!ismissing.(df.dur_h), :]
@printf("  Dropped unparsable      : %d\n", n0 - nrow(df))

# narrow types now that missings are gone
ts = Vector{DateTime}(df.t_start)
te = Vector{DateTime}(df.t_end)
en = Vector{Float64}(df.energy)
du = Vector{Float64}(df.dur_h)

# diagnostic: what unit is the raw `Duration` column? (non-essential)
try
    rd = tonum.(df.Duration); v = .!ismissing.(rd) .& (du .> 0)
    if any(v)
        ratio = median(Float64.(rd[v]) ./ (du[v] .* 60))
        unit = isapprox(ratio, 1; atol = 0.15)    ? "minutes" :
               isapprox(ratio, 1/60; atol = 0.02) ? "hours"   :
               @sprintf("%.3f×min (unknown)", ratio)
        println("  (raw 'Duration' column ≈ ", unit, ")")
    end
catch; end

# ----- Outlier removal (physical bounds) -------------------------------------
mask = (en .> ENERGY_MIN_KWH) .& (en .< ENERGY_MAX_KWH) .&
       (du .>= DUR_MIN_HOURS) .& (du .<= DUR_MAX_HOURS)
df = df[mask, :]; ts = ts[mask]; te = te[mask]; en = en[mask]; du = du[mask]
@printf("  After outlier removal   : %d sessions (dropped %d)\n",
        nrow(df), count(.!mask))

# ----- Build 16-column schema (matches norway_sorensen_sessions.csv) ----------
uids   = unique(df.UserID)
uidnum = Dict(u => i - 1 for (i, u) in enumerate(uids))   # 0-based, like Norway

out = DataFrame(
    location         = string.("CH", df.ChargerID),
    user_id          = string.("KR_U", df.UserID),
    session_id       = [ "KR$(i)" for i in 1:nrow(df) ],
    duration_hours   = du,
    energy           = en,
    start_date       = Dates.format.(ts, dateformat"yyyy-mm-dd"),
    start_time       = Dates.format.(ts, dateformat"HH:MM:SS"),
    end_date         = Dates.format.(te, dateformat"yyyy-mm-dd"),
    end_time         = Dates.format.(te, dateformat"HH:MM:SS"),
    start_hour       = hour.(ts),
    end_hour         = hour.(te),
    month            = monthname.(ts),
    day              = dayname.(ts),
    duration_minutes = du .* 60,
    day_type         = [ dayofweek(t) >= 6 ? "Weekend" : "Weekday" for t in ts ],
    user_id_num      = [ uidnum[u] for u in df.UserID ],
)

# ----- Save ------------------------------------------------------------------
CSV.write(OUTPUT_PATH, out)
spc = combine(groupby(out, :user_id), nrow => :n).n
@printf("  Saved → dataset/korea_gist/korea_gist_sessions.csv\n")
@printf("  FINAL: %d sessions, %d users (median %.0f, range %d–%d sessions/user)\n",
        nrow(out), length(uids), median(spc), minimum(spc), maximum(spc))
@printf("  Median duration %.2f h | median energy %.2f kWh | weekday/weekend %d/%d\n",
        median(out.duration_hours), median(out.energy),
        count(==("Weekday"), out.day_type), count(==("Weekend"), out.day_type))
println("─"^64)
