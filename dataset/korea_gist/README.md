# South Korea — GIST charging transactions (second dataset)

Session-level EV charging dataset; we use the **residential (apartment)** subset.

- **Raw file:** `ChargingRecords.csv` (7.6 MB, 72,856 sessions, md5 `09522f5d04d44a18548ac82c2b993663`)
- **Source:** Kim et al., *Scientific Data* (2024), "A dataset for multi-faceted analysis of electric vehicle charging transactions"
  - Paper: https://www.nature.com/articles/s41597-024-02942-9
  - Data: https://doi.org/10.6084/m9.figshare.22495141.v1 (figshare)
- **Licence:** CC BY 4.0
- **Region:** South Korea (Gwangju Institute of Science and Technology operator data)
- **Period:** 2021-09-30 to 2022-09-30
- **Scale:** 72,856 sessions, 2,337 users, 2,119 chargers across mixed location types

## Raw schema (13 columns)

```
UserID, ChargerID, ChargerCompany, Location, ChargerType,
StartDay, StartTime, EndDay, EndTime, StartDatetime, EndDatetime, Duration, Demand
```
- `StartDatetime` / `EndDatetime` — e.g. `2022-09-17 22:42`
- `Demand` — delivered energy (kWh; apartment median ≈ 12.95)
- `Duration` — integer (units to confirm in preprocessing; sample values 118, 138, 73 look like minutes)
- `UserID == 0` — anonymous non-subscribers (must be dropped for per-user modelling)
- `Location` — category incl. `apartment, hotel, resort, accommodation, camping, company, public area, ...`

## Residential subset (target for the framework)

Filter: `Location == "apartment"` AND `UserID != 0`
→ **8,755 sessions, 327 users** (86 users with ≥30 sessions; 54 with ≥50; 20 with ≥100).
Apartment is the habitual home-charging analog; hotel/resort/camping are transient/opportunistic and out of the paper's scope.

## Schema mapping to the pipeline (`SMC_framework`)

| pipeline field | source |
|---|---|
| `user_id`    | `UserID` |
| `start`      | `StartDatetime` |
| `end`        | `EndDatetime` (→ `duration_hours`) |
| `energy`     | `Demand` |

## TODO
- [ ] Write `preprocess_korea_gist.{py,jl}` → filter apartment + named users, derive `duration_hours`, day-type, etc., emit `korea_gist_sessions.csv` in the Norway 16-column schema.
- [ ] Run `SMC_framework`-style pipeline on the Korean residential subset.
