# Datasets

Source EV charging session datasets used to fit and validate the per-user
semi-Markov framework. Each dataset lives in its own subfolder.

| Folder | Region | Users | Sessions | Status |
|---|---|---|---|---|
| `norway_sorensen/` | Norway 🇳🇴 | 264 (top-100 used) | ~34k raw | primary (in paper) |
| `uk_electric_nation/` | UK 🇬🇧 | ~700 | 2M+ charging hours | second dataset (to add) |

## Schema expected by the pipeline

The framework only needs the minimal per-session tuple
`(user_id, start_datetime, end_datetime → duration, energy)`; every other
column (`day_type`, `start_hour`, `arrival_hour`, ...) is derived in Stage 1
(`SMC_framework.jl`). Any new dataset must be reduced to this schema before it
can be fed in via `DATA_PATH`.

Current canonical columns (see `norway_sorensen/norway_sorensen_sessions.csv`):

```
location, user_id, session_id, duration_hours, energy,
start_date, start_time, end_date, end_time, start_hour, end_hour,
month, day, duration_minutes, day_type, user_id_num
```
