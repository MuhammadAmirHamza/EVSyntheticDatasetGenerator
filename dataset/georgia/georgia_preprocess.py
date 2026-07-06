#!/usr/bin/env python3
# =============================================================================
# Georgia Tech Workplace Charging (Asensio et al., Nature Sci. Data 2021)
#   -> SPARC 16-column schema, keyed by per-DRIVER userId.
# =============================================================================
# Genuine per-DRIVER workplace data (ACN-Caltech userID coverage was ~1% and
# forced a per-station key). facilityType {1:manufacturing,2:office,3:R&D,4:other}.
# Quirks handled:
#   * Dates anonymised to year 0014/0015 (below pandas range) -> shift +2000;
#     preserves time-of-day, ordering, inter-session spacing (gaps stay valid).
#   * Shifted weekday is not real -> day/day_type from dataset's true `weekday`.
#   * duration = dwell (ended-created) = connection time (NOT chargeTimeHrs).
#   * WEEKDAYS_ONLY: workplace routine; ~3% weekend share too sparse -> Mon-Fri.
# Output: dataset/georgia/georgia_sessions.csv  ->  julia run_smc.jl georgia
# =============================================================================
import os
import pandas as pd

N_MIN = 30
TOP_N = 30
WEEKDAYS_ONLY = True
ENERGY_MIN, ENERGY_MAX = 0.5, 150.0
DUR_MIN_H,  DUR_MAX_H  = 2/60, 120.0

MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
WKMAP  = {'Mon':'Monday','Tue':'Tuesday','Tues':'Tuesday','Wed':'Wednesday',
          'Thu':'Thursday','Thur':'Thursday','Thurs':'Thursday','Fri':'Friday',
          'Sat':'Saturday','Sun':'Sunday'}
FACILITY = {1:'manufacturing', 2:'office', 3:'R&D', 4:'other'}
HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, 'georgia_raw.csv')


def fix_year(s):
    s = str(s)
    return str(int(s[:4]) + 2000) + s[4:]


def main():
    df = pd.read_csv(RAW)
    df = df[df['userId'].notna()].copy()
    df['t0'] = pd.to_datetime(df['created'].map(fix_year), errors='coerce')
    df['t1'] = pd.to_datetime(df['ended'].map(fix_year),   errors='coerce')
    df['energy'] = pd.to_numeric(df['kwhTotal'], errors='coerce')
    df = df.dropna(subset=['t0', 't1', 'energy', 'userId'])
    df['dur_h'] = (df['t1'] - df['t0']).dt.total_seconds() / 3600.0
    df = df[(df.energy > ENERGY_MIN) & (df.energy < ENERGY_MAX) &
            (df.dur_h >= DUR_MIN_H) & (df.dur_h <= DUR_MAX_H)]

    if WEEKDAYS_ONLY:
        wk = df['weekday'].map(lambda w: WKMAP.get(str(w).strip(), ''))
        n0 = len(df)
        df = df[~wk.isin(['Saturday', 'Sunday'])]
        print("weekday-only filter: dropped %d weekend sessions (%d -> %d)"
              % (n0 - len(df), n0, len(df)))

    cnt  = df.groupby('userId').size()
    keep = cnt[cnt >= N_MIN].sort_values(ascending=False).head(TOP_N).index
    df   = df[df.userId.isin(keep)].sort_values(['userId', 't0']).reset_index(drop=True)
    print("kept %d sessions from %d drivers (>= %d sessions, top %d)"
          % (len(df), df.userId.nunique(), N_MIN, TOP_N))

    ids = list(dict.fromkeys(df.userId)); uidnum = {u: i for i, u in enumerate(ids)}
    day_full = df['weekday'].map(lambda w: WKMAP.get(str(w).strip(), 'Monday'))
    is_wknd  = day_full.isin(['Saturday', 'Sunday'])
    arrival  = df.t0.dt.hour + df.t0.dt.minute / 60.0

    out = pd.DataFrame({
        'location':         ['US_workplace_GT'] * len(df),
        'user_id':          ['GT_' + str(int(u)) for u in df.userId],
        'session_id':       ['GT' + str(i) for i in range(1, len(df) + 1)],
        'duration_hours':   df.dur_h.values,
        'energy':           df.energy.values,
        'arrival_hour':     arrival.values,
        'start_date':       df.t0.dt.strftime('%Y-%m-%d'), 'start_time': df.t0.dt.strftime('%H:%M:%S'),
        'end_date':         df.t1.dt.strftime('%Y-%m-%d'), 'end_time':   df.t1.dt.strftime('%H:%M:%S'),
        'start_hour':       df.t0.dt.hour, 'end_hour': df.t1.dt.hour,
        'month':            [MONTHS[mo - 1] for mo in df.t0.dt.month],
        'day':              day_full.values,
        'duration_minutes': df.dur_h.values * 60,
        'day_type':         ['Weekend' if w else 'Weekday' for w in is_wknd],
        'facility':         [FACILITY.get(int(f), 'unknown') for f in df.facilityType],
        'user_id_num':      [uidnum[u] for u in df.userId],
    })
    fp = os.path.join(HERE, 'georgia_sessions.csv'); out.to_csv(fp, index=False)

    spc = out.groupby('user_id').size()
    print("\nSaved -> %s" % fp)
    print("FINAL: %d sessions, %d drivers (median %d, range %d-%d sessions/driver)"
          % (len(out), out.user_id.nunique(), int(spc.median()), spc.min(), spc.max()))
    print("weekend share: %.1f%%  |  facility mix: %s"
          % (100 * is_wknd.mean(), out.facility.value_counts().to_dict()))
    print("arrival_hour median %.1f | duration_h median %.2f | energy median %.2f"
          % (out.arrival_hour.median(), out.duration_hours.median(), out.energy.median()))


if __name__ == '__main__':
    main()
