#!/usr/bin/env python3
# =============================================================================
# Kim et al. 2024 — commercial EV charging transactions (Nature Sci. Data,
# figshare 22495141, CC BY 4.0) -> SPARC 16-col schema, keyed per-DRIVER UserID.
# =============================================================================
# 1-year commercial network (Sep 2021-Sep 2022): 72,856 sessions, 2,337
# subscribers (RF-card UserID) + walk-ins (UserID=0), 2,119 chargers, 14 Location
# categories. Notes:
#   * UserID = 0 (non-subscribed walk-ins) dropped.
#   * Real timestamps -> weekday/day_type direct; both day types kept.
#   * duration = EndTime-StartTime (connection time); Demand is energy (kWh).
#   * SINGLE_LOCATION_ONLY: keep only "location-loyal" drivers (charge at exactly
#     ONE Location type) -> location-pure entities with full histories.
#   * ALLOWED_LOCATIONS: restrict to the four location types that exhibit genuine
#     recurring patterns and have enough loyal drivers to model.
# Output: dataset/kim-commercial/kim_commercial_sessions.csv -> julia run_smc.jl kim
# =============================================================================
import os
import pandas as pd

N_MIN = 30
TOP_N = 30                 # keep top-30 drivers by session count
SINGLE_LOCATION_ONLY = True
ALLOWED_LOCATIONS = ['apartment', 'company', 'public area', 'public institution']
ENERGY_MIN, ENERGY_MAX = 0.5, 150.0
DUR_MIN_H,  DUR_MAX_H  = 2/60, 120.0

MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, 'ChargingRecords.csv')


def main():
    d = pd.read_csv(RAW); d.columns = [c.strip() for c in d.columns]
    d = d[d.UserID != 0].copy()
    d['t0'] = pd.to_datetime(d.StartDay.astype(str) + ' ' + d.StartTime.astype(str), errors='coerce')
    d['t1'] = pd.to_datetime(d.EndDay.astype(str)   + ' ' + d.EndTime.astype(str),   errors='coerce')
    d['energy'] = pd.to_numeric(d.Demand, errors='coerce')
    d = d.dropna(subset=['t0', 't1', 'energy', 'UserID'])
    d['dur_h'] = (d.t1 - d.t0).dt.total_seconds() / 3600.0
    d = d[(d.energy > ENERGY_MIN) & (d.energy < ENERGY_MAX) &
          (d.dur_h >= DUR_MIN_H) & (d.dur_h <= DUR_MAX_H)]

    if SINGLE_LOCATION_ONLY:
        nloc = d.groupby('UserID').Location.nunique()
        d = d[d.UserID.isin(nloc[nloc == 1].index)]
    if ALLOWED_LOCATIONS:
        d = d[d.Location.isin(ALLOWED_LOCATIONS)]
        print("restricted to locations %s" % ALLOWED_LOCATIONS)

    cnt  = d.groupby('UserID').size().sort_values(ascending=False)
    keep = cnt[cnt >= N_MIN]
    if TOP_N and TOP_N > 0:
        keep = keep.head(TOP_N)
    d = d[d.UserID.isin(keep.index)].sort_values(['UserID', 't0']).reset_index(drop=True)
    print("kept %d sessions from %d drivers (>= %d sessions)" % (len(d), d.UserID.nunique(), N_MIN))

    loc_of = d.groupby('UserID').Location.first()
    ids = list(dict.fromkeys(d.UserID)); uidnum = {u: i for i, u in enumerate(ids)}
    wkd = d.t0.dt.weekday
    arrival = d.t0.dt.hour + d.t0.dt.minute / 60.0

    out = pd.DataFrame({
        'location':         ['KR_commercial'] * len(d),
        'user_id':          ['KR_' + str(int(u)) for u in d.UserID],
        'session_id':       ['KR' + str(i) for i in range(1, len(d) + 1)],
        'duration_hours':   d.dur_h.values,
        'energy':           d.energy.values,
        'arrival_hour':     arrival.values,
        'start_date':       d.t0.dt.strftime('%Y-%m-%d'), 'start_time': d.t0.dt.strftime('%H:%M:%S'),
        'end_date':         d.t1.dt.strftime('%Y-%m-%d'), 'end_time':   d.t1.dt.strftime('%H:%M:%S'),
        'start_hour':       d.t0.dt.hour, 'end_hour': d.t1.dt.hour,
        'month':            [MONTHS[m - 1] for m in d.t0.dt.month],
        'day':              [DAYS[w] for w in wkd],
        'duration_minutes': d.dur_h.values * 60,
        'day_type':         ['Weekend' if w >= 5 else 'Weekday' for w in wkd],
        'facility':         [loc_of[u] for u in d.UserID],
        'user_id_num':      [uidnum[u] for u in d.UserID],
    })
    fp = os.path.join(HERE, 'kim_commercial_sessions.csv'); out.to_csv(fp, index=False)

    spc = out.groupby('user_id').size()
    print("\nSaved -> %s" % fp)
    print("FINAL: %d sessions, %d drivers (median %d, range %d-%d sessions/driver)"
          % (len(out), out.user_id.nunique(), int(spc.median()), spc.min(), spc.max()))
    print("weekend share: %.1f%%" % (100 * (wkd >= 5).mean()))
    print("drivers by location: %s" % loc_of.value_counts().to_dict())


if __name__ == '__main__':
    main()
