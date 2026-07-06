"""
Rebuild dataset/combined/combined_sessions.csv from the FIVE selected cohorts
(exactly the users SPARC used in each case study), pooled into one population:

    norway 100 + korea 30 + pecan 10 + georgia 30 + kim 30  = 200 users

Sources are each dataset's SPARC-selected raw_clean.csv (post quality-filter,
post top-N selection), so the combined population corresponds to ONLY the
selected N users per dataset. user_id is source-prefixed to guarantee global
uniqueness; a 'facility' column records the source so L4 can be split by regime.

    python dataset/combined/build_combined.py        # run from project root
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'norway':  ROOT/'artifacts'/'norway_sorensen'/'raw_clean.csv',
    'korea':   ROOT/'artifacts'/'korea_gist'/'raw_clean.csv',
    'pecan':   ROOT/'artifacts'/'pecanstreet'/'raw_clean.csv',
    'georgia': ROOT/'artifacts'/'georgia_module'/'raw_clean.csv',
    'kim':     ROOT/'artifacts'/'kim_commercial_module'/'raw_clean.csv',
}
OUT = ROOT/'dataset'/'combined'/'combined_sessions.csv'
KEEP = ['duration_hours', 'energy', 'day_type', 'start_date', 'start_time',
        'end_date', 'end_time']

frames = []
for src, f in SOURCES.items():
    d = pd.read_csv(f).dropna(subset=['abs_start', 'abs_end', 'duration_hours', 'energy', 'user_id'])
    d = d[['user_id', 'abs_start', 'abs_end'] + KEEP].copy()
    d['src'] = src
    d['user_id'] = src + '|' + d['user_id'].astype(str)     # globally-unique id
    frames.append(d)
    print(f"  {src:8s}: {d.user_id.nunique():3d} users, {len(d):5d} sessions")

c = pd.concat(frames, ignore_index=True)
ts = pd.to_datetime(c['abs_start']); te = pd.to_datetime(c['abs_end'])
c['start_date'] = ts.dt.strftime('%Y-%m-%d'); c['start_time'] = ts.dt.strftime('%H:%M:%S')
c['end_date']   = te.dt.strftime('%Y-%m-%d'); c['end_time']   = te.dt.strftime('%H:%M:%S')
c['abs_start']  = ts.dt.strftime('%Y-%m-%dT%H:%M:%S')
c['abs_end']    = te.dt.strftime('%Y-%m-%dT%H:%M:%S')
c['start_hour'] = ts.dt.hour; c['end_hour'] = te.dt.hour
c['arrival_hour'] = ts.dt.hour + ts.dt.minute/60.0
c['duration_minutes'] = c['duration_hours'] * 60
c['location'] = 'combined'
c['facility'] = c['src']                                    # source/regime tag
c['user_id_num'] = c.user_id.map({u: i for i, u in enumerate(dict.fromkeys(c.user_id))})
c['session_id']  = ['C%d' % (i + 1) for i in range(len(c))]

c.to_csv(OUT, index=False)
print(f"\nwrote {OUT}\n  {len(c)} sessions, {c.user_id.nunique()} users")
print("  per-source users:", c.groupby('src').user_id.nunique().to_dict())
print("  day_type mix:", c.day_type.value_counts().to_dict())
