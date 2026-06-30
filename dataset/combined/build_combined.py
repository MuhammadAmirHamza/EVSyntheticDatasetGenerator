"""
Rebuild dataset/combined/combined_sessions.csv from the three exact selected
profiles (artifacts/<dataset>/raw_clean.csv).

Date/time columns are normalized to canonical, fraction-free strings
(YYYY-MM-DD / HH:MM:SS) so the Julia framework's DateTime parser does not trip
on mixed sub-second formats coming out of raw_clean.csv.

    python dataset/combined/build_combined.py        # run from project root
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # project root (…/SDG)
SOURCES = [ROOT / 'artifacts' / 'norway_sorensen' / 'raw_clean.csv',
           ROOT / 'artifacts' / 'korea_gist'      / 'raw_clean.csv',
           ROOT / 'artifacts' / 'pecanstreet'     / 'raw_clean.csv']
OUT = ROOT / 'dataset' / 'combined' / 'combined_sessions.csv'

frames = [pd.read_csv(f).dropna(subset=['abs_start', 'abs_end', 'duration_hours',
                                        'energy', 'user_id']) for f in SOURCES]
c = pd.concat(frames, ignore_index=True)

ts = pd.to_datetime(c['abs_start']); te = pd.to_datetime(c['abs_end'])
c['start_date'] = ts.dt.strftime('%Y-%m-%d'); c['start_time'] = ts.dt.strftime('%H:%M:%S')
c['end_date']   = te.dt.strftime('%Y-%m-%d'); c['end_time']   = te.dt.strftime('%H:%M:%S')
c['abs_start']  = ts.dt.strftime('%Y-%m-%dT%H:%M:%S')
c['abs_end']    = te.dt.strftime('%Y-%m-%dT%H:%M:%S')
c['start_hour'] = ts.dt.hour; c['end_hour'] = te.dt.hour

# globally dense ids; user_id strings are already unique across sources
c['user_id_num'] = c.user_id.map({u: i for i, u in enumerate(dict.fromkeys(c.user_id))})
c['session_id']  = ['C%d' % (i + 1) for i in range(len(c))]

c.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(c)} sessions, {c.user_id.nunique()} users")
