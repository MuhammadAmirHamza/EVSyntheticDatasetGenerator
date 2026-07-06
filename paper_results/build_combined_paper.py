"""Build paper_results combined pool from the FINALIZED 4 cohorts (raw_clean is
bin-independent). Source-prefixed user_ids; facility=src tags regime for L4 split."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; A=ROOT/'artifacts'
SOURCES={ 'norway':  A/'norway_N100'/'raw_clean.csv',          # residential NO
          'gist':    A/'korea_gist_N030'/'raw_clean.csv',      # residential KR
          'kim':     A/'kim_commercial_N025'/'raw_clean.csv' } # commercial KR (wd)
OUT=ROOT/'dataset'/'combined'/'combined_sessions.csv'
KEEP=['duration_hours','energy','day_type','start_date','start_time','end_date','end_time']
frames=[]
for src,f in SOURCES.items():
    d=pd.read_csv(f).dropna(subset=['abs_start','abs_end','duration_hours','energy','user_id'])
    d=d[['user_id','abs_start','abs_end']+KEEP].copy()
    d['src']=src; d['user_id']=src+'|'+d['user_id'].astype(str)
    frames.append(d); print(f"  {src:8s}: {d.user_id.nunique():3d} users, {len(d):6d} sessions")
c=pd.concat(frames,ignore_index=True)
ts=pd.to_datetime(c['abs_start']); te=pd.to_datetime(c['abs_end'])
c['start_date']=ts.dt.strftime('%Y-%m-%d'); c['start_time']=ts.dt.strftime('%H:%M:%S')
c['end_date']=te.dt.strftime('%Y-%m-%d'); c['end_time']=te.dt.strftime('%H:%M:%S')
c['abs_start']=ts.dt.strftime('%Y-%m-%dT%H:%M:%S'); c['abs_end']=te.dt.strftime('%Y-%m-%dT%H:%M:%S')
c['start_hour']=ts.dt.hour; c['end_hour']=te.dt.hour
c['arrival_hour']=ts.dt.hour+ts.dt.minute/60.0; c['duration_minutes']=c['duration_hours']*60
c['location']='combined'; c['facility']=c['src']
c['user_id_num']=c.user_id.map({u:i for i,u in enumerate(dict.fromkeys(c.user_id))})
c['session_id']=['C%d'%(i+1) for i in range(len(c))]
c.to_csv(OUT,index=False)
print(f"\nwrote {OUT.name}: {len(c)} sessions, {c.user_id.nunique()} users")
print("  per-source:",c.groupby('src').user_id.nunique().to_dict())
print("  day_type:",c.day_type.value_counts().to_dict())
