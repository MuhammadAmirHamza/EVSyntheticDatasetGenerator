#!/usr/bin/env python3
# Location-specific profile graphs from the Kim commercial raw file.
#   python kim_loc_profile_plots.py "public institution" [TOP_N]
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOC   = sys.argv[1] if len(sys.argv) > 1 else 'public institution'
TOP_N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_MIN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
slug = LOC.replace(' ', '_')
OUT  = os.path.join(ROOT, 'plots', 'kim'); os.makedirs(OUT, exist_ok=True)
C, C2 = '#3b7a57', '#2c6fbb'
DOW = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

d = pd.read_csv(os.path.join(HERE, 'ChargingRecords.csv')); d.columns=[c.strip() for c in d.columns]
d = d[(d.UserID != 0) & (d.Location == LOC)].copy()
d['t0'] = pd.to_datetime(d.StartDay.astype(str)+' '+d.StartTime.astype(str), errors='coerce')
d['t1'] = pd.to_datetime(d.EndDay.astype(str)+' '+d.EndTime.astype(str), errors='coerce')
d['energy'] = pd.to_numeric(d.Demand, errors='coerce')
d = d.dropna(subset=['t0','t1','energy'])
d['dur_h'] = (d.t1 - d.t0).dt.total_seconds()/3600
d = d[(d.energy>0.5)&(d.energy<150)&(d.dur_h>=2/60)&(d.dur_h<=120)]
cnt = d.groupby('UserID').size()
keep = cnt[cnt>=N_MIN].sort_values(ascending=False).head(TOP_N).index
d = d[d.UserID.isin(keep)].copy()
d['arrival_hour'] = d.t0.dt.hour + d.t0.dt.minute/60
d['day'] = d.t0.dt.weekday.map(lambda w: DOW[w])
print(f'{LOC}: {len(d)} sessions, {d.UserID.nunique()} drivers (top {TOP_N}, >= {N_MIN})')

gaps=[]
for _,g in d.sort_values(['UserID','t0']).groupby('UserID'):
    gp=(g.t0.shift(-1)-g.t1).dt.total_seconds().dropna()/3600
    gaps.extend(gp[(gp>0)&(gp<24*14)].tolist())
gaps=np.array(gaps)

fig, ax = plt.subplots(3, 3, figsize=(19, 13))
fig.suptitle(f'Kim commercial — "{LOC}" per-driver profile ({d.UserID.nunique()} drivers, {len(d)} sessions)',
             fontsize=15, fontweight='bold')
ax[0,0].hist(d.arrival_hour,bins=24,range=(0,24),color=C,edgecolor='w'); ax[0,0].set(title='Arrival hour',xlabel='hour',ylabel='sessions',xticks=range(0,25,4))
ax[0,1].hist(d.dur_h,bins=40,range=(0,8),color=C,edgecolor='w'); ax[0,1].axvline(d.dur_h.median(),color='k',ls='--',lw=1); ax[0,1].set(title='Plug-in duration',xlabel='hours',ylabel='sessions')
ax[0,2].hist(d.energy,bins=40,range=(0,60),color=C,edgecolor='w'); ax[0,2].axvline(d.energy.median(),color='k',ls='--',lw=1); ax[0,2].set(title='Energy delivered',xlabel='kWh',ylabel='sessions')
lg=gaps[gaps>0]; ax[1,0].hist(np.log10(lg),bins=40,color=C,edgecolor='w')
for h,lab in [(12,'12h'),(24,'1d'),(72,'3d'),(168,'1w')]:
    ax[1,0].axvline(np.log10(h),color='grey',ls=':',lw=1); ax[1,0].text(np.log10(h),ax[1,0].get_ylim()[1]*0.92,lab,fontsize=8,ha='center')
ax[1,0].set(title="Idle gap between a driver's sessions",xlabel='log10 hours',ylabel='count')
spc=d.groupby('UserID').size().sort_values(ascending=False); ax[1,1].bar(range(len(spc)),spc.values,color=C); ax[1,1].set(title='Sessions per driver',xlabel='driver (rank)',ylabel='sessions')
cnt2=d.day.value_counts().reindex(DOW).fillna(0); cols=[C2 if x in('Saturday','Sunday') else C for x in DOW]
ax[1,2].bar([x[:3] for x in DOW],cnt2.values,color=cols); ax[1,2].set(title='Day-of-week (weekend=blue)',ylabel='sessions')
H=np.zeros((7,24))
for _,r in d.iterrows(): H[DOW.index(r.day),int(r.t0.hour)]+=1
im=ax[2,0].imshow(H,aspect='auto',cmap='viridis',origin='lower'); ax[2,0].set(title='Arrivals: weekday x hour',xlabel='hour',yticks=range(7),yticklabels=[x[:3] for x in DOW],xticks=range(0,24,4)); fig.colorbar(im,ax=ax[2,0],fraction=0.046)
order=d.groupby('UserID').arrival_hour.median().sort_values().index; data=[d[d.UserID==u].arrival_hour.values for u in order]
ax[2,1].boxplot(data,showfliers=False,medianprops=dict(color=C),boxprops=dict(color=C)); ax[2,1].set(title='Per-driver arrival hour (heterogeneity)',xlabel='driver (sorted by median)',ylabel='arrival hour'); ax[2,1].set_xticks([])
ch=d.groupby('UserID').ChargerID.nunique(); ax[2,2].hist(ch,bins=range(1,ch.max()+2),color=C,edgecolor='w',align='left'); ax[2,2].set(title='Distinct chargers per driver',xlabel='# chargers used',ylabel='drivers')
plt.tight_layout(rect=[0,0,1,0.97])
p1=os.path.join(OUT, f'kim_{slug}_profile.png'); plt.savefig(p1,dpi=125); plt.close()
print('saved',p1)
print('median gap %.1f h | dwell median %.2f h | arrival mean %.2f | energy median %.2f | weekend %.1f%%'
      % (np.median(gaps), d.dur_h.median(), d.arrival_hour.mean(), d.energy.median(), 100*d.day.isin(['Saturday','Sunday']).mean()))
