#!/usr/bin/env python3
# Profile graphs for the Georgia Tech weekday workplace dataset (georgia_sessions.csv).
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT  = os.path.join(ROOT, 'plots', 'georgia'); os.makedirs(OUT, exist_ok=True)
C    = '#2c6fbb'

d = pd.read_csv(os.path.join(HERE, 'georgia_sessions.csv'))
d['t0'] = pd.to_datetime(d.start_date + ' ' + d.start_time)
d['t1'] = pd.to_datetime(d.end_date + ' ' + d.end_time)
DOW = ['Monday','Tuesday','Wednesday','Thursday','Friday']

# ---- per-driver idle gaps (hours between consecutive sessions) ----
gaps = []
for _, g in d.sort_values(['user_id', 't0']).groupby('user_id'):
    gp = (g.t0.shift(-1) - g.t1).dt.total_seconds().dropna() / 3600.0
    gaps.extend(gp[(gp > 0) & (gp < 24*14)].tolist())
gaps = np.array(gaps)

fig, ax = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('Georgia Tech Workplace Charging — weekday profile (30 drivers, %d sessions)'
             % len(d), fontsize=15, fontweight='bold')

ax[0,0].hist(d.arrival_hour, bins=24, range=(0,24), color=C, edgecolor='w')
ax[0,0].set(title='Arrival hour', xlabel='hour of day', ylabel='sessions', xticks=range(0,25,4))

ax[0,1].hist(d.duration_hours, bins=40, range=(0,12), color=C, edgecolor='w')
ax[0,1].axvline(d.duration_hours.median(), color='k', ls='--', lw=1)
ax[0,1].set(title='Plug-in duration (dwell)', xlabel='hours', ylabel='sessions')

ax[0,2].hist(d.energy, bins=40, range=(0,30), color=C, edgecolor='w')
ax[0,2].axvline(d.energy.median(), color='k', ls='--', lw=1)
ax[0,2].set(title='Energy delivered', xlabel='kWh', ylabel='sessions')

lg = gaps[gaps > 0]
ax[0,3].hist(np.log10(lg), bins=40, color=C, edgecolor='w')
for h, lab in [(12,'12h'),(24,'1d'),(72,'3d'),(168,'1w')]:
    ax[0,3].axvline(np.log10(h), color='grey', ls=':', lw=1)
    ax[0,3].text(np.log10(h), ax[0,3].get_ylim()[1]*0.92, lab, fontsize=8, ha='center')
ax[0,3].set(title='Idle gap between a driver''s sessions', xlabel='log10 hours', ylabel='count')

spc = d.groupby('user_id').size().sort_values(ascending=False)
ax[1,0].bar(range(len(spc)), spc.values, color=C)
ax[1,0].set(title='Sessions per driver', xlabel='driver (rank)', ylabel='sessions')

cnt = d.day.value_counts().reindex(DOW).fillna(0)
ax[1,1].bar([x[:3] for x in DOW], cnt.values, color=C)
ax[1,1].set(title='Day-of-week (weekday-only)', ylabel='sessions')

# hour x weekday heatmap
H = np.zeros((5, 24))
for _, r in d.iterrows():
    if r.day in DOW:
        H[DOW.index(r.day), int(r.start_hour)] += 1
im = ax[1,2].imshow(H, aspect='auto', cmap='viridis', origin='lower')
ax[1,2].set(title='Arrivals: weekday x hour', xlabel='hour', yticks=range(5),
            yticklabels=[x[:3] for x in DOW], xticks=range(0,24,4))
fig.colorbar(im, ax=ax[1,2], fraction=0.046)

# per-driver arrival-hour spread (heterogeneity)
order = d.groupby('user_id').arrival_hour.median().sort_values().index
data  = [d[d.user_id == u].arrival_hour.values for u in order]
ax[1,3].boxplot(data, vert=True, showfliers=False,
                medianprops=dict(color=C), boxprops=dict(color=C))
ax[1,3].set(title='Per-driver arrival hour (heterogeneity)',
            xlabel='driver (sorted by median)', ylabel='arrival hour')
ax[1,3].set_xticks([])

plt.tight_layout(rect=[0,0,1,0.96])
p1 = os.path.join(OUT, 'georgia_weekday_profile.png')
plt.savefig(p1, dpi=130); plt.close()
print('saved', p1)

# ---- summary numbers ----
print('sessions %d | drivers %d | median gap %.1f h | mean arrival %.2f | dwell median %.2f h'
      % (len(d), d.user_id.nunique(), np.median(gaps), d.arrival_hour.mean(), d.duration_hours.median()))
