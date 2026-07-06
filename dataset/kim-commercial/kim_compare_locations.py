#!/usr/bin/env python3
# Compare the 4 kept location regimes in kim_commercial_sessions.csv.
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT  = os.path.join(ROOT, 'plots', 'kim'); os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(HERE, 'kim_commercial_sessions.csv'))
d['t0'] = pd.to_datetime(d.start_date + ' ' + d.start_time)
d['t1'] = pd.to_datetime(d.end_date + ' ' + d.end_time)
DOW = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
LOCS = ['apartment', 'company', 'public area', 'public institution']
COL  = {'apartment':'#2c6fbb','company':'#b5442c','public area':'#3b7a57','public institution':'#8452a1'}

def gaps_of(g):
    out=[]
    for _,gu in g.sort_values(['user_id','t0']).groupby('user_id'):
        gp=(gu.t0.shift(-1)-gu.t1).dt.total_seconds().dropna()/3600
        out.extend(gp[(gp>0)&(gp<24*14)].tolist())
    return np.array(out)

fig, ax = plt.subplots(4, 5, figsize=(24, 16))
fig.suptitle('Kim commercial — four location regimes compared (per-driver, single-location loyal)',
             fontsize=17, fontweight='bold')
for r, loc in enumerate(LOCS):
    g = d[d.facility == loc]; c = COL[loc]; nd = g.user_id.nunique()
    ax[r,0].set_ylabel(f'{loc}\n({nd} drivers, {len(g)} sess.)', fontsize=12, fontweight='bold')
    ax[r,0].hist(g.arrival_hour, bins=24, range=(0,24), color=c, edgecolor='w')
    ax[r,0].set(xticks=range(0,25,6)); ax[r,0].axvline(g.arrival_hour.median(),color='k',ls='--',lw=1)
    ax[r,1].hist(g.duration_hours, bins=40, range=(0,10), color=c, edgecolor='w')
    ax[r,1].axvline(g.duration_hours.median(),color='k',ls='--',lw=1)
    ax[r,2].hist(g.energy, bins=40, range=(0,60), color=c, edgecolor='w')
    ax[r,2].axvline(g.energy.median(),color='k',ls='--',lw=1)
    gp=gaps_of(g); gp=gp[gp>0]
    ax[r,3].hist(np.log10(gp), bins=35, color=c, edgecolor='w')
    for h,lab in [(24,'1d'),(72,'3d'),(168,'1w')]:
        ax[r,3].axvline(np.log10(h),color='grey',ls=':',lw=1); ax[r,3].text(np.log10(h),ax[r,3].get_ylim()[1]*0.9,lab,fontsize=8,ha='center')
    H=np.zeros((7,24))
    for _,row in g.iterrows(): H[DOW.index(row.day),int(row.start_hour)]+=1
    im=ax[r,4].imshow(H,aspect='auto',cmap='viridis',origin='lower')
    ax[r,4].set(yticks=range(7),yticklabels=[x[:2] for x in DOW],xticks=range(0,24,6))
    if r==0:
        ax[0,0].set_title('Arrival hour',fontsize=13); ax[0,1].set_title('Plug-in duration (h)',fontsize=13)
        ax[0,2].set_title('Energy (kWh)',fontsize=13); ax[0,3].set_title('Idle gap (log10 h)',fontsize=13)
        ax[0,4].set_title('Weekday x hour',fontsize=13)
plt.tight_layout(rect=[0,0,1,0.97])
p1=os.path.join(OUT,'kim_four_locations_compare.png'); plt.savefig(p1,dpi=115); plt.close()
print('saved',p1)

# overlay figure: arrival + idle-gap curves for direct comparison
fig2,ax2=plt.subplots(1,3,figsize=(20,5.5))
for loc in LOCS:
    g=d[d.facility==loc]; c=COL[loc]
    h,_=np.histogram(g.arrival_hour,bins=24,range=(0,24),density=True)
    ax2[0].plot(np.arange(24)+.5,h,color=c,lw=2,label=loc)
    h2,e2=np.histogram(g.duration_hours,bins=40,range=(0,8),density=True)
    ax2[1].plot((e2[:-1]+e2[1:])/2,h2,color=c,lw=2,label=loc)
    gp=gaps_of(g); gp=gp[gp>0]
    h3,e3=np.histogram(np.log10(gp),bins=35,density=True)
    ax2[2].plot((e3[:-1]+e3[1:])/2,h3,color=c,lw=2,label=loc)
ax2[0].set(title='Arrival hour (density)',xlabel='hour',xticks=range(0,25,4))
ax2[1].set(title='Plug-in duration (density)',xlabel='hours')
ax2[2].set(title='Idle gap (density)',xlabel='log10 hours')
for a in ax2: a.legend(fontsize=9)
for h,lab in [(24,'1d'),(72,'3d'),(168,'1w')]: ax2[2].axvline(np.log10(h),color='grey',ls=':',lw=1)
plt.tight_layout()
p2=os.path.join(OUT,'kim_four_locations_overlay.png'); plt.savefig(p2,dpi=125); plt.close()
print('saved',p2)
# print summary table
print('\nloc | drivers | arr_med | dwell_med | energy_med | gap_med(h) | weekend%')
for loc in LOCS:
    g=d[d.facility==loc]; gp=gaps_of(g)
    print('%-19s %3d   %5.1f   %6.2f    %6.1f    %6.1f    %4.1f'%(loc,g.user_id.nunique(),
        g.arrival_hour.median(),g.duration_hours.median(),g.energy.median(),np.median(gp),
        100*g.day.isin(['Saturday','Sunday']).mean()))
