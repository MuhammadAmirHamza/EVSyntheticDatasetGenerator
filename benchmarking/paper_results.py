"""
Paper results: the sequential + per-user story (combined dataset).

Baselines: GMMNet (Li 2024) and EV-SDG. (Vine copula dropped.)

Figure (plots/paper/heterogeneity_and_sequence.png), 2x2:
  A  idle-gap distribution        real vs Ours  (baselines emit NO gap)
  B  per-user median arrival      real (x) vs generated (y): Ours tracks the diagonal;
  C  per-user median duration     the pooled baselines assign the SAME population value
  D  per-user median energy       to every user -> a flat line (no heterogeneity).

Table (plots/paper/results_table.csv): sequential + per-user MAE (where we lead) on top,
marginal/dependence (on par with SOTA) below.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT/'plots'/'paper'; OUT.mkdir(parents=True, exist_ok=True)

REAL = pd.read_csv(ROOT/'dataset'/'combined'/'combined_sessions.csv').dropna(
    subset=['abs_start','abs_end','duration_hours','energy','user_id'])
OURS = pd.read_csv(ROOT/'artifacts'/'combined'/'sim_sessions.csv')
GMM  = pd.read_csv(HERE/'gmmnet'/'results'/'gmmnet_combined.csv')
EV   = pd.read_csv(HERE/'evsdg'/'results'/'EVSDG_combined_generated.csv')

VARS = ['arrival','duration','energy']
# unify column access
R = {'arrival':REAL['arrival_hour'].to_numpy(float),'duration':REAL['duration_hours'].to_numpy(float),
     'energy':REAL['energy'].to_numpy(float),'user':REAL['user_id'].to_numpy()}
O = {'arrival':OURS['arrival_hour'].to_numpy(float),'duration':OURS['duration_h'].to_numpy(float),
     'energy':OURS['energy_kwh'].to_numpy(float),'user':OURS['user_id'].to_numpy()}
G = {'arrival':GMM['arrival_hour'].to_numpy(float),'duration':GMM['duration_h'].to_numpy(float),
     'energy':GMM['energy_kwh'].to_numpy(float)}
E = {'arrival':EV['Arrival'].to_numpy(float),'duration':EV['Connected_time'].to_numpy(float),
     'energy':EV['Energy_required'].to_numpy(float)}

# ---- real & ours per-user medians ----
def per_user_median(d):
    df = pd.DataFrame({'u':d['user'],'arrival':d['arrival'],'duration':d['duration'],'energy':d['energy']})
    return df.groupby('u').median()
rm = per_user_median(R); om = per_user_median(O)
users = [u for u in rm.index if u in om.index]
rm = rm.loc[users]; om = om.loc[users]
pool = {'GMMNet':{v:np.median(G[v]) for v in VARS}, 'EV-SDG':{v:np.median(E[v]) for v in VARS}}

# ---- real idle gap & ours gap ----
t0 = pd.to_datetime(REAL['abs_start']); t1 = pd.to_datetime(REAL['abs_end'])
rr = REAL.assign(_t0=t0,_t1=t1); gaps=[]
for _,g in rr.sort_values('_t0').groupby('user_id'):
    gp=(g['_t0'].shift(-1)-g['_t1']).dt.total_seconds().to_numpy()[:-1]/3600.0; gaps.append(gp[gp>=0])
real_gap=np.concatenate(gaps); ours_gap=OURS['gap_h'].to_numpy(float)

# ================= FIGURE =================
fig,ax=plt.subplots(2,2,figsize=(11,8.4)); ax=ax.ravel()
COL={'Ours':'#1f77b4','GMMNet':'#2ca02c','EV-SDG':'#d62728'}
# A: idle gap
b=np.linspace(0,72,40)
ax[0].hist(np.clip(real_gap,0,72),bins=b,density=True,histtype='step',color='k',lw=2.4,label='Real')
ax[0].hist(np.clip(ours_gap,0,72),bins=b,density=True,histtype='step',color=COL['Ours'],lw=2.0,label='Ours (SMC)')
ax[0].set_title('(A) Idle gap between sessions',fontweight='bold')
ax[0].set_xlabel('Idle gap (h)'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=9)
ax[0].text(0.97,0.55,'GMMNet & EV-SDG:\nno gap (independent\nsessions)',transform=ax[0].transAxes,
           ha='right',va='top',fontsize=8.5,color='#555',bbox=dict(boxstyle='round',fc='#f4f4f4',ec='#bbb'))
# B,C,D per-user median scatter
labels={'arrival':('(B) Per-user median arrival','Real (h)','Generated (h)',(0,24)),
        'duration':('(C) Per-user median duration','Real (h)','Generated (h)',(0,30)),
        'energy':('(D) Per-user median energy','Real (kWh)','Generated (kWh)',(0,30))}
for i,v in enumerate(VARS):
    a=ax[i+1]; ttl,xl,yl,lim=labels[v]
    a.plot(lim,lim,'k--',lw=1,alpha=0.6,label='ideal (y=x)')
    a.scatter(rm[v],om[v],s=18,color=COL['Ours'],alpha=0.7,label='Ours (per-user)')
    a.axhline(pool['GMMNet'][v],color=COL['GMMNet'],lw=1.8,ls='-',label='GMMNet (pooled)')
    a.axhline(pool['EV-SDG'][v],color=COL['EV-SDG'],lw=1.8,ls=':',label='EV-SDG (pooled)')
    a.set_title(ttl,fontweight='bold'); a.set_xlabel(xl); a.set_ylabel(yl)
    a.set_xlim(lim); a.set_ylim(lim); a.legend(fontsize=7.5,loc='upper left')
fig.suptitle('Per-user heterogeneity and sequential structure (combined, 140 users)',fontweight='bold',fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/'heterogeneity_and_sequence.png',dpi=140); plt.close(fig)

# ================= TABLE =================
def kl(p,q): p=p+1e-9;q=q+1e-9;p/=p.sum();q/=q.sum();return float(np.sum(p*np.log(p/q)))
def Hh(x,lo,hi,nb): h,_=np.histogram(np.clip(np.asarray(x,float),lo,hi),bins=np.linspace(lo,hi,nb+1)); return h.astype(float)
def per_user_mae(gen_pool_or_ours, is_ours):
    out={}
    for v in VARS:
        if is_ours: err=np.abs(rm[v].to_numpy()-om[v].to_numpy())
        else:       err=np.abs(rm[v].to_numpy()-gen_pool_or_ours[v])
        out[v]=float(np.mean(err))
    return out
mae_ours=per_user_mae(None,True); mae_gmm=per_user_mae(pool['GMMNet'],False); mae_ev=per_user_mae(pool['EV-SDG'],False)
# dependence
def dep(D):
    pr=[('arrival','duration'),('arrival','energy'),('duration','energy')]; te=[]
    for x,y in pr: te.append(abs(stats.kendalltau(R[x],R[y]).statistic-stats.kendalltau(D[x],D[y]).statistic))
    return float(np.mean(te))
BINS={'arrival':(0,24,24),'duration':(0,48,48),'energy':(0,80,40)}
popkl=lambda D: float(np.mean([kl(Hh(R[v],*BINS[v]),Hh(D[v],*BINS[v])) for v in VARS]))
gapkl_ours=kl(Hh(real_gap,0,72,36),Hh(ours_gap,0,72,36))
rows=[
 ['Idle-gap KL (sequential)', f'{gapkl_ours:.3f}', 'N/A', 'N/A'],
 ['Per-user arrival MAE (h)', f'{mae_ours["arrival"]:.2f}', f'{mae_gmm["arrival"]:.2f}', f'{mae_ev["arrival"]:.2f}'],
 ['Per-user duration MAE (h)', f'{mae_ours["duration"]:.2f}', f'{mae_gmm["duration"]:.2f}', f'{mae_ev["duration"]:.2f}'],
 ['Per-user energy MAE (kWh)', f'{mae_ours["energy"]:.2f}', f'{mae_gmm["energy"]:.2f}', f'{mae_ev["energy"]:.2f}'],
 ['Population marginal KL', f'{popkl(O):.3f}', f'{popkl(G):.3f}', f'{popkl(E):.3f}'],
 ['Dependence (Kendall-tau err)', f'{dep(O):.3f}', f'{dep(G):.3f}', f'{dep(E):.3f}'],
]
tab=pd.DataFrame(rows,columns=['Metric','Ours (SMC)','GMMNet','EV-SDG'])
tab.to_csv(OUT/'results_table.csv',index=False)
print(tab.to_string(index=False))
print("\nsaved: plots/paper/heterogeneity_and_sequence.png , results_table.csv")
