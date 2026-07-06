"""
Per-dataset comparison + plots for the synthetic EV charging generators.

    python benchmarking/plots_and_metrics.py <combined|norway>

Prints the metric panel and writes to plots/comparison/<dataset>/:
    marginals_<ds>.png     arrival / duration / energy / idle-gap PDF overlays
    kl_bars_<ds>.png       per-variable KL per generator
    load_profile_<ds>.png  average daily load curve (real vs generators)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(24)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DS = sys.argv[1] if len(sys.argv) > 1 else 'combined'
REAL = {'combined': ROOT/'dataset'/'combined'/'combined_sessions.csv',
        'norway':   ROOT/'artifacts'/'norway_sorensen'/'raw_clean.csv'}[DS]
GENS = {  # label -> (path, colmap, color)
 'Ours (SMC)': (ROOT/'artifacts'/('combined_module' if DS=='combined' else 'norway_sorensen')/'sim_sessions.csv',
                {'arrival_hour':'arrival_hour','duration_h':'duration_h','energy_kwh':'energy_kwh'}, '#1f77b4'),
 'GMMNet':     (HERE/'gmmnet'/'results'/f'gmmnet_{DS}.csv',
                {'arrival_hour':'arrival_hour','duration_h':'duration_h','energy_kwh':'energy_kwh'}, '#2ca02c'),
 'EV-SDG':     (HERE/'evsdg'/'results'/f'EVSDG_{"combined" if DS=="combined" else "norway"}_generated.csv',
                {'arrival_hour':'Arrival','duration_h':'Connected_time','energy_kwh':'Energy_required'}, '#d62728'),
 'Vine copula':(HERE/'copula'/'results'/f'copula_{DS}.csv',
                {'arrival_hour':'arrival_hour','duration_h':'duration_h','energy_kwh':'energy_kwh'}, '#9467bd'),
 'TVAE (deep gen.)':(HERE/'ctgan'/'results'/f'tvae_{DS}.csv',
                {'arrival_hour':'arrival_hour','duration_h':'duration_h','energy_kwh':'energy_kwh'}, '#8c564b'),
}
VARS = ['arrival_hour','duration_h','energy_kwh']
BINS = {'arrival_hour':(0,24,24),'duration_h':(0,48,48),'energy_kwh':(0,80,40)}
LAB  = {'arrival_hour':'Arrival hour','duration_h':'Plug-in duration (h)',
        'energy_kwh':'Energy (kWh)','gap_h':'Idle gap (h)'}

def kl(p,q): p=p+1e-9;q=q+1e-9;p/=p.sum();q/=q.sum();return float(np.sum(p*np.log(p/q)))
def H(x,var): lo,hi,nb=BINS[var]; h,_=np.histogram(np.clip(np.asarray(x,float),lo,hi),bins=np.linspace(lo,hi,nb+1)); return h.astype(float)

def load_real():
    d=pd.read_csv(REAL).dropna(subset=['abs_start','abs_end','duration_hours','energy','user_id'])
    t=pd.to_datetime(d['abs_start']); te=pd.to_datetime(d['abs_end'])
    arr = d['arrival_hour'].to_numpy(float) if 'arrival_hour' in d and DS=='norway' else (t.dt.hour+t.dt.minute/60.0).to_numpy()
    dd=d.assign(_a=arr,_d=d['duration_hours'].astype(float),_e=d['energy'].astype(float),_t0=t,_t1=te)
    gaps=[]
    for _,g in dd.sort_values('_t0').groupby('user_id'):
        gp=(g['_t0'].shift(-1)-g['_t1']).dt.total_seconds().to_numpy()[:-1]/3600.0; gaps.append(gp[gp>=0])
    return {'arrival_hour':dd['_a'].to_numpy(),'duration_h':dd['_d'].to_numpy(),
            'energy_kwh':dd['_e'].to_numpy(),'gap_h':np.concatenate(gaps)}

def load_gen(path,cmap):
    if not Path(path).exists(): return None
    g=pd.read_csv(path)
    out={v:np.asarray(g[cmap[v]],float) for v in VARS}
    out['gap_h']=np.asarray(g['gap_h'],float) if 'gap_h' in g.columns else None
    return out

def load_profile(a,d,e,cap=48.0,nb=24):
    prof=np.zeros(nb); a=np.asarray(a,float); d=np.minimum(np.asarray(d,float),cap); e=np.asarray(e,float)
    for ai,di,ei in zip(a,d,e):
        if di<=0: continue
        p=ei/di; t=0.0; cur=ai
        while t<di:
            hod=int(np.floor(cur))%nb; seg=min(np.floor(cur)+1-cur,di-t); seg=seg if seg>0 else min(1.0,di-t)
            prof[hod]+=p*seg; t+=seg; cur+=seg
    return prof/len(a)

real=load_real()
gens={lab:load_gen(p,c) for lab,(p,c,_) in GENS.items()}
colors={lab:col for lab,(_,_,col) in GENS.items()}
gens={k:v for k,v in gens.items() if v is not None}
print(f"=== {DS.upper()} | real {len(real['arrival_hour'])} sessions ===")

# ---- metric table (per-variable KL + mean, gap KL) ----
print(f"{'generator':14s} {'KL_arr':>8s} {'KL_dur':>8s} {'KL_en':>8s} {'meanKL':>8s} {'gapKL':>8s}")
rows=[]
for lab,g in gens.items():
    kA=kl(H(real['arrival_hour'],'arrival_hour'),H(g['arrival_hour'],'arrival_hour'))
    kD=kl(H(real['duration_h'],'duration_h'),H(g['duration_h'],'duration_h'))
    kE=kl(H(real['energy_kwh'],'energy_kwh'),H(g['energy_kwh'],'energy_kwh'))
    m=np.mean([kA,kD,kE])
    if g['gap_h'] is not None:
        e=np.linspace(0,72,37); gk=kl(np.histogram(np.clip(real['gap_h'],0,72),bins=e)[0].astype(float),
                                       np.histogram(np.clip(g['gap_h'],0,72),bins=e)[0].astype(float))
    else: gk=np.nan
    rows.append((lab,kA,kD,kE,m,gk))
    print(f"{lab:14s} {kA:8.4f} {kD:8.4f} {kE:8.4f} {m:8.4f} {gk if np.isnan(gk) else round(gk,4):>8}")

OUT=ROOT/'plots'/'comparison'/DS; OUT.mkdir(parents=True,exist_ok=True)

# ---- Plot 1: marginal PDF overlays (arrival, duration, energy, gap) ----
fig,axes=plt.subplots(2,2,figsize=(11,7.5)); axes=axes.ravel()
panels=[('arrival_hour',0,24),('duration_h',0,48),('energy_kwh',0,80),('gap_h',0,72)]
for ax,(v,lo,hi) in zip(axes,panels):
    bins=np.linspace(lo,hi,40)
    ax.hist(np.clip(real[v],lo,hi),bins=bins,density=True,histtype='step',color='k',lw=2.4,label='Real')
    for lab,g in gens.items():
        if g.get(v) is None: continue
        ax.hist(np.clip(g[v],lo,hi),bins=bins,density=True,histtype='step',color=colors[lab],lw=1.8,label=lab)
    ax.set_title(LAB[v],fontweight='bold'); ax.set_xlabel(LAB[v]); ax.set_ylabel('density')
    ax.legend(fontsize=8)
fig.suptitle(f'{DS.capitalize()} — marginal distributions (real vs. generators)',fontweight='bold',fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/f'marginals_{DS}.png',dpi=130); plt.close(fig)

# ---- Plot 2: per-variable KL grouped bars ----
fig,ax=plt.subplots(figsize=(9,5))
labels=[r[0] for r in rows]; x=np.arange(3); w=0.8/len(rows)
for i,r in enumerate(rows):
    ax.bar(x+i*w,[r[1],r[2],r[3]],w,label=r[0],color=colors[r[0]])
ax.set_xticks(x+w*(len(rows)-1)/2); ax.set_xticklabels(['Arrival','Duration','Energy'])
ax.set_ylabel('KL divergence (nats)'); ax.set_title(f'{DS.capitalize()} — per-variable KL (lower = better)',fontweight='bold')
ax.legend(); ax.set_yscale('log'); fig.tight_layout(); fig.savefig(OUT/f'kl_bars_{DS}.png',dpi=130); plt.close(fig)

# ---- Plot 3: average daily load profile ----
fig,ax=plt.subplots(figsize=(9,5))
pr=load_profile(real['arrival_hour'],real['duration_h'],real['energy_kwh'])
ax.plot(range(24),pr,'k-',lw=2.6,marker='o',ms=3,label='Real')
for lab,g in gens.items():
    pg=load_profile(g['arrival_hour'],g['duration_h'],g['energy_kwh'])
    ax.plot(range(24),pg,color=colors[lab],lw=1.8,marker='.',label=lab)
ax.set_xlabel('Hour of day'); ax.set_ylabel('Avg load (kWh/h per session)')
ax.set_title(f'{DS.capitalize()} - average daily load profile',fontweight='bold'); ax.legend()
fig.tight_layout(); fig.savefig(OUT/f'load_profile_{DS}.png',dpi=130); plt.close(fig)

print(f'\nplots -> plots/comparison/{DS}/  (marginals, kl_bars, load_profile)')
