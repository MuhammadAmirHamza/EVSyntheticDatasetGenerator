#!/usr/bin/env python3
"""Seed-swept benchmark with error bars.

SPARC error = variability across the 5 simulation seeds (s01..s05), matching the
validation table's methodology. Baseline error = estimator variability: bootstrap
resampling of the generated session bag for the marginal KL, and the random
pseudo-user partition spread for the heterogeneity SDs.
"""
import glob
import numpy as np
import pandas as pd

ART="paper_results/artifacts"; EVS="benchmarking/evsdg/results"
GMM="benchmarking/gmmnet/results"; COP="benchmarking/copula/results"
RNG=np.random.default_rng(0)
MIN_SESS=3; R_BOOT=30; B_KL=100

DATASETS={
 "Norway":("norway_sorensen_N100_b12","norway"),
 "GIST-Res":("korea_gist_N030_b24","gist"),
 "GIST-Comm":("kim_commercial_N025_b24","kim"),
 "ACN":("acn_caltech_N015_b12","acn"),
 "Combined":("combined_all_N200_b24","combined"),
}
BASE=[("Copula",COP,"einolander"),("GMMNet",GMM,"gmmnet"),("EV-SDG",EVS,"EVSDG")]

def kl(real,model,edges):
    pr,_=np.histogram(real,bins=edges); pm,_=np.histogram(model,bins=edges)
    pr=pr+1e-9; pm=pm+1e-9; pr=pr/pr.sum(); pm=pm/pm.sum()
    return float(np.sum(pr*np.log(pr/pm)))

def edges_for(x,n=30): return np.linspace(0,np.percentile(x,99),n+1)
def circ_mean(h):
    th=2*np.pi*np.asarray(h,float)/24.0
    return (np.arctan2(np.sin(th).mean(),np.cos(th).mean())%(2*np.pi))*24/(2*np.pi)
def circ_sd(hours):
    th=2*np.pi*np.asarray(hours,float)/24.0
    R=np.hypot(np.cos(th).mean(),np.sin(th).mean())
    return np.sqrt(max(-2*np.log(max(R,1e-9)),0))*24/(2*np.pi)

def hetero(df,ucol,acol,dcol):
    ma,md=[],[]
    for _,g in df.groupby(ucol):
        if len(g)<MIN_SESS: continue
        ma.append(circ_mean(g[acol].values)); md.append(np.mean(g[dcol].values))
    return circ_sd(ma), float(np.std(md))

def hetero_pooled_dist(A,D,counts):
    A=np.asarray(A); D=np.asarray(D); n=len(A)
    counts=[c for c in counts if c>=MIN_SESS]; sa,sd=[],[]
    for _ in range(R_BOOT):
        idx=RNG.permutation(n); pos=0; ma,md=[],[]
        for c in counts:
            if pos+c>n: break
            sl=idx[pos:pos+c]; pos+=c
            ma.append(circ_mean(A[sl])); md.append(np.mean(D[sl]))
        sa.append(circ_sd(ma)); sd.append(np.std(md))
    return np.array(sa),np.array(sd)

def load_shape(A,D,E):
    A=np.asarray(A,float)%24; D=np.clip(np.asarray(D,float),1e-3,None); E=np.asarray(E,float)
    P=E/D; L=np.zeros(24)
    for a,d,p in zip(A,D,P):
        end=a+d; h=int(np.floor(a))
        while h<end:
            lo=max(a,h); hi=min(end,h+1)
            if hi>lo: L[h%24]+=p*(hi-lo)
            h+=1
    s=L.sum(); return L/s if s>0 else L
def kl_shape(pr,pm):
    pr=pr+1e-9; pm=pm+1e-9; pr=pr/pr.sum(); pm=pm/pm.sum()
    return float(np.sum(pr*np.log(pr/pm)))

mk=[]; mh=[]   # marginal-KL rows, heterogeneity rows
for name,(prefix,key) in DATASETS.items():
    real=pd.read_csv(f"{ART}/{prefix}_s01/raw_clean.csv")
    real=real.dropna(subset=["arrival_hour","duration_hours","energy"])
    real=real[(real.duration_hours>0)&(real.energy>0)]
    rA=real.arrival_hour.values; rD=real.duration_hours.values; rE=real.energy.values
    eA=np.linspace(0,24,25); eD=edges_for(rD); eE=edges_for(rE)
    counts=real.groupby("user_id").size().values
    rSDa,rSDd=hetero(real,"user_id","arrival_hour","duration_hours")
    rShape=load_shape(rA,rD,rE)
    mh.append([name,"Real",rSDa,0,rSDd,0,np.nan,np.nan])

    # ---- SPARC per seed ----
    seeds=sorted(glob.glob(f"{ART}/{prefix}_s0*/sim_sessions.csv"))
    kA=[];kD=[];kE=[]; hA=[];hD=[];hL=[]
    for sp in seeds:
        s=pd.read_csv(sp)
        A=np.mod(s.arrival_hour.values,24); D=s.duration_h.values; E=s.energy_kwh.values
        m=(D>0)&(E>0);
        kA.append(kl(rA,A[m],eA)); kD.append(kl(rD,D[m],eD)); kE.append(kl(rE,E[m],eE))
        sa,sd=hetero(s,"user_id","arrival_hour","duration_h")
        hA.append(sa); hD.append(sd); hL.append(kl_shape(rShape,load_shape(A[m],D[m],E[m])))
    mk.append([name,"SPARC",np.mean(kA),np.std(kA),np.mean(kD),np.std(kD),np.mean(kE),np.std(kE)])
    mh.append([name,"SPARC",np.mean(hA),np.std(hA),np.mean(hD),np.std(hD),np.mean(hL),np.std(hL)])

    # ---- baselines ----
    for label,folder,pref in BASE:
        m=pd.read_csv(f"{folder}/{pref}_{key}.csv")
        A=pd.to_numeric(m.arrival_hour,errors="coerce").values
        D=pd.to_numeric(m.duration_h,errors="coerce").values
        E=pd.to_numeric(m.energy_kwh,errors="coerce").values
        ok=~(np.isnan(A)|np.isnan(D)|np.isnan(E)); A,D,E=np.mod(A[ok],24),D[ok],E[ok]
        pos=(D>0)&(E>0); A,D,E=A[pos],D[pos],E[pos]; n=len(A)
        # bootstrap marginal KL
        bA=[];bD=[];bE=[]
        for _ in range(B_KL):
            bi=RNG.integers(0,n,n)
            bA.append(kl(rA,A[bi],eA)); bD.append(kl(rD,D[bi],eD)); bE.append(kl(rE,E[bi],eE))
        mk.append([name,label,np.mean(bA),np.std(bA),np.mean(bD),np.std(bD),np.mean(bE),np.std(bE)])
        # heterogeneity via pseudo-user partitions + load-shape bootstrap
        sa,sd=hetero_pooled_dist(A,D,counts)
        bL=[kl_shape(rShape,load_shape(A[RNG.integers(0,n,n)],D[RNG.integers(0,n,n)],E[RNG.integers(0,n,n)])) for _ in range(20)]
        mh.append([name,label,sa.mean(),sa.std(),sd.mean(),sd.std(),np.mean(bL),np.std(bL)])

cols=["Dataset","Method","A","A_sd","D","D_sd","E","E_sd"]
hcols=["Dataset","Method","SDarr","SDarr_sd","SDdur","SDdur_sd","LoadKL","LoadKL_sd"]
dfk=pd.DataFrame(mk,columns=cols); dfh=pd.DataFrame(mh,columns=hcols)
dfk.to_csv("benchmarking/benchmark_kl_seedsweep.csv",index=False)
dfh.to_csv("benchmarking/heterogeneity_seedsweep.csv",index=False)
pd.set_option("display.float_format",lambda v:f"{v:.3f}")
print("=== MARGINAL KL (mean +/- std) ==="); print(dfk.to_string(index=False))
print("\n=== HETEROGENEITY + LOAD (mean +/- std) ==="); print(dfh.to_string(index=False))
