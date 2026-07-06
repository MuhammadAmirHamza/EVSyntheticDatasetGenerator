#!/usr/bin/env python3
"""Discriminating metrics that aggregate marginals cannot capture.

(1) Between-user heterogeneity (P1): across-user spread of each user's mean
    arrival hour (circular) and mean plug-in duration. SPARC and the real data
    carry user identity; the pooled baselines (copula, GMMNet, EV-SDG) emit an
    exchangeable session bag, so their per-user spread collapses to sampling
    noise. We quantify this by assigning each baseline's pooled sessions to
    K pseudo-users drawn from the real per-user session-count distribution.

(2) Aggregate daily load-profile divergence: the 24 h fleet load shape is a
    JOINT functional of (A, D, E), L(h)=sum_i (E_i/D_i)*overlap(session_i, hour h).
    Reported as forward KL(real||model) of the normalised load shape.
"""
import glob
import numpy as np
import pandas as pd

ART = "paper_results/artifacts"
EVS = "benchmarking/evsdg/results"; GMM = "benchmarking/gmmnet/results"; COP = "benchmarking/copula/results"
RNG = np.random.default_rng(0)
MIN_SESS = 3           # min sessions for a stable per-user mean
R_BOOT   = 30          # random pseudo-user partitions per baseline

DATASETS = {
    "Norway":    ("norway_sorensen_N100_b12", "norway"),
    "GIST-Res":  ("korea_gist_N030_b24",      "gist"),
    "GIST-Comm": ("kim_commercial_N025_b24",  "kim"),
    "ACN":       ("acn_caltech_N015_b12",     "acn"),
    "Combined":  ("combined_all_N200_b24",    "combined"),
}

def circ_mean_hour(h):
    th = 2*np.pi*np.asarray(h)/24.0
    return (np.arctan2(np.sin(th).mean(), np.cos(th).mean()) % (2*np.pi))*24/(2*np.pi)

def between_user_sd(df, ucol, acol, dcol):
    """Return (SD across users of circular-mean arrival hour, SD of mean duration)."""
    ma, md = [], []
    for _, g in df.groupby(ucol):
        if len(g) < MIN_SESS: continue
        ma.append(circ_mean_hour(g[acol].values))
        md.append(np.mean(g[dcol].values))
    ma = np.array(ma); md = np.array(md)
    # circular SD across the per-user mean hours
    th = 2*np.pi*ma/24.0
    Rbar = np.hypot(np.cos(th).mean(), np.sin(th).mean())
    sd_a = np.sqrt(max(-2*np.log(max(Rbar,1e-9)),0))*24/(2*np.pi)
    return sd_a, float(np.std(md))

def between_user_sd_pooled(A, D, counts):
    """Assign pooled sessions to pseudo-users with given session counts; avg over R_BOOT."""
    A = np.asarray(A); D = np.asarray(D); n = len(A)
    sa, sd = [], []
    counts = [c for c in counts if c >= MIN_SESS]
    for _ in range(R_BOOT):
        idx = RNG.permutation(n); pos = 0; ma, md = [], []
        for c in counts:
            if pos+c > n: break
            sl = idx[pos:pos+c]; pos += c
            ma.append(circ_mean_hour(A[sl])); md.append(np.mean(D[sl]))
        ma = np.array(ma); md = np.array(md)
        th = 2*np.pi*ma/24.0
        Rbar = np.hypot(np.cos(th).mean(), np.sin(th).mean())
        sa.append(np.sqrt(max(-2*np.log(max(Rbar,1e-9)),0))*24/(2*np.pi)); sd.append(np.std(md))
    return float(np.mean(sa)), float(np.mean(sd))

def load_shape(A, D, E):
    A = np.asarray(A,float)%24; D = np.clip(np.asarray(D,float),1e-3,None); E = np.asarray(E,float)
    P = E/D                                   # avg power per session (kW)
    L = np.zeros(24)
    for a,d,p in zip(A,D,P):
        end = a+d; h = int(np.floor(a))
        while h < end:
            lo=max(a,h); hi=min(end,h+1)
            if hi>lo: L[h%24]+=p*(hi-lo)
            h+=1
    s=L.sum()
    return L/s if s>0 else L

def kl_shape(pr,pm):
    pr=pr+1e-9; pm=pm+1e-9; pr/=pr.sum(); pm/=pm.sum()
    return float(np.sum(pr*np.log(pr/pm)))

rows=[]
for name,(prefix,key) in DATASETS.items():
    real=pd.read_csv(f"{ART}/{prefix}_s01/raw_clean.csv")
    rc={"u":"user_id","a":"arrival_hour","d":"duration_hours","e":"energy"}
    real=real.dropna(subset=[rc["a"],rc["d"],rc["e"]])
    real=real[(real[rc["d"]]>0)&(real[rc["e"]]>0)]
    counts=real.groupby(rc["u"]).size().values
    rSDa,rSDd=between_user_sd(real,rc["u"],rc["a"],rc["d"])
    rShape=load_shape(real[rc["a"]],real[rc["d"]],real[rc["e"]])

    sim=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{ART}/{prefix}_s0*/sim_sessions.csv"))],ignore_index=True)
    sSDa,sSDd=between_user_sd(sim,"user_id","arrival_hour","duration_h")
    sShape=load_shape(sim["arrival_hour"],sim["duration_h"],sim["energy_kwh"])

    # Real reference row
    rows.append([name,"Real",rSDa,rSDd,0.0])
    rows.append([name,"SPARC",sSDa,sSDd,kl_shape(rShape,sShape)])
    for label,path in [("Copula",f"{COP}/einolander_{key}.csv"),
                       ("GMMNet",f"{GMM}/gmmnet_{key}.csv"),
                       ("EV-SDG",f"{EVS}/EVSDG_{key}.csv")]:
        m=pd.read_csv(path)
        A=pd.to_numeric(m["arrival_hour"],errors="coerce").values
        D=pd.to_numeric(m["duration_h"],errors="coerce").values
        E=pd.to_numeric(m["energy_kwh"],errors="coerce").values
        ok=~(np.isnan(A)|np.isnan(D)|np.isnan(E)); A,D,E=A[ok],D[ok],E[ok]
        pSDa,pSDd=between_user_sd_pooled(A,D,counts)
        rows.append([name,label,pSDa,pSDd,kl_shape(rShape,load_shape(A,D,E))])

df=pd.DataFrame(rows,columns=["Dataset","Method","SDarr_h","SDdur_h","LoadKL"])
df.to_csv("benchmarking/heterogeneity.csv",index=False)
pd.set_option("display.float_format",lambda v:f"{v:.3f}")
print(df.to_string(index=False))
