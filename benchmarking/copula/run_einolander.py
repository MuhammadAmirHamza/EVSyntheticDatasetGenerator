#!/usr/bin/env python3
"""
Einolander & Lahdelma (Energy 2022) multivariate-copula baseline, implemented
from the paper: empirical marginals + a single multivariate STUDENT-t copula
over the per-session variables (arrival hour, plug-in duration, energy).
  - Correlation R calibrated from Kendall's tau: R_ij = sin(pi/2 * tau_ij).
  - Degrees of freedom nu fitted by maximum likelihood of the t-copula.
  - Marginals matched by inverse empirical CDF (copula supplies dependence only).
Generates synthetic sessions and reports L4-style population KL vs real.
Note: the copula has NO temporal/inter-arrival structure, so it models only the
per-session marginals (A, D, E); the idle gap G is out of its scope.
"""
import numpy as np, pandas as pd, os
from scipy.stats import rankdata, kendalltau, t as tdist, multivariate_t

ART="/sessions/awesome-friendly-euler/mnt/SDG/SDG/paper_results/artifacts"
OUTD=os.path.join(os.path.dirname(__file__),"results"); os.makedirs(OUTD,exist_ok=True)
DS={"norway":"norway_sorensen_N100_b12_s01","gist":"korea_gist_N030_b24_s01",
    "kim":"kim_commercial_N025_b24_s01","acn":"acn_caltech_N015_b12_s01",
    "combined":"combined_all_N200_b24_s01"}
np.random.seed(7)

def nearest_pd_corr(R):
    R=(R+R.T)/2; w,V=np.linalg.eigh(R); w=np.clip(w,1e-6,None)
    R=V@np.diag(w)@V.T; d=np.sqrt(np.diag(R)); return R/np.outer(d,d)

def fit_t_copula(X):
    n,d=X.shape; U=np.clip(rankdata(X,axis=0)/(n+1),1e-6,1-1e-6)
    R=np.eye(d)
    for i in range(d):
        for j in range(i+1,d):
            tau=kendalltau(X[:,i],X[:,j]).statistic
            R[i,j]=R[j,i]=np.sin(np.pi/2*tau)
    R=nearest_pd_corr(R)
    grid=np.r_[np.arange(2.5,12,0.5),np.arange(12,61,4)]
    def negll(nu):
        Z=tdist.ppf(U,nu)
        return -(multivariate_t.logpdf(Z,loc=np.zeros(d),shape=R,df=nu)-tdist.logpdf(Z,nu).sum(1)).sum()
    nu=min(grid,key=negll)
    return R,float(nu)

def sample(R,nu,cols,N):
    d=R.shape[0]; Z=multivariate_t.rvs(loc=np.zeros(d),shape=R,df=nu,size=N)
    U=np.clip(tdist.cdf(Z,nu),1e-6,1-1e-6)
    return np.column_stack([np.quantile(cols[j],U[:,j]) for j in range(d)])

def kl(real,synt,lo,hi,nb):
    b=np.linspace(lo,hi,nb+1)
    pr,_=np.histogram(real,b); ps,_=np.histogram(synt,b)
    pr=pr/pr.sum()+1e-9; ps=ps/ps.sum()+1e-9
    return float(np.sum(pr*np.log(pr/ps)))

print(f"{'dataset':9}{'nu':>6}{'KL_A':>8}{'KL_D':>8}{'KL_E':>8}   (Student-t copula, A/D/E)")
for name,d in DS.items():
    raw=pd.read_csv(f"{ART}/{d}/raw_clean.csv")
    X=raw[["arrival_hour","duration_hours","energy"]].values.astype(float)
    m=np.all(np.isfinite(X),1)&(X[:,1]>0)&(X[:,2]>0); X=X[m]
    R,nu=fit_t_copula(X)
    Nsim=len(pd.read_csv(f"{ART}/{d}/sim_sessions.csv"))
    S=sample(R,nu,[X[:,0],X[:,1],X[:,2]],Nsim)
    out=pd.DataFrame({"arrival_hour":S[:,0],"duration_h":S[:,1],"energy_kwh":S[:,2]})
    out.to_csv(f"{OUTD}/einolander_{name}.csv",index=False)
    kA=kl(X[:,0],S[:,0],0,24,24)
    kD=kl(X[:,1],S[:,1],0,np.percentile(X[:,1],99),50)
    kE=kl(X[:,2],S[:,2],0,np.percentile(X[:,2],99),50)
    print(f"{name:9}{nu:6.1f}{kA:8.4f}{kD:8.4f}{kE:8.4f}")
print("\nwrote results/einolander_<dataset>.csv")
