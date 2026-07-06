# Evidence PoC: does a full VINE over (A,D,G,E) capture dependence that SPARC's
# current per-bin Gaussian-copula-on-(D,G) misses? Runs on the REAL combined pairs.
import numpy as np, pandas as pd, pyvinecopulib as pv
from scipy.stats import kendalltau
np.random.seed(24)
d=pd.read_csv('../artifacts/combined_module/pairs.csv')
X=d[['A','D','G_eff','E']].to_numpy(float); names=['A','D','G','E']
u=pv.to_pseudo_obs(X); N=len(u)

def taumat(M):
    k=M.shape[1]; T=np.zeros((k,k))
    for i in range(k):
        for j in range(k): T[i,j]=kendalltau(M[:,i],M[:,j])[0]
    return T
Treal=taumat(X)
print('=== Real Kendall-tau matrix (A,D,G,E) ===')
print('      '+'   '.join(f'{n:>5}' for n in names))
for i,n in enumerate(names): print(f'{n:>3} '+' '.join(f'{Treal[i,j]:6.3f}' for j in range(4)))

def fit(fams,label):
    ctrl=pv.FitControlsVinecop(family_set=fams, selection_criterion='aic', num_threads=4)
    cop=pv.Vinecop.from_data(u,controls=ctrl)
    sim=np.asarray(cop.simulate(N,seeds=[24,7,13,99]))
    Ts=taumat(sim)
    err=np.mean(np.abs(Ts-Treal)[np.triu_indices(4,1)])
    ll=float(cop.loglik()); aic=float(cop.aic()); bic=float(cop.bic())
    print(f'{label:14s} | loglik {ll:9.1f} | AIC {aic:9.1f} | BIC {bic:9.1f} | mean|tau err| {err:.4f}')
    return err
print('\n=== copula model comparison (lower AIC & tau-err = better) ===')
fit([pv.BicopFamily.gaussian],'Gaussian')
fit([pv.BicopFamily.student],'Student-t')
fit([pv.BicopFamily.gaussian,pv.BicopFamily.student,pv.BicopFamily.clayton,
     pv.BicopFamily.gumbel,pv.BicopFamily.frank,pv.BicopFamily.joe,pv.BicopFamily.bb1],'Vine (full)')
# what SPARC models now: only D-G coupling; report cross-pair strengths it ignores
print('\n=== dependence SPARC currently IGNORES (only models D-G) ===')
for a,b in [('A','D'),('A','G'),('A','E'),('D','E'),('G','E')]:
    i,j=names.index(a),names.index(b); print(f'  tau({a},{b}) = {Treal[i,j]:+.3f}')
print(f'  tau(D,G) = {Treal[1,2]:+.3f}   <- the only pair SPARC couples')
