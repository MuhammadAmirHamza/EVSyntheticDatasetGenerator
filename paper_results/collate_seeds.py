#!/usr/bin/env python3
"""Aggregate seed-tagged runs artifacts/<name>_N###_b##_s##/ -> mean±std over seeds.
Groups by everything before the _s tag (i.e. same dataset/N/bins)."""
import pandas as pd, numpy as np, glob, os, re
HERE=os.path.dirname(os.path.abspath(__file__)); AD=os.path.join(HERE,"artifacts")
def tb(s): return s.astype(str).str.lower().isin(['true','1','1.0'])
def rate(df,var=None):
    d=df if var is None else df[df.variable==var]
    return d.passed.mean()*100 if len(d) else np.nan
rows=[]
for d in sorted(glob.glob(os.path.join(AD,"*_s*"))):
    m=re.search(r"^(.*)_s(\d+)$",os.path.basename(d))
    if not m or not os.path.isfile(f"{d}/validation_l1.csv"): continue
    cfg,seed=m.group(1),int(m.group(2))
    try:
        l1=pd.read_csv(f"{d}/validation_l1.csv");l2=pd.read_csv(f"{d}/validation_l2.csv");l3=pd.read_csv(f"{d}/validation_l3.csv")
        for x in (l1,l2,l3): x['passed']=tb(x['passed'])
        kl=pd.read_csv(f"{d}/validation_l4.csv")['KL_nats'].max()
        rows.append(dict(cfg=cfg,seed=seed,L1A=rate(l1,'A'),L1D=rate(l1,'D'),L1G=rate(l1,'G'),
                         L1E=rate(l1,'E'),L2=rate(l2),L3=rate(l3),KLmax=kl))
    except Exception as e: print(f"  {os.path.basename(d)}: {e}")
if not rows: print("No seed-tagged runs (…_sNN). Example: for s in 1 2 3 4 5; do julia --project=. run_georgia_N.jl 30 16 $s; done"); raise SystemExit
t=pd.DataFrame(rows); metrics=['L1A','L1D','L1G','L1E','L2','L3','KLmax']
g=t.groupby('cfg')
print(f"{'config':32} {'seeds':>5}  "+"  ".join(f"{m:>11}" for m in metrics))
for cfg,grp in g:
    cells=[]
    for m in metrics:
        mu,sd=grp[m].mean(),grp[m].std(ddof=0)
        cells.append(f"{mu:5.1f}±{sd:4.1f}" if m!='KLmax' else f"{mu:.3f}±{sd:.3f}")
    print(f"{cfg:32} {len(grp):>5}  "+"  ".join(cells))
