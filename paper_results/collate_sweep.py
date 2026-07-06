#!/usr/bin/env python3
"""Collate the Norway user-count sweep. Reads artifacts/norway_N*/validation_l1..l4
and kl_divergence, prints a pass-rate table so you can pick the results-optimal N."""
import pandas as pd, numpy as np, glob, os, re
HERE=os.path.dirname(os.path.abspath(__file__)); AD=os.path.join(HERE,"artifacts")
def tobool(s): return s.astype(str).str.lower().isin(['true','1','1.0'])
def rate(df,var=None):
    d=df if var is None else df[df.variable==var]
    return d.passed.mean()*100 if len(d) else float('nan')
rows=[]
for d in sorted(glob.glob(os.path.join(AD,"norway_N*"))):
    N=int(re.search(r"N(\d+)",os.path.basename(d)).group(1))
    try:
        l1=pd.read_csv(f"{d}/validation_l1.csv"); l2=pd.read_csv(f"{d}/validation_l2.csv"); l3=pd.read_csv(f"{d}/validation_l3.csv")
        for x in (l1,l2,l3): x['passed']=tobool(x['passed'])
        l4=pd.read_csv(f"{d}/validation_l4.csv"); kl=dict(zip(l4.short,l4.KL_nats))
        rows.append(dict(N=N,users=l2.user_id.nunique(),
            L1A=rate(l1,'A'),L1D=rate(l1,'D'),L1G=rate(l1,'G'),L1E=rate(l1,'E'),
            L2=rate(l2),L3=rate(l3),
            KLA=kl.get('A'),KLD=kl.get('D'),KLG=kl.get('G'),KLE=kl.get('E')))
    except Exception as e: print(f"  N={N}: {e}")
if not rows: print("No sweep artifacts found. Run: for N in 100 120 140 161; do julia run_norway_N.jl $N; done"); raise SystemExit
t=pd.DataFrame(rows).sort_values('N')
pd.set_option('display.width',160,'display.float_format',lambda v:f'{v:.2f}')
print(t.to_string(index=False))
# composite score: mean of L1(A,D,G,E),L2,L3 minus KL penalty
t['pass_mean']=t[['L1A','L1D','L1G','L1E','L2','L3']].mean(axis=1)
t['kl_max']=t[['KLA','KLD','KLG','KLE']].max(axis=1)
print("\nBy mean pass-rate:"); print(t.loc[t.pass_mean.idxmax(),['N','users','pass_mean','kl_max']].to_string())
