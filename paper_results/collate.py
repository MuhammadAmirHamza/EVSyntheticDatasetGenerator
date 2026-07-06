#!/usr/bin/env python3
"""Collate ALL runs under artifacts/*/ (any dataset). Prints pass-% + L4 KL per run."""
import pandas as pd, glob, os
HERE=os.path.dirname(os.path.abspath(__file__)); AD=os.path.join(HERE,"artifacts")
def tb(s): return s.astype(str).str.lower().isin(['true','1','1.0'])
def rate(df,var=None):
    d=df if var is None else df[df.variable==var]
    return round(d.passed.mean()*100,1) if len(d) else float('nan')
rows=[]
for d in sorted(glob.glob(os.path.join(AD,"*"))):
    if not os.path.isfile(f"{d}/validation_l1.csv"): continue
    try:
        l1=pd.read_csv(f"{d}/validation_l1.csv"); l2=pd.read_csv(f"{d}/validation_l2.csv"); l3=pd.read_csv(f"{d}/validation_l3.csv")
        for x in (l1,l2,l3): x['passed']=tb(x['passed'])
        kl=dict(zip(*[pd.read_csv(f"{d}/validation_l4.csv")[c] for c in ('short','KL_nats')]))
        rows.append(dict(run=os.path.basename(d), users=l2.user_id.nunique(),
            L1A=rate(l1,'A'),L1D=rate(l1,'D'),L1G=rate(l1,'G'),L1E=rate(l1,'E'),
            L2=rate(l2),L3=rate(l3),
            KLmax=round(max(kl.values()),3)))
    except Exception as e: print(f"  {os.path.basename(d)}: {e}")
if rows:
    t=pd.DataFrame(rows); pd.set_option('display.width',160)
    print(t.to_string(index=False))
else: print("No runs found under artifacts/.")
