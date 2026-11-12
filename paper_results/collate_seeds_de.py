#!/usr/bin/env python3
"""Aggregate seed-tagged runs artifacts_de/ (runs made with run_de_N.jl) <name>_N###_b##_s##/ -> mean±std over seeds.
Groups by everything before the _s tag (i.e. same dataset/N/bins).

    python collate_seeds_de.py              # all seeds found
    python collate_seeds_de.py 1 2 3 4 5    # only these seeds

Runs made with the code before the occupancy repair (no occupancy_repair.csv in
the folder) are skipped and listed, so old and new results are never averaged
together.  Add --all to include them anyway.
"""
import pandas as pd, numpy as np, glob, os, re, sys
HERE=os.path.dirname(os.path.abspath(__file__)); AD=os.path.join(HERE,"artifacts_de")
ARGS=[a for a in sys.argv[1:] if a!="--all"]; KEEP_OLD="--all" in sys.argv[1:]
SEEDS={int(a) for a in ARGS} if ARGS else None
def tb(s): return s.astype(str).str.lower().isin(['true','1','1.0'])
def rate(df,var=None):
    d=df if var is None else df[df.variable==var]
    return d.passed.mean()*100 if len(d) else np.nan
rows=[]; old=[]
for d in sorted(glob.glob(os.path.join(AD,"*_s*"))):
    m=re.search(r"^(.*)_s(\d+)$",os.path.basename(d))
    if not m or not os.path.isfile(f"{d}/validation_l1.csv"): continue
    cfg,seed=m.group(1),int(m.group(2))
    if SEEDS is not None and seed not in SEEDS: continue
    new=os.path.isfile(f"{d}/occupancy_repair.csv")
    if not new and not KEEP_OLD: old.append(os.path.basename(d)); continue
    try:
        l1=pd.read_csv(f"{d}/validation_l1.csv");l2=pd.read_csv(f"{d}/validation_l2.csv");l3=pd.read_csv(f"{d}/validation_l3.csv")
        for x in (l1,l2,l3): x['passed']=tb(x['passed'])
        kl=pd.read_csv(f"{d}/validation_l4.csv")['KL_nats'].max()
        row=dict(cfg=cfg,seed=seed,L1A=rate(l1,'A'),L1D=rate(l1,'D'),L1G=rate(l1,'G'),
                 L1E=rate(l1,'E'),L2=rate(l2),L3=rate(l3),KLmax=kl,Moved=np.nan,KLgapOut=np.nan)
        if new:
            o=pd.read_csv(f"{d}/occupancy_repair.csv").iloc[0]
            row['Moved']=100*o.n_moved/max(o.n_drawn,1)
            f=f"{d}/validation_timeline.csv"
            if os.path.isfile(f): row['KLgapOut']=float(pd.read_csv(f)['KL_nats'].iloc[0])
        rows.append(row)
    except Exception as e: print(f"  {os.path.basename(d)}: {e}")
if old:
    print(f"skipped {len(old)} run(s) made with the old code (no occupancy_repair.csv); re-run them or pass --all:")
    for o in old: print("   ",o)
if not rows: print("No seed-tagged runs to collate. Example: for s in 1 2 3 4 5; do julia --project=. run_korea_N.jl 30 24 $s; done"); raise SystemExit
t=pd.DataFrame(rows); metrics=['L1A','L1D','L1G','L1E','L2','L3','KLmax','Moved','KLgapOut']
g=t.groupby('cfg')
print(f"{'config':32} {'seeds':>9}  "+"  ".join(f"{m:>11}" for m in metrics))
for cfg,grp in g:
    cells=[]
    for m in metrics:
        mu,sd=grp[m].mean(),grp[m].std(ddof=0)
        cells.append(f"{mu:5.1f}±{sd:4.1f}" if m not in ('KLmax','KLgapOut') else f"{mu:.3f}±{sd:.3f}")
    print(f"{cfg:32} {','.join(str(x) for x in sorted(grp.seed)):>9}  "+"  ".join(cells))
print("\nMoved = sessions moved by the occupancy repair (%); KLgapOut = idle-gap KL on the output calendar (nats)")
