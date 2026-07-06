#!/usr/bin/env python3
"""
CTMC-vs-SMC evidence: per-profile coefficient of variation of the effective
inter-arrival time  CV = sigma_tau / mu_tau.  A CTMC forces exponential
sojourns (CV = 1 exactly); the share of profiles with CV != 1 quantifies how
often that memoryless assumption is violated, motivating the SMC's arbitrary F_b.
Reads each dataset's pairs.csv (tau_eff), computes CV per (entity, day_type)
profile with >= N_MIN pairs, and prints a summary table (+ LaTeX).
"""
import pandas as pd, numpy as np, os

N_MIN = 10           # min pairs for a stable per-profile CV
BAND  = 0.10         # "consistent with exponential" = CV in [1-BAND, 1+BAND]

DATASETS = {   # label : (pairs.csv path, entity word)
    "Norway (res., driver)"   : ("artifacts/norway_sorensen_N100_b12_s01/pairs.csv","driver"),
    "GIST (res., driver)"     : ("artifacts/korea_gist_N030_b24_s01/pairs.csv","driver"),
    "Kim (comm., driver)"     : ("artifacts/kim_commercial_N025_b24_s01/pairs.csv","driver"),
    "ACN (workpl., station)"  : ("artifacts/acn_caltech_N015_b12_s01/pairs.csv","station"),
    "Combined"                : ("artifacts/combined_N200_b24_s01/pairs.csv","mixed"),
}

def profile_cvs(path):
    df = pd.read_csv(path)
    tcol = "tau_eff" if "tau_eff" in df.columns else [c for c in df.columns if "tau" in c.lower()][0]
    cvs=[]
    for (u,dt),g in df.groupby(["user_id","day_type"]):
        t=g[tcol].values; t=t[np.isfinite(t)&(t>0)]
        if len(t)<N_MIN or t.mean()==0: continue
        cvs.append(t.std(ddof=1)/t.mean())
    return np.array(cvs)

rows=[]
for label,(path,ent) in DATASETS.items():
    if not os.path.exists(path): print("MISSING",path); continue
    cv=profile_cvs(path)
    sub =np.mean(cv<1-BAND)*100      # sub-exponential (more regular)
    expo=np.mean((cv>=1-BAND)&(cv<=1+BAND))*100  # ~ exponential
    over=np.mean(cv>1+BAND)*100      # over-dispersed (bursty)
    rows.append(dict(Dataset=label, Profiles=len(cv), medCV=np.median(cv),
                     sub=sub, expo=expo, over=over, neq1=sub+over))
t=pd.DataFrame(rows)
pd.set_option("display.width",160,"display.float_format",lambda v:f"{v:.1f}")
print(t.to_string(index=False))
print("\nsub = CV<0.9 (regular, CTMC over-estimates dispersion); "
      "expo = 0.9-1.1 (CTMC adequate); over = CV>1.1 (bursty, CTMC under-estimates tail)")
print(f"neq1 = % profiles with CV != 1  -> CTMC (exponential sojourn) inappropriate")

# LaTeX
with open("cv_ctmc_table.tex","w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("Dataset & Profiles & Median CV & CV$<$0.9 & CV$\\approx$1 & CV$>$1.1 \\\\\n")
    f.write("        &          &           & (\\%)     & (\\%)        & (\\%) \\\\\n\\midrule\n")
    for _,r in t.iterrows():
        f.write(f"{r.Dataset} & {int(r.Profiles)} & {r.medCV:.2f} & {r['sub']:.0f} & {r['expo']:.0f} & {r['over']:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("\nwrote cv_ctmc_table.tex")
