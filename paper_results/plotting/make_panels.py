#!/usr/bin/env python3
"""Clean, tight, uniform marginal panels for LaTeX-table assembly.
Every panel: identical figsize + axes rect (=> perfect alignment), small margins,
x-ticks on all (per-row scale), y-ticks only on the left (Norway) column."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mpl.rcParams.update({"savefig.dpi":600,"pdf.fonttype":42,"font.family":"serif",
 "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "axes.linewidth":0.6,"xtick.major.width":0.5,"ytick.major.width":0.5,
 "xtick.major.size":1.8,"ytick.major.size":1.8})
REAL_C,SIM_C="0.45","#1f6fb4"
IN="/sessions/awesome-friendly-euler/mnt/SDG/SDG/paper_results/artifacts"
OUT="/sessions/awesome-friendly-euler/mnt/SDG/SPARC/figures/panels"; os.makedirs(OUT,exist_ok=True)
DS=[("norway","norway_sorensen_N100_b12_s01"),("gistres","korea_gist_N030_b24_s01"),
    ("gistcomm","kim_commercial_N025_b24_s01"),("acn","acn_caltech_N015_b12_s01"),
    ("combined","combined_all_N200_b24_s01")]
QT=[("A","pdf"),("D","pdf"),("G","cdf"),("E","cdf")]
def load(d): return (pd.read_csv(f"{IN}/{d}/raw_clean.csv"),pd.read_csv(f"{IN}/{d}/sim_sessions.csv"),
                     pd.read_csv(f"{IN}/{d}/pairs.csv"))
alld=[load(d) for _,d in DS]
XL={"A":24,"D":np.percentile(np.concatenate([r.duration_hours.values for r,_,_ in alld]),98),
    "G":np.percentile(np.concatenate([p.G_eff.values for _,_,p in alld]),97),
    "E":np.percentile(np.concatenate([r.energy.values for r,_,_ in alld]),99)}
FIGSZ=(1.42,1.16); RECT=[0.135,0.17,0.825,0.79]     # small uniform margins
for q,kind in QT:
  for di,(dk,ddir) in enumerate(DS):
    raw,sim,prs=alld[di]
    real={"A":raw.arrival_hour,"D":raw.duration_hours,"G":prs.G_eff,"E":raw.energy}[q]
    simd={"A":sim.arrival_hour,"D":sim.duration_h,"G":sim.gap_h,"E":sim.energy_kwh}[q]
    xm=XL[q]; fig=plt.figure(figsize=FIGSZ); ax=fig.add_axes(RECT)
    if kind=="pdf":
        b=np.linspace(0,xm,25 if q=="A" else 40)
        r=np.asarray(real,float); r=r[(r>=0)&np.isfinite(r)]
        s=np.asarray(simd,float); s=s[(s>=0)&np.isfinite(s)]
        ax.hist(r,bins=b,density=True,color=REAL_C,alpha=0.55,lw=0)
        h,_=np.histogram(s,bins=b,density=True); ax.stairs(h,b,color=SIM_C,lw=1.0)
        ax.set_yticks([])
    else:
        for data,c,lw,ls in [(real,REAL_C,1.2,"-"),(simd,SIM_C,1.0,"--")]:
            d=np.sort(np.asarray(data,float)); d=d[np.isfinite(d)]
            ax.plot(d,np.arange(1,len(d)+1)/len(d),color=c,lw=lw,ls=ls)
        ax.set_ylim(0,1.02); ax.set_yticks([0,.5,1] if di==0 else [])
    ax.set_xlim(0,xm); ax.grid(True,lw=0.35,color="0.9"); ax.tick_params(labelsize=5.6,pad=1.5)
    if q=="A": ax.set_xticks([0,12,24])
    else: ax.xaxis.set_major_locator(MaxNLocator(3,prune="upper",integer=True))
    fig.savefig(f"{OUT}/p_{q}_{dk}.pdf"); plt.close(fig)
print("wrote",len(QT)*len(DS),"panels")
