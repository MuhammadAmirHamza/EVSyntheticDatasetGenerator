#!/usr/bin/env python3
"""Average WEEKDAY daily load profiles (kW vs hour), real vs simulated, per cohort.
Staircase (hourly) style, no smoothing, no normalisation."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"savefig.dpi":600,"pdf.fonttype":42,"font.family":"serif",
 "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
 "legend.fontsize":8,"axes.linewidth":0.6})
REAL_C,SIM_C="0.45","#1f6fb4"; COL2=7.16
IN="/sessions/awesome-friendly-euler/mnt/SDG/SDG/paper_results/artifacts"
OUT="/sessions/awesome-friendly-euler/mnt/SDG/SPARC/figures"
DS=[("Norway","norway_sorensen_N100_b12_s01"),("GIST-Res","korea_gist_N030_b24_s01"),
    ("GIST-Comm","kim_commercial_N025_b24_s01"),("ACN","acn_caltech_N015_b12_s01"),
    ("Combined","combined_all_N200_b24_s01")]
def profile(t0,D,E,ndays):
    prof=np.zeros(24)
    for a,d,e in zip(t0,D,E):
        if not (np.isfinite(d) and d>0 and np.isfinite(e) and e>0): continue
        p=e/d; t=float(a); end=a+d
        while t<end-1e-9:
            hb=np.floor(t)+1.0; seg=min(hb,end)-t; prof[int(np.floor(t))%24]+=p*seg; t+=seg
    return prof/max(ndays,1)
fig,axes=plt.subplots(1,len(DS),figsize=(COL2,1.9),sharex=True)
for ax,(lab,d) in zip(axes,DS):
    raw=pd.read_csv(f"{IN}/{d}/raw_clean.csv"); sim=pd.read_csv(f"{IN}/{d}/sim_sessions.csv")
    raw=raw[raw.day_type=="Weekday"]; sim=sim[sim.day_type=="Weekday"]   # weekday only
    rdays=pd.to_datetime(raw.abs_start,errors="coerce").dt.date.nunique()
    sdays=pd.to_datetime(sim.calendar_date,errors="coerce").dt.date.nunique()
    R=profile(raw.arrival_hour.values,raw.duration_hours.values,raw.energy.values,rdays)
    S=profile(sim.arrival_hour.values,sim.duration_h.values,sim.energy_kwh.values,sdays)
    h=np.arange(24)+0.5
    ax.fill_between(h,R,color=REAL_C,alpha=0.35,step="mid",lw=0)
    ax.plot(h,R,color=REAL_C,lw=1.3,drawstyle="steps-mid")
    ax.plot(h,S,color=SIM_C,lw=1.3,ls="--",drawstyle="steps-mid")
    ax.set_xlim(0,24); ax.set_xticks([0,6,12,18,24]); ax.set_ylim(bottom=0)
    ax.set_title(lab,fontweight="bold",pad=4); ax.grid(True,axis="y",lw=0.4,color="0.9")
    ax.tick_params(labelsize=6.8); ax.set_xlabel("Hour of day",fontsize=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Load (kW)",fontsize=8)
fig.legend(handles=[Line2D([0],[0],color=REAL_C,lw=3,label="Real"),
                    Line2D([0],[0],color=SIM_C,lw=2,ls="--",label="Simulated")],
           loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.12))
fig.tight_layout(rect=(0,0,1,0.98),w_pad=0.5); fig.savefig(f"{OUT}/fig_load_profiles.pdf",bbox_inches="tight")
print("wrote fig_load_profiles.pdf (weekday, staircase, absolute)")
