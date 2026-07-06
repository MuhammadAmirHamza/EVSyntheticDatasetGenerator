#!/usr/bin/env python3
"""Publication-grade validation figures for SPARC (real vs simulated populations)."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

mpl.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman","Nimbus Roman No9 L","STIXGeneral","DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3, "lines.linewidth": 1.1, "axes.grid": True, "grid.color":"0.9", "grid.linewidth":0.4,
    "axes.axisbelow": True,
})
REAL_C, SIM_C = "0.45", "#1f6fb4"
COL2 = 7.16
IN  = "/sessions/awesome-friendly-euler/mnt/SDG/SDG/paper_results/artifacts"
OUT = "/sessions/awesome-friendly-euler/mnt/SDG/SPARC/figures"
os.makedirs(OUT, exist_ok=True)
DATASETS = [("Norway","norway_sorensen_N100_b12_s01"),("GIST-Res","korea_gist_N030_b24_s01"),
            ("GIST-Comm","kim_commercial_N025_b24_s01"),("ACN","acn_caltech_N015_b12_s01"),
            ("Combined","combined_all_N200_b24_s01")]
def load(d):
    return (pd.read_csv(f"{IN}/{d}/raw_clean.csv"),pd.read_csv(f"{IN}/{d}/sim_sessions.csv"),
            pd.read_csv(f"{IN}/{d}/pairs.csv"))
def cdf(ax,real,sim,xmax):
    for data,c,lw,ls in [(real,REAL_C,1.4,"-"),(sim,SIM_C,1.1,"--")]:
        d=np.sort(np.asarray(data,float)); d=d[np.isfinite(d)]; y=np.arange(1,len(d)+1)/len(d)
        ax.plot(d,y,color=c,lw=lw,ls=ls)
    ax.set_xlim(0,xmax)
def pdf(ax,real,sim,xmax,bins=40):
    b=np.linspace(0,xmax,bins+1)
    r=np.asarray(real,float); r=r[np.isfinite(r)&(np.asarray(real,float)>=0)]
    s=np.asarray(sim,float);  s=s[np.isfinite(s)&(np.asarray(sim,float)>=0)]
    ax.hist(r,bins=b,density=True,color=REAL_C,alpha=0.55,lw=0)
    h,_=np.histogram(s,bins=b,density=True); ax.stairs(h,b,color=SIM_C,lw=1.2); ax.set_xlim(0,xmax)

def fig_marginals():
    D=[(l,load(d)) for l,d in DATASETS]
    rows=[("Arrival hour (h)","Density"),("Plug-in duration (h)","Density"),
          ("Idle gap (h)","CDF"),("Energy (kWh)","CDF")]
    kinds=["pdf","pdf","cdf","cdf"]
    dur=np.concatenate([load(d)[0].duration_hours.values for _,d in DATASETS])
    gap=np.concatenate([load(d)[2].G_eff.values for _,d in DATASETS])
    ene=np.concatenate([load(d)[0].energy.values for _,d in DATASETS])
    xlims=[24,30,np.percentile(gap,97),np.percentile(ene,99)]   # duration capped at 30h
    nr,nc=4,len(D)
    fig,axes=plt.subplots(nr,nc,figsize=(COL2,4.3),sharex="row",sharey="row")
    for j,(lab,(raw,sim,prs)) in enumerate(D):
        real=dict(A=raw.arrival_hour,D=raw.duration_hours,G=prs.G_eff,E=raw.energy)
        simd=dict(A=sim.arrival_hour,D=sim.duration_h,G=sim.gap_h,E=sim.energy_kwh)
        for i in range(nr):
            ax=axes[i,j]; key="ADGE"[i]; xm=xlims[i]
            if kinds[i]=="pdf": pdf(ax,real[key],simd[key],xm,bins=24 if i==0 else 40)
            else:               cdf(ax,real[key],simd[key],xm)
            ax.tick_params(length=2.2,labelsize=6.3)
            if i==0: ax.set_title(lab,fontweight="bold",pad=4); ax.set_xticks([0,6,12,18,24])
            if j==0: ax.set_ylabel(rows[i][1],fontsize=8.5)     # Density / CDF
            if kinds[i]=="pdf": ax.yaxis.set_major_locator(MaxNLocator(3)); ax.tick_params(axis="y",labelsize=5.6)
            else: ax.set_yticks([0,0.5,1.0]); ax.set_ylim(0,1.03)
            if j==2: ax.set_xlabel(rows[i][0],fontsize=8.5)     # variable name, centred under row
            ax.grid(True,axis="both",lw=0.4,color="0.9")
            ax.margins(x=0)
    fig.align_ylabels(axes[:,0])
    fig.legend(handles=[Patch(facecolor=REAL_C,alpha=0.55,label="Real"),
                        Line2D([0],[0],color=SIM_C,ls="--",label="Simulated")],
               loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.005))
    fig.tight_layout(rect=(0,0,1,0.965),h_pad=0.45,w_pad=0.4)
    fig.savefig(f"{OUT}/fig_marginals_grid.pdf",bbox_inches="tight"); plt.close(fig); print("marginals")

def fig_sessions_day():
    labels,rmu,rse,smu,sse=[],[],[],[],[]
    for lab,d in DATASETS:
        raw,sim,_=load(d); raw=raw.copy()
        raw["date"]=pd.to_datetime(raw.abs_start,errors="coerce").dt.date
        rc=raw.groupby(["user_id","date"]).size(); sc=sim.groupby(["user_id","calendar_date"]).size()
        labels.append(lab); rmu.append(rc.mean()); rse.append(rc.std()/np.sqrt(len(rc)))
        smu.append(sc.mean()); sse.append(sc.std()/np.sqrt(len(sc)))
    x=np.arange(len(labels)); w=0.38
    fig,ax=plt.subplots(figsize=(3.5,2.05))
    ax.bar(x-w/2,rmu,w,yerr=rse,color=REAL_C,alpha=0.75,label="Real",error_kw=dict(lw=0.7,capsize=2))
    ax.bar(x+w/2,smu,w,yerr=sse,color=SIM_C,alpha=0.9,label="Simulated",error_kw=dict(lw=0.7,capsize=2))
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("Mean sessions / active day")
    top=max(max(np.add(rmu,rse)),max(np.add(smu,sse)))*1.34
    ax.set_ylim(0,top); ax.margins(x=0.04)
    ax.tick_params(length=2)
    ax.legend(frameon=False,loc="upper left",borderaxespad=0.7,handlelength=1.3)
    fig.tight_layout(pad=0.2); fig.savefig(f"{OUT}/fig_sessions_per_day.pdf",bbox_inches="tight",pad_inches=0.01); plt.close(fig); print("sessions")

def trans_real(prs,B):
    M=np.zeros((B,B))
    for a,b in zip(prs.bin_i.astype(int)-1,prs.bin_next.astype(int)-1):
        if 0<=a<B and 0<=b<B: M[a,b]+=1
    r=M.sum(1,keepdims=True); r[r==0]=1; return M/r
def trans_sim(sim,B):
    M=np.zeros((B,B))
    for _,g in sim.groupby(["user_id","day_type"]):
        bn=g.sort_values("calendar_date")["bin"].astype(int).values-1
        for a,b in zip(bn[:-1],bn[1:]):
            if 0<=a<B and 0<=b<B: M[a,b]+=1
    r=M.sum(1,keepdims=True); r[r==0]=1; return M/r
def fig_transitions(cmap="Blues"):
    lab,d = DATASETS[-1]                       # Combined pool only
    raw,sim,prs=load(d); B=int(max(prs.bin_i.max(),prs.bin_next.max(),sim.bin.max()))
    R=trans_real(prs,B); S=trans_sim(sim,B); vmax=max(R.max(),S.max())
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.95)); im=None
    for i,(M,name) in enumerate([(R,"Real"),(S,"Simulated")]):
        ax=axes[i]; ax.grid(False)
        im=ax.imshow(M,origin="lower",cmap=cmap,vmin=0,vmax=vmax,aspect="equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name,fontweight="bold",pad=3)
        ax.set_xlabel("next bin",fontsize=6.5)
    axes[0].set_ylabel("current bin",fontsize=6.5)
    fig.subplots_adjust(left=0.09,right=0.85,top=0.84,bottom=0.15,wspace=0.15)
    cax=fig.add_axes([0.875,0.17,0.026,0.60]); cb=fig.colorbar(im,cax=cax)
    cb.set_label(r"$P(\mathrm{next}\,|\,\mathrm{current})$",fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(f"{OUT}/fig_transition_heatmaps.pdf",bbox_inches="tight"); plt.close(fig); print("transitions")

def fig_dur_energy():
    fig,axes=plt.subplots(1,len(DATASETS),figsize=(COL2,1.75))
    for j,(lab,d) in enumerate(DATASETS):
        raw,sim,_=load(d); ax=axes[j]
        xm=np.percentile(raw.duration_hours,99); ym=np.percentile(raw.energy,99.5)
        ax.scatter(raw.duration_hours,raw.energy,s=2,c=REAL_C,alpha=0.25,lw=0,rasterized=True)
        ax.scatter(sim.duration_h,sim.energy_kwh,s=2,c=SIM_C,alpha=0.20,lw=0,rasterized=True)
        ax.set_xlim(0,xm); ax.set_ylim(0,ym); ax.set_title(lab,fontweight="bold",pad=3); ax.tick_params(length=2)
        if j==0: ax.set_ylabel("Energy (kWh)")
        ax.set_xlabel("Duration (h)")
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=REAL_C,label="Real",ms=3),
                        Line2D([0],[0],marker='o',ls='',color=SIM_C,label="Simulated",ms=3)],
               loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.12))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_duration_energy.pdf",bbox_inches="tight",dpi=600); plt.close(fig); print("dur-energy")

if __name__=="__main__":
    fig_marginals(); fig_sessions_day(); fig_transitions(); fig_dur_energy(); print("done ->",OUT)
