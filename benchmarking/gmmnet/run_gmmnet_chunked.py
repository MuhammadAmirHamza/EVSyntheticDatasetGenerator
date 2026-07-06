# Checkpointed chunk-trainer for GMMNet on large datasets (e.g. combined-200),
# so a full 200-epoch fit can be built across several <45s sandbox calls.
# Reuses the canonical model/helpers from run_gmmnet.py (that script is untouched).
#   python run_gmmnet_chunked.py combined --epochs 50 --total 200 [--fresh]
# Repeat until it prints DONE; the final chunk runs Gibbs and writes gmmnet_combined.csv.
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from run_gmmnet import GMMCond, reset_startT_6, reset_startT_0, DATA, NVARS

HERE = Path(__file__).resolve().parent

def load_data(ds):
    df = pd.read_csv(DATA[ds]).dropna(subset=['arrival_hour','duration_hours','energy'])
    arr = reset_startT_6(df['arrival_hour'].to_numpy(float))
    dur = df['duration_hours'].to_numpy(float); en = df['energy'].to_numpy(float)
    d0,d1 = dur.min(),dur.max(); e0,e1 = en.min(),en.max()
    durn=(dur-d0)/(d1-d0); enn=(en-e0)/(e1-e0)
    X = torch.tensor(np.stack([arr,durn,enn],1), dtype=torch.float32)
    return X,(d0,d1,e0,e1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--total',type=int,default=200); ap.add_argument('--batch',type=int,default=512)
    ap.add_argument('--steps',type=int,default=60); ap.add_argument('--fresh',action='store_true')
    a=ap.parse_args()
    ck=HERE/'results'/f'{a.dataset}_gmmnet_ckpt.pt'; (HERE/'results').mkdir(exist_ok=True)
    X,bnds=load_data(a.dataset); N=len(X)
    nets=[GMMCond(NVARS-1,32,32,5,d) for d in range(NVARS)]
    opts=[torch.optim.Adam(n.parameters(),lr=5e-3) for n in nets]
    done=0
    if ck.exists() and not a.fresh:
        s=torch.load(ck)
        for n,st in zip(nets,s['nets']): n.load_state_dict(st)
        for o,st in zip(opts,s['opts']): o.load_state_dict(st)
        done=s['done']
    dl=torch.utils.data.DataLoader(X,batch_size=a.batch,shuffle=True,drop_last=True)
    target=min(a.total,done+a.epochs)
    for ep in range(done,target):
        for n in nets: n.train()
        for b in dl:
            for n,o in zip(nets,opts):
                o.zero_grad(); _,l=n(b); l.backward(); o.step()
    done=target
    torch.save({'nets':[n.state_dict() for n in nets],'opts':[o.state_dict() for o in opts],'done':done}, ck)
    print(f'[{a.dataset}] trained {done}/{a.total} epochs')
    if done>=a.total:
        for n in nets: n.eval()
        x=torch.rand(N,NVARS)
        with torch.no_grad():
            for _ in range(a.steps):
                for d in range(NVARS):
                    gmm,_=nets[d](x); x[:,d]=torch.clamp(gmm.sample().squeeze(),0.,1.)
        s=x.numpy(); d0,d1,e0,e1=bnds
        out=pd.DataFrame({'arrival_hour':reset_startT_0(s[:,0]),'duration_h':s[:,1]*(d1-d0)+d0,'energy_kwh':s[:,2]*(e1-e0)+e0})
        out=out[(out.duration_h>0)&(out.energy_kwh>0)].reset_index(drop=True)
        fp=HERE/'results'/f'gmmnet_{a.dataset}.csv'; out.to_csv(fp,index=False)
        os.remove(ck)
        print('DONE ->',fp,'(%d sessions)'%len(out))

if __name__=='__main__': main()
