"""
GMMNet baseline on our datasets  (Li et al. 2024, "Synthesis of Electric Vehicle
Charging Data: A Real-World Data-Driven Approach").

The authors' code (this repo) models the Shanghai fleet schema
[start_soc, end_soc, start_hour, duration, battery_capacity, veh_label, month,
location] with one conditional density network per variable (Beta / GMM /
Discrete), sampled jointly by Gibbs sampling. Our data has no SOC / battery /
geo — it has ENERGY directly — so we apply the *same method* to our three
continuous variables (arrival hour, duration, energy): three conditional GMM
networks + Gibbs sampling. The GMM class and Gibbs procedure mirror the authors'
Network.py / data_generation.py (embedding on the categorical location column is
dropped since we have no categorical conditioner).

    python run_gmmnet.py <combined|norway|korea|pecan>   [--epochs N]

Output: results/gmmnet_<dataset>.csv  (arrival_hour, duration_h, energy_kwh)
        -> read by benchmarking/compare.py
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.distributions as D
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # project root
torch.manual_seed(24); np.random.seed(24)

DATA = {  # finalized paper cohorts (same real sessions SPARC trained on)
    'norway':   ROOT/'paper_results'/'artifacts'/'norway_sorensen_N100_b12_s01'/'raw_clean.csv',
    'gist':     ROOT/'paper_results'/'artifacts'/'korea_gist_N030_b24_s01'/'raw_clean.csv',
    'kim':      ROOT/'paper_results'/'artifacts'/'kim_commercial_N025_b24_s01'/'raw_clean.csv',
    'acn':      ROOT/'paper_results'/'artifacts'/'acn_caltech_N015_b12_s01'/'raw_clean.csv',
    'combined': ROOT/'paper_results'/'artifacts'/'combined_all_N200_b24_s01'/'raw_clean.csv',
}
NVARS = 3   # arrival, duration, energy


class GMMCond(nn.Module):
    """One variable's conditional GMM density given the others (Li et al. GMM,
    minus the location embedding)."""
    def __init__(self, cond_dim, H1, H2, K, dim):
        super().__init__()
        self.l1 = nn.Linear(cond_dim, H1); self.bn1 = nn.BatchNorm1d(H1)
        self.l2 = nn.Linear(H1, H2);       self.bn2 = nn.BatchNorm1d(H2)
        self.mu = nn.Linear(H2, K); self.sg = nn.Linear(H2, K); self.wt = nn.Linear(H2, K)
        self.sp = nn.Softplus(beta=1, threshold=20); self.dim = dim

    def forward(self, x):                    # x: (B, NVARS) in [0,1]
        y = x[:, self.dim]
        cond = x[:, torch.arange(x.size(1)) != self.dim]
        h = torch.relu(self.bn1(self.l1(cond)))
        h = torch.relu(self.bn2(self.l2(h)))
        mus = self.sp(self.mu(h)); sigs = self.sp(self.sg(h)) + 1e-4
        w = torch.softmax(self.wt(h), dim=1)
        gmm = D.MixtureSameFamily(D.Categorical(w),
              D.Independent(D.Normal(mus.unsqueeze(-1), sigs.unsqueeze(-1)), 1))
        loss = -gmm.log_prob(y.unsqueeze(1)).mean()
        return gmm, loss


def reset_startT_6(h):        # 6 AM origin -> [0,1]  (authors' util.py)
    return ((h - 6.0) % 24.0) / 24.0

def reset_startT_0(n):        # invert -> hour in [0,24)
    return (n * 24.0 + 6.0) % 24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATA))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--steps', type=int, default=60)      # Gibbs iterations
    ap.add_argument('--mult', type=float, default=1.0)    # generate mult x N sessions (horizon proxy)
    ap.add_argument('--max_train', type=int, default=6000)  # subsample for speed; densities need few
    a = ap.parse_args()

    df = pd.read_csv(DATA[a.dataset]).dropna(subset=['arrival_hour', 'duration_hours', 'energy'])
    arr = reset_startT_6(df['arrival_hour'].to_numpy(float))
    dur = df['duration_hours'].to_numpy(float)
    en  = df['energy'].to_numpy(float)
    # min-max the two positive vars to [0,1]; keep bounds to invert
    d0, d1 = dur.min(), dur.max(); e0, e1 = en.min(), en.max()
    durn = (dur - d0) / (d1 - d0); enn = (en - e0) / (e1 - e0)
    X = torch.tensor(np.stack([arr, durn, enn], 1), dtype=torch.float32)
    N = len(X)
    if a.max_train and N > a.max_train:
        X = X[torch.randperm(N)[:a.max_train]]
    print(f"[{a.dataset}] fit on {len(X)} of {N} sessions", flush=True)

    nets = [GMMCond(NVARS - 1, 32, 32, 5, d) for d in range(NVARS)]
    opts = [torch.optim.Adam(n.parameters(), lr=5e-3) for n in nets]
    dl = torch.utils.data.DataLoader(X, batch_size=512, shuffle=True, drop_last=True)
    for ep in range(a.epochs):
        for n in nets: n.train()
        losses = []
        for b in dl:
            ls = []
            for n, o in zip(nets, opts):
                o.zero_grad(); _, l = n(b); l.backward(); o.step(); ls.append(l.item())
            losses.append(ls)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  NLL {np.mean(losses,0).round(3)}")

    # ---- Gibbs sampling ----
    for n in nets: n.eval()
    M = int(N * a.mult)
    x = torch.rand(M, NVARS)
    with torch.no_grad():
        for _ in range(a.steps):
            for d in range(NVARS):
                gmm, _ = nets[d](x)
                x[:, d] = torch.clamp(gmm.sample().squeeze(), 0.0, 1.0)
    s = x.numpy()
    out = pd.DataFrame({
        'arrival_hour': reset_startT_0(s[:, 0]),
        'duration_h':   s[:, 1] * (d1 - d0) + d0,
        'energy_kwh':   s[:, 2] * (e1 - e0) + e0,
    })
    out = out[(out.duration_h > 0) & (out.energy_kwh > 0)].reset_index(drop=True)
    (HERE / 'results').mkdir(exist_ok=True)
    suffix = '' if a.mult == 1.0 else ('_x%g' % a.mult)
    fp = HERE / 'results' / ('gmmnet_%s%s.csv' % (a.dataset, suffix))
    out.to_csv(fp, index=False)
    print('generated %d sessions -> %s' % (len(out), fp))


if __name__ == '__main__':
    main()
