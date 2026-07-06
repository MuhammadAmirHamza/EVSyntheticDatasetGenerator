"""
CTGAN / TVAE baseline (deep generative tabular).

CTGAN = Conditional Tabular GAN; TVAE = Tabular VAE (Xu et al., NeurIPS 2019,
"Modeling Tabular Data using Conditional GAN"). Both learn the joint distribution
of the three continuous session variables (arrival hour, duration, energy) and
sample new sessions. Uses the `ctgan` package (github.com/sdv-dev/CTGAN).

    python run_ctgan.py <combined|combined_top50|norway|korea|pecan> [--model ctgan|tvae] [--epochs N]

Output: results/ctgan_<dataset>.csv (or tvae_...)  columns arrival_hour, duration_h, energy_kwh
        -> read by benchmarking/compare.py / metrics.py / plots_and_metrics.py
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
from ctgan import CTGAN, TVAE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
np.random.seed(24); torch.manual_seed(24)

DATA = {
    'combined':        ROOT/'dataset'/'combined'/'combined_sessions.csv',
    'combined_top50':  ROOT/'dataset'/'combined_top50'/'combined_top50_sessions.csv',
    'acn':             ROOT/'dataset'/'acn_caltech'/'acn_sessions.csv',
    'georgia':         ROOT/'dataset'/'georgia'/'georgia_sessions.csv',
    'kim': ROOT/'dataset'/'kim-commercial'/'kim_commercial_sessions.csv',
    'norway':          ROOT/'artifacts'/'norway_sorensen'/'raw_clean.csv',
    'korea':           ROOT/'artifacts'/'korea_gist'/'raw_clean.csv',
    'pecan':           ROOT/'artifacts'/'pecanstreet'/'raw_clean.csv',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATA))
    ap.add_argument('--model', choices=['ctgan', 'tvae'], default='ctgan')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--subsample', type=int, default=0)   # 0 = all; else train on this many rows (speed)
    a = ap.parse_args()

    d = pd.read_csv(DATA[a.dataset]).dropna(subset=['arrival_hour', 'duration_hours', 'energy'])
    n_gen = len(d)
    if a.subsample and len(d) > a.subsample:
        d = d.sample(a.subsample, random_state=24).reset_index(drop=True)
    arr = (d['arrival_hour'].to_numpy(float) - 6.0) % 24.0        # 6 AM origin (avoid wrap)
    train = pd.DataFrame({'arrival': arr,
                          'duration': d['duration_hours'].to_numpy(float),
                          'energy': d['energy'].to_numpy(float)})
    N = len(train); print(f"[{a.dataset}] {a.model} fit on {N} sessions, {a.epochs} epochs")

    Model = CTGAN if a.model == 'ctgan' else TVAE
    kw = dict(epochs=a.epochs, batch_size=a.batch, cuda=False)
    if a.model == 'ctgan':
        kw['verbose'] = False
    m = Model(**kw)
    m.fit(train)
    s = m.sample(n_gen)

    out = pd.DataFrame({
        'arrival_hour': (np.clip(s['arrival'].to_numpy(float), 0, 24) + 6.0) % 24.0,
        'duration_h':   s['duration'].to_numpy(float),
        'energy_kwh':   s['energy'].to_numpy(float),
    })
    out = out[(out.duration_h > 0) & (out.energy_kwh > 0)].reset_index(drop=True)
    (HERE / 'results').mkdir(exist_ok=True)
    fp = HERE / 'results' / ('%s_%s.csv' % (a.model, a.dataset))
    out.to_csv(fp, index=False)
    print('generated %d sessions -> %s' % (len(out), fp))


if __name__ == '__main__':
    main()
