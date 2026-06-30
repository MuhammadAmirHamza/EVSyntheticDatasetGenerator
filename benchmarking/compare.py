"""
Head-to-head comparison on the COMBINED dataset (Norway 100 + Korea 30 + Pecan 10
= 140 exact selected profiles, 28,159 sessions).

Scores every generator against the real combined data with ONE identical
metric/binning, so the numbers are directly comparable.

    python benchmarking/compare.py

Generators are listed in GENERATORS below. To add a future baseline, drop its
synthetic-sessions CSV somewhere and add one line: (label, path, column-map).
A generator whose file is missing is simply skipped with a note.

Columns expected per generator (any alias in the list is accepted):
    arrival hour, plug-in duration (h), energy (kWh).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent          # benchmarking/
ROOT = HERE.parent                              # project root (…/SDG)

REAL_CSV = ROOT / 'dataset' / 'combined' / 'combined_sessions.csv'

# (label, csv path, {canonical: [accepted column names]})
GENERATORS = [
    ('OURS — per-user semi-Markov',
     ROOT / 'artifacts' / 'combined' / 'sim_sessions.csv',
     {'arrival_hour': ['arrival_hour'], 'duration_h': ['duration_h'],
      'energy_kwh': ['energy_kwh']}),
    ('EV-SDG — pooled baseline',
     HERE / 'evsdg' / 'results' / 'EVSDG_combined_generated.csv',
     {'arrival_hour': ['Arrival'], 'duration_h': ['Connected_time'],
      'energy_kwh': ['Energy_required']}),
    # --- add baseline #2 / #3 here, e.g.: ---
    # ('Gaussian copula', HERE/'copula'/'results'/'gen.csv',
    #  {'arrival_hour':['arrival_hour'],'duration_h':['duration_h'],'energy_kwh':['energy_kwh']}),
]

PANELS = [('arrival_hour', 0, 24, 24),
          ('duration_h',   0, 48, 48),
          ('energy_kwh',   0, 80, 40)]


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js(p, q):
    p = p + 1e-9; q = q + 1e-9; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def four(real, synth, lo, hi, nb):
    e = np.linspace(lo, hi, nb + 1)
    pr, _ = np.histogram(np.clip(real, lo, hi), bins=e)
    ps, _ = np.histogram(np.clip(synth, lo, hi), bins=e)
    return (kl(pr.astype(float), ps.astype(float)),
            js(pr.astype(float), ps.astype(float)),
            stats.ks_2samp(real, synth).statistic,
            stats.wasserstein_distance(real, synth))


def load_real():
    d = pd.read_csv(REAL_CSV).dropna(subset=['abs_start', 'duration_hours', 'energy'])
    t = pd.to_datetime(d['abs_start'])
    return {'arrival_hour': np.asarray(t.dt.hour + t.dt.minute / 60.0, float),
            'duration_h':   np.asarray(d['duration_hours'], float),
            'energy_kwh':   np.asarray(d['energy'], float)}


def load_gen(path, cmap):
    if not Path(path).exists():
        return None
    g = pd.read_csv(path)
    out = {}
    for k, cands in cmap.items():
        col = next((c for c in cands if c in g.columns), None)
        out[k] = np.asarray(g[col], float) if col else None
    return out


def report(label, real, gen):
    if gen is None:
        print(f"\n{label}: file not found — run its generator first.")
        return None
    print(f"\n=== {label} ===")
    print(f"{'variable':14s} {'KL':>8s} {'JS':>8s} {'KS':>8s} {'W1':>8s}")
    kls = []
    for name, lo, hi, nb in PANELS:
        k, j, ks, w = four(real[name], gen[name], lo, hi, nb)
        kls.append(k)
        print(f"{name:14s} {k:8.4f} {j:8.4f} {ks:8.4f} {w:8.3f}")
    m = float(np.mean(kls))
    print(f"{'MEAN KL':14s} {m:8.4f}")
    return m


if __name__ == '__main__':
    real = load_real()
    print("REAL combined: %d sessions" % len(real['energy_kwh']))
    scores = {}
    for label, path, cmap in GENERATORS:
        m = report(label, real, load_gen(path, cmap))
        if m is not None:
            scores[label] = m
    if len(scores) >= 2:
        print("\n" + "=" * 50)
        print("MEAN KL ranking (lower = better):")
        for lab, m in sorted(scores.items(), key=lambda kv: kv[1]):
            print(f"  {m:7.4f}  {lab}")
