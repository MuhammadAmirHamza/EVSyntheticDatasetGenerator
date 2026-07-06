"""
Expanded evaluation panel for the combined-dataset comparison.

For every generator it reports, against the REAL combined data:

  1. Population marginal fidelity   mean KL over (arrival, duration, energy)
  2. Per-user fidelity              mean KL(real_u || synth_u), averaged over users
                                    (weighted by sessions/user). Baselines are pooled
                                    (no user identity), so every user is scored against
                                    the same pooled output -> this is the heterogeneity test.
  3. Dependence structure          Kendall-tau matrix error + tail-dependence MAE on the
                                    joint (arrival, duration, energy).
  4. Distinguishability            classifier two-sample test (C2ST) AUC; 0.5 = ideal.
  5. Grid utility                  average daily load-profile MAE (kWh/h per session).
  6. Sequential structure          idle-gap KL (real vs synth). Only our method emits a
                                    gap; pooled baselines generate independent sessions.
  7. Privacy / memorisation        median distance-to-closest-record (DCR), vs a
                                    real-vs-real reference.

    python benchmarking/metrics.py           # prints table, writes metrics_panel.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
np.random.seed(24)

REAL_CSV = ROOT / 'dataset' / 'combined' / 'combined_sessions.csv'
VARS  = ['arrival_hour', 'duration_h', 'energy_kwh']
BINS  = {'arrival_hour': (0, 24, 24), 'duration_h': (0, 48, 48), 'energy_kwh': (0, 80, 40)}

# (label, path, colmap, has_user, has_gap)
GENERATORS = [
    ('Ours (per-user SMC)', ROOT/'artifacts'/'combined'/'sim_sessions.csv',
     {'arrival_hour':['arrival_hour'],'duration_h':['duration_h'],'energy_kwh':['energy_kwh']}, True, True),
    ('GMMNet (Li 2024)', HERE/'gmmnet'/'results'/'gmmnet_combined.csv',
     {'arrival_hour':['arrival_hour'],'duration_h':['duration_h'],'energy_kwh':['energy_kwh']}, False, False),
    ('EV-SDG', HERE/'evsdg'/'results'/'EVSDG_combined_generated.csv',
     {'arrival_hour':['Arrival'],'duration_h':['Connected_time'],'energy_kwh':['Energy_required']}, False, False),
    ('Vine copula', HERE/'copula'/'results'/'copula_combined.csv',
     {'arrival_hour':['arrival_hour'],'duration_h':['duration_h'],'energy_kwh':['energy_kwh']}, False, False),
]


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def hist(x, var):
    lo, hi, nb = BINS[var]
    h, _ = np.histogram(np.clip(np.asarray(x, float), lo, hi), bins=np.linspace(lo, hi, nb+1))
    return h.astype(float)

def pop_marginal_kl(real, gen):
    return float(np.mean([kl(hist(real[v], v), hist(gen[v], v)) for v in VARS]))

def per_user_kl(real_by_user, gen, gen_by_user):
    tot, wsum = 0.0, 0
    for u, ru in real_by_user.items():
        n = len(ru['arrival_hour'])
        su = gen_by_user.get(u, gen) if gen_by_user else gen     # matched user, else pooled
        if len(su['arrival_hour']) < 5:
            su = gen
        k = np.mean([kl(hist(ru[v], v), hist(su[v], v)) for v in VARS])
        tot += k * n; wsum += n
    return float(tot / wsum)

def pseudo(x):
    r = stats.rankdata(x)
    return r / (len(x) + 1.0)

def dependence(real, gen):
    pairs = [('arrival_hour','duration_h'), ('arrival_hour','energy_kwh'), ('duration_h','energy_kwh')]
    tau_err, tail = [], []
    for a, b in pairs:
        tr = stats.kendalltau(real[a], real[b]).statistic
        tg = stats.kendalltau(gen[a],  gen[b]).statistic
        tau_err.append(abs(tr - tg))
        ua, va = pseudo(real[a]), pseudo(real[b]); ug, vg = pseudo(gen[a]), pseudo(gen[b])
        for q, side in [(0.95, 'u'), (0.05, 'l')]:
            if side == 'u':
                lr = np.mean((ua > q) & (va > q)) / (1 - q)
                lg = np.mean((ug > q) & (vg > q)) / (1 - q)
            else:
                lr = np.mean((ua < q) & (va < q)) / q
                lg = np.mean((ug < q) & (vg < q)) / q
            tail.append(abs(lr - lg))
    return float(np.mean(tau_err)), float(np.mean(tail))

def c2st_auc(real, gen):
    Xr = np.stack([real[v] for v in VARS], 1); Xg = np.stack([gen[v] for v in VARS], 1)
    n = min(len(Xr), len(Xg), 6000)
    Xr = Xr[np.random.choice(len(Xr), n, replace=False)]
    Xg = Xg[np.random.choice(len(Xg), n, replace=False)]
    X = np.vstack([Xr, Xg]); y = np.r_[np.ones(n), np.zeros(n)]
    mu, sd = X.mean(0), X.std(0) + 1e-9; X = (X - mu) / sd
    clf = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=24)
    return float(np.mean(cross_val_score(clf, X, y, cv=4, scoring='roc_auc')))

def load_profile(a, d, e, cap=48.0, nb=24):
    prof = np.zeros(nb)
    a = np.asarray(a, float); d = np.minimum(np.asarray(d, float), cap); e = np.asarray(e, float)
    for ai, di, ei in zip(a, d, e):
        if di <= 0: continue
        p = ei / di; t = 0.0; cur = ai
        while t < di:
            hod = int(np.floor(cur)) % nb
            seg = min(np.floor(cur) + 1 - cur, di - t)
            if seg <= 0: seg = min(1.0, di - t)
            prof[hod] += p * seg; t += seg; cur += seg
    return prof / len(a)

def load_profile_mae(real, gen):
    pr = load_profile(real['arrival_hour'], real['duration_h'], real['energy_kwh'])
    pg = load_profile(gen['arrival_hour'],  gen['duration_h'],  gen['energy_kwh'])
    return float(np.mean(np.abs(pr - pg)))

def gap_kl(real_gap, gen_gap):
    e = np.linspace(0, 72, 37)
    hr, _ = np.histogram(np.clip(real_gap, 0, 72), bins=e)
    hg, _ = np.histogram(np.clip(gen_gap, 0, 72), bins=e)
    return kl(hr.astype(float), hg.astype(float))

def dcr(real, gen, ref_real):
    Xr = np.stack([ref_real[v] for v in VARS], 1)
    sd = Xr.std(0) + 1e-9; Xr = Xr / sd
    nn = NearestNeighbors(n_neighbors=1).fit(Xr)
    Xg = np.stack([gen[v] for v in VARS], 1) / sd
    m = min(len(Xg), 5000); Xg = Xg[np.random.choice(len(Xg), m, replace=False)]
    return float(np.median(nn.kneighbors(Xg)[0].ravel()))


def load_real():
    d = pd.read_csv(REAL_CSV).dropna(subset=['abs_start','abs_end','duration_hours','energy','user_id'])
    t = pd.to_datetime(d['abs_start']); te = pd.to_datetime(d['abs_end'])
    d = d.assign(_arr=t.dt.hour + t.dt.minute/60.0, _dur=d['duration_hours'].astype(float),
                 _en=d['energy'].astype(float), _t0=t, _t1=te)
    pooled = {'arrival_hour': d['_arr'].to_numpy(), 'duration_h': d['_dur'].to_numpy(),
              'energy_kwh': d['_en'].to_numpy()}
    by_user = {}
    gaps = []
    for u, g in d.sort_values('_t0').groupby('user_id'):
        by_user[u] = {'arrival_hour': g['_arr'].to_numpy(), 'duration_h': g['_dur'].to_numpy(),
                      'energy_kwh': g['_en'].to_numpy()}
        gp = (g['_t0'].shift(-1) - g['_t1']).dt.total_seconds().to_numpy()[:-1] / 3600.0
        gaps.append(gp[gp >= 0])
    return pooled, by_user, np.concatenate(gaps)


def load_gen(path, cmap, has_user, has_gap):
    if not Path(path).exists():
        return None
    g = pd.read_csv(path)
    def col(cands): return next((c for c in cands if c in g.columns), None)
    pooled = {v: np.asarray(g[col(cmap[v])], float) for v in VARS}
    by_user = None
    if has_user and 'user_id' in g.columns:
        by_user = {u: {v: np.asarray(gg[col(cmap[v])], float) for v in VARS}
                   for u, gg in g.groupby('user_id')}
    gap = np.asarray(g['gap_h'], float) if (has_gap and 'gap_h' in g.columns) else None
    return pooled, by_user, gap


if __name__ == '__main__':
    real, real_by_user, real_gap = load_real()
    # real-vs-real DCR reference (split)
    idx = np.random.permutation(len(real['arrival_hour'])); half = len(idx)//2
    ref_a = {v: real[v][idx[:half]] for v in VARS}; ref_b = {v: real[v][idx[half:]] for v in VARS}
    ref_dcr = dcr(ref_a, ref_b, ref_a)
    print(f"REAL combined: {len(real['arrival_hour'])} sessions | real-vs-real DCR ref = {ref_dcr:.3f}\n")

    rows = []
    for label, path, cmap, hu, hg in GENERATORS:
        g = load_gen(path, cmap, hu, hg)
        if g is None:
            print(f"{label}: file missing — skipped"); continue
        pooled, by_user, gap = g
        r = {'generator': label,
             'pop_KL':      round(pop_marginal_kl(real, pooled), 4),
             'perUser_KL':  round(per_user_kl(real_by_user, pooled, by_user), 4),
             'tau_err':     None, 'tail_MAE': None,
             'C2ST_AUC':    round(c2st_auc(real, pooled), 3),
             'load_MAE':    round(load_profile_mae(real, pooled), 4),
             'gap_KL':      round(gap_kl(real_gap, gap), 4) if gap is not None else np.nan,
             'DCR':         round(dcr(real, pooled, real), 3)}
        te, tl = dependence(real, pooled); r['tau_err'] = round(te, 4); r['tail_MAE'] = round(tl, 4)
        rows.append(r)

    df = pd.DataFrame(rows).set_index('generator')
    pd.set_option('display.width', 160)
    print(df.to_string())
    df.to_csv(HERE / 'metrics_panel.csv')
    print("\nsaved -> benchmarking/metrics_panel.csv")
    print('\nReading guide: pop_KL/perUser_KL/tau_err/tail_MAE/load_MAE/gap_KL/DCR -> lower is better; '
          'C2ST_AUC -> closer to 0.5 is better. gap_KL is NaN for pooled baselines. '
          'Vine copula uses EMPIRICAL marginals, so its pop/perUser KL are ~0 by construction '
          '(not a modelling win) -- judge it on tau_err / tail_MAE / C2ST.')
