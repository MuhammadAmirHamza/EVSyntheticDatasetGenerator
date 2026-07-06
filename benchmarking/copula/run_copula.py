"""
Vine copula baseline on our datasets.

Semiparametric R-vine over the three session variables (arrival hour, plug-in
duration, energy): empirical marginals + a fitted regular-vine copula with
per-pair family selection (Gaussian / Student-t / Clayton / Gumbel / Frank /
Joe / BB1 / indep, AIC-selected). Marginals are matched by construction (inverse
empirical CDF), so the copula's job is purely the dependence structure.

    python run_copula.py <combined|norway|korea|pecan>

Output: results/copula_<dataset>.csv  (arrival_hour, duration_h, energy_kwh)
        -> read by benchmarking/compare.py / metrics.py / plots_and_metrics.py

Precedent: Einolander & Lahdelma (Energy 2022); Vyboh & Grmanova (2026, vine+CODINE).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import pyvinecopulib as pv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DATA = {  # same exact selected-profile files the other generators use
    'combined': ROOT/'dataset'/'combined'/'combined_sessions.csv',
    'acn': ROOT/'dataset'/'acn_caltech'/'acn_sessions.csv',
    'georgia': ROOT/'dataset'/'georgia'/'georgia_sessions.csv',
    'kim': ROOT/'dataset'/'kim-commercial'/'kim_commercial_sessions.csv',
    'combined_top50': ROOT/'dataset'/'combined_top50'/'combined_top50_sessions.csv',
    'norway':   ROOT/'artifacts'/'norway_sorensen'/'raw_clean.csv',
    'korea':    ROOT/'artifacts'/'korea_gist'/'raw_clean.csv',
    'pecan':    ROOT/'artifacts'/'pecanstreet'/'raw_clean.csv',
}
FAMILIES = [pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe,
            pv.BicopFamily.bb1, pv.BicopFamily.indep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATA))
    a = ap.parse_args()

    d = pd.read_csv(DATA[a.dataset]).dropna(subset=['arrival_hour', 'duration_hours', 'energy'])
    arr = d['arrival_hour'].to_numpy(float)
    dur = d['duration_hours'].to_numpy(float)
    en  = d['energy'].to_numpy(float)
    arr_shift = (arr - 6.0) % 24.0                     # 6 AM origin (avoid midnight wrap)
    X = np.column_stack([arr_shift, dur, en])
    N = len(X); print(f"[{a.dataset}] fit on {N} sessions, {X.shape[1]} variables")

    # 1) pseudo-observations  2) fit R-vine with per-pair family selection
    u = pv.to_pseudo_obs(X)
    ctrl = pv.FitControlsVinecop(family_set=FAMILIES, selection_criterion='aic', num_threads=4)
    cop = pv.Vinecop.from_data(u, controls=ctrl)
    print(cop)

    # 3) simulate uniforms  4) invert empirical marginals (matches real shape incl. tails)
    U = np.asarray(cop.simulate(N, seeds=[24, 7, 13, 99, 42]))
    sim_arr_shift = np.quantile(arr_shift, U[:, 0])
    out = pd.DataFrame({
        'arrival_hour': (sim_arr_shift + 6.0) % 24.0,
        'duration_h':   np.quantile(dur, U[:, 1]),
        'energy_kwh':   np.quantile(en,  U[:, 2]),
    })
    out = out[(out.duration_h > 0) & (out.energy_kwh > 0)].reset_index(drop=True)
    (HERE / 'results').mkdir(exist_ok=True)
    fp = HERE / 'results' / ('copula_%s.csv' % a.dataset)
    out.to_csv(fp, index=False)
    print('generated %d sessions -> %s' % (len(out), fp))


if __name__ == '__main__':
    main()
