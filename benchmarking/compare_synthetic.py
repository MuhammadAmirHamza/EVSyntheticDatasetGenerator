"""Compare real EV sessions against two synthetic generators.

Inputs (paths are absolute relative to project root):
  - REAL:     benchmarking/raw_clean.csv
  - USER_SGD: artifacts/sim_sessions.csv         (Proposed approach)
  - EV_SDG:   latest CSV in benchmarking/EV-SDG-master/res/generated_samples/

Population-level metrics computed per variable:
  - KL divergence (real || synth) with Laplace-smoothed histograms
  - Jensen-Shannon divergence (symmetric, bounded in [0, ln 2])
  - Kolmogorov-Smirnov statistic
  - Wasserstein-1 distance (Earth-mover's)

Variables: arrival_hour, connected_time, energy, sessions_per_day.

Outputs (in benchmarking/comparison/):
  - metrics.csv         long-format (variable, comparison, metric, value)
  - metrics_pivot.csv   wide pivot
  - fig_compare_pdf.png 2x2 PDF overlay (matches results_generation.jl style)
  - fig_compare_cdf.png 2x2 CDF overlay (matches results_generation.jl style)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(r'D:\codes\probabilitistic codes\SMC files\v2\benchmarking').resolve()
ROOT = HERE.parent

REAL_CSV = ROOT / 'benchmarking' / 'raw_clean.csv'
USER_CSV = ROOT / 'artifacts' / 'sim_sessions.csv'
SDG_DIR  = ROOT / 'benchmarking' / 'EV-SDG-master' / 'res' / 'generated_samples'
OUT_DIR  = ROOT / 'benchmarking' / 'comparison'
OUT_DIR.mkdir(exist_ok=True)

# ── Figure styling (mirrors results_generation.jl PANEL_* constants) ──────────
FIG_W_IN          = 11.0     # 2x2 figure width (in)  ~ 1100 px @ 100 dpi
FIG_H_IN          = 8.2      # 2x2 figure height (in) ~ 820  px @ 100 dpi
PANEL_TITLE_SIZE  = 18
PANEL_LABEL_SIZE  = 18
PANEL_TICK_SIZE   = 14
PANEL_LEGEND_SIZE = 13
SAVE_DPI          = 130

REAL_COLOR    = 'black'
USER_COLOR    = 'steelblue'      # "ours" / proposed approach
EV_COLOR      = 'tomato'         # EV-SDG baseline
LINEWIDTH_CDF = 3.5              # matches Julia after recent bump

plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize':   PANEL_TITLE_SIZE,
    'axes.labelsize':   PANEL_LABEL_SIZE,
    'xtick.labelsize':  PANEL_TICK_SIZE,
    'ytick.labelsize':  PANEL_TICK_SIZE,
    'legend.fontsize':  PANEL_LEGEND_SIZE,
    'axes.grid':        False,
})

# ── Panel config: (key, panel_title, xlabel, x_clip_max_or_None) ──────────────
PANELS = [
    ('arrival_hour',     '(a) Arrival hour',        'Hour of day',        24.0),
    ('connected_time',   '(b) Plug-in duration',    'Duration (h)',       48.0),
    ('energy',           '(c) Energy required',     'Energy (kWh)',       80.0),
    ('sessions_per_day', '(d) Sessions per day',    'Sessions per day',   100.0),
]


def load_real():
    df = pd.read_csv(REAL_CSV)
    df['abs_start'] = pd.to_datetime(df['abs_start'])
    return pd.DataFrame({
        'arrival_hour':   df['abs_start'].dt.hour + df['abs_start'].dt.minute / 60.0,
        'connected_time': df['duration_hours'].astype(float),
        'energy':         df['energy'].astype(float),
        'date':           df['abs_start'].dt.date,
    })


def load_user_sgd():
    df = pd.read_csv(USER_CSV)
    return pd.DataFrame({
        'arrival_hour':   df['arrival_hour'].astype(float) % 24,
        'connected_time': df['duration_h'].astype(float),
        'energy':         df['energy_kwh'].astype(float),
        'date':           pd.to_datetime(df['calendar_date']).dt.date,
    })


def load_ev_sdg():
    files = sorted(SDG_DIR.glob('Generated sample*.csv'))
    if not files:
        raise FileNotFoundError(f'No EV-SDG generated samples in {SDG_DIR}')
    df = pd.read_csv(files[-1])
    print(f'  EV-SDG file: {files[-1].name}')
    return pd.DataFrame({
        'arrival_hour':   df['Arrival'].astype(float) % 24,
        'connected_time': df['Connected_time'].astype(float),
        'energy':         df['Energy_required'].astype(float),
        'date':           pd.to_datetime(df['Date']).dt.date,
    })


def compute_metrics(real, synth, n_bins=50):
    real = np.asarray(real)
    synth = np.asarray(synth)
    lo = min(real.min(), synth.min())
    hi = max(real.max(), synth.max())
    if hi == lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(real, bins=bins)
    q_hist, _ = np.histogram(synth, bins=bins)
    p = (p_hist + 1) / (p_hist.sum() + n_bins)  # Laplace smoothing
    q = (q_hist + 1) / (q_hist.sum() + n_bins)
    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    ks, _ = stats.ks_2samp(real, synth)
    wd = float(stats.wasserstein_distance(real, synth))
    return {'KL': kl, 'JS': js, 'KS': float(ks), 'W1': wd}


def empirical_cdf(x):
    s = np.sort(np.asarray(x))
    return s, np.arange(1, len(s) + 1) / len(s)


def clip(x, hi):
    if hi is None:
        return x
    return x[x <= hi]


def panel_pdf(ax, real, user, ev, title, xlabel, x_max):
    real_c = clip(real, x_max)
    user_c = clip(user, x_max)
    ev_c   = clip(ev,   x_max)
    lo = 0.0
    hi = x_max if x_max is not None else max(real.max(), user.max(), ev.max())
    bins = np.linspace(lo, hi, 49)

    # Proportion (matches Julia normalization=:probability): each bar = count / total.
    def weights(v): return np.full_like(v, 1.0 / len(v), dtype=float) if len(v) else None
    ax.hist(real_c, bins=bins, weights=weights(real_c),
            color=REAL_COLOR, alpha=0.30, label='Real')
    ax.hist(user_c, bins=bins, weights=weights(user_c),
            histtype='step', color=USER_COLOR, linewidth=2.2, label='Proposed')
    ax.hist(ev_c, bins=bins, weights=weights(ev_c),
            histtype='step', color=EV_COLOR, linewidth=2.2, label='EV-SDG')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Proportion')
    if x_max is not None:
        ax.set_xlim(0, x_max)


def panel_cdf(ax, real, user, ev, title, xlabel, x_max):
    real_c = clip(real, x_max)
    user_c = clip(user, x_max)
    ev_c   = clip(ev,   x_max)

    # Extend each CDF to the right edge at y=1 so all three lines visually
    # reach the same x_max (otherwise truncated distributions look incomplete).
    def with_tail(x, y, x_end):
        if x_end is None or len(x) == 0:
            return x, y
        return np.append(x, x_end), np.append(y, 1.0)

    rx, ry = with_tail(*empirical_cdf(real_c), x_max)
    ux, uy = with_tail(*empirical_cdf(user_c), x_max)
    ex, ey = with_tail(*empirical_cdf(ev_c),   x_max)

    ax.plot(rx, ry, color=REAL_COLOR, linewidth=LINEWIDTH_CDF, label='Real')
    ax.plot(ux, uy, color=USER_COLOR, linewidth=LINEWIDTH_CDF, linestyle='--', label='Proposed')
    ax.plot(ex, ey, color=EV_COLOR,   linewidth=LINEWIDTH_CDF, linestyle=':',  label='EV-SDG')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative probability')
    ax.set_ylim(0, 1.02)
    if x_max is not None:
        ax.set_xlim(0, x_max)


def make_grid(panel_fn, real, user, ev, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN))
    for ax, (key, title, xlabel, x_max) in zip(axes.flat, PANELS):
        panel_fn(ax, real[key], user[key], ev[key], title, xlabel, x_max)
    axes.flat[0].legend(loc='upper left', frameon=False)
    for ax in axes.flat[1:]:
        ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def main():
    print('Loading datasets:')
    real_df = load_real()
    user_df = load_user_sgd()
    ev_df   = load_ev_sdg()
    print(f'  Real      : {len(real_df):>7,} sessions')
    print(f'  User SGD  : {len(user_df):>7,} sessions')
    print(f'  EV-SDG    : {len(ev_df):>7,} sessions')
    print()

    # Materialize series dicts indexed by panel key (so panel_* functions stay simple).
    def to_series(df):
        return {
            'arrival_hour':     df['arrival_hour'].dropna().values,
            'connected_time':   df['connected_time'].dropna().values,
            'energy':           df['energy'].dropna().values,
            'sessions_per_day': df.groupby('date').size().values,
        }
    real_s = to_series(real_df)
    user_s = to_series(user_df)
    ev_s   = to_series(ev_df)

    # ── Metrics ───────────────────────────────────────────────────────────────
    rows = []
    for key, _, _, _ in PANELS:
        for synth_name, synth in [('user_sgd', user_s[key]), ('ev_sdg', ev_s[key])]:
            for k, v in compute_metrics(real_s[key], synth).items():
                rows.append({'variable': key, 'synth': synth_name, 'metric': k, 'value': v})
    df_metrics = pd.DataFrame(rows)
    pivot = df_metrics.pivot_table(
        index=['variable', 'metric'], columns='synth', values='value'
    ).round(4)
    pivot['winner'] = np.where(pivot['user_sgd'] < pivot['ev_sdg'], 'user_sgd',
                       np.where(pivot['user_sgd'] > pivot['ev_sdg'], 'ev_sdg', 'tie'))
    print('Population-level distance to Real (lower = closer):\n')
    print(pivot.to_string())
    print()
    df_metrics.to_csv(OUT_DIR / 'metrics.csv', index=False)
    pivot.to_csv(OUT_DIR / 'metrics_pivot.csv')

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_grid(panel_pdf, real_s, user_s, ev_s, OUT_DIR / 'fig_compare_pdf.png')
    make_grid(panel_cdf, real_s, user_s, ev_s, OUT_DIR / 'fig_compare_cdf.png')

    print(f'Metrics written: {OUT_DIR / "metrics_pivot.csv"}')
    print(f'Plots written  : {OUT_DIR / "fig_compare_pdf.png"}')
    print(f'                 {OUT_DIR / "fig_compare_cdf.png"}')


if __name__ == '__main__':
    main()
