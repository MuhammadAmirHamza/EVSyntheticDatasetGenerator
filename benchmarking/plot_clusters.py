import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

df = pd.read_csv('benchmarking/raw_clean.csv')
df['abs_start'] = pd.to_datetime(df['abs_start'])
df = df[df['abs_start'].dt.year == 2019].copy()
df = df[df['energy'] > 0]
df = df[df['duration_hours'] <= 24]

start_time = df['abs_start'].dt.hour + df['abs_start'].dt.minute / 60 + df['abs_start'].dt.second / 3600
departure_time = start_time + df['duration_hours']

X = np.column_stack([start_time.values, departure_time.values])
print(f'n sessions (2019, after EV-SDG cleaning): {len(df)}')

# Try GMM with 2/3/4 components and compare BIC — gives an empirical answer to "how many natural clusters"
print('\n--- Gaussian Mixture Model fit (informs natural cluster count) ---')
for k in [2, 3, 4, 5]:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0, n_init=3).fit(X)
    print(f'  k={k}: BIC={gmm.bic(X):,.0f}  AIC={gmm.aic(X):,.0f}')

# Best k
bics = {k: GaussianMixture(n_components=k, covariance_type='full', random_state=0, n_init=3).fit(X).bic(X)
        for k in [2, 3, 4, 5]}
best_k = min(bics, key=bics.get)
print(f'\nBest k by BIC: {best_k}')

gmm3 = GaussianMixture(n_components=3, covariance_type='full', random_state=0, n_init=3).fit(X)
labels_3 = gmm3.predict(X)
print('\nIf forced to 3 clusters (what EV-SDG needs):')
print(f'  cluster sizes: {dict(zip(*np.unique(labels_3, return_counts=True)))}')
print(f'  means (Start, Departure):')
for i, m in enumerate(gmm3.means_):
    print(f'    cluster {i}: start={m[0]:.1f}h  departure={m[1]:.1f}h  duration={m[1]-m[0]:.1f}h')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

axes[0, 0].scatter(X[:, 0], X[:, 1], s=3, alpha=0.25, c='steelblue')
axes[0, 0].plot([0, 24], [0, 24], 'k--', alpha=0.3, label='y=x (0h)')
axes[0, 0].plot([0, 24], [24, 48], 'r--', alpha=0.3, label='y=x+24 (24h)')
axes[0, 0].set_xlabel('Start_time (hour of day)')
axes[0, 0].set_ylabel('Departure_time (hours)')
axes[0, 0].set_title(f'Raw scatter — 2019, n={len(df)}')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 24); axes[0, 0].set_ylim(0, 48)

H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=[48, 96], range=[[0, 24], [0, 48]])
im = axes[0, 1].pcolormesh(xedges, yedges, H.T, cmap='viridis', shading='auto')
axes[0, 1].set_xlabel('Start_time'); axes[0, 1].set_ylabel('Departure_time')
axes[0, 1].set_title('2D density (0.5h bins)')
axes[0, 1].set_xlim(0, 24); axes[0, 1].set_ylim(0, 48)
plt.colorbar(im, ax=axes[0, 1])

colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(3):
    mask = labels_3 == i
    axes[1, 0].scatter(X[mask, 0], X[mask, 1], s=3, alpha=0.3, c=colors[i],
                        label=f'cluster {i} (n={mask.sum()})')
axes[1, 0].scatter(gmm3.means_[:, 0], gmm3.means_[:, 1], s=200, marker='X', c='red',
                    edgecolor='black', linewidth=2, label='means')
axes[1, 0].set_xlabel('Start_time'); axes[1, 0].set_ylabel('Departure_time')
axes[1, 0].set_title('Forced 3 clusters (GMM) — what EV-SDG assumes')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 24); axes[1, 0].set_ylim(0, 48)

axes[1, 1].hist(df['duration_hours'], bins=80, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('ConnectedTime (hours)')
axes[1, 1].set_ylabel('count')
axes[1, 1].set_title('Distribution of session durations')
axes[1, 1].set_xlim(0, 24)

plt.tight_layout()
plt.savefig('benchmarking/start_vs_departure_2019.png', dpi=140)
print('\nSaved: benchmarking/start_vs_departure_2019.png')
