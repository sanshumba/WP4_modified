"""Compute component Spearman correlations and paired school-bootstrap intervals."""

from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import rankdata
P = Path(__file__).resolve().parents[1]
ind = pd.read_csv(P / 'data/school_observed_indicators.csv')
ind['relative_irregularity'] = ind.residual_mae / (ind.annualised_kwh / (365 * 48))
cols = ['annualised_cost_zar', 'annualised_kwh', 'mean_daily_peak_kw', 'daily_kwh_cv', 'relative_irregularity']
rng = np.random.default_rng(1042)
ix = rng.integers(0, 53, (10000, 53))
r = []

# Re-rank within each resample to handle repeated schools correctly.
for i, c in enumerate(cols):
    for cc in cols[i + 1:]:
        x = rankdata(ind[c].to_numpy()[ix], axis=1)
        y = rankdata(ind[cc].to_numpy()[ix], axis=1)
        x -= x.mean(1)[:, None]
        y -= y.mean(1)[:, None]
        z = (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
        lo, hi = np.quantile(z, [0.025, 0.975])
        r.append([c, cc, ind[c].corr(ind[cc], method='spearman'), lo, hi])
pd.DataFrame(r, columns=['component_1', 'component_2', 'rho', 'ci_low', 'ci_high']).to_csv(P / 'outputs/component_correlation_intervals.csv', index=False)
