"""Compute the five-component score and Dirichlet weight sensitivity."""

from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'school_observed_indicators.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
ind = pd.read_csv(DATA)
ind['mean_interval_kwh'] = ind['annualised_kwh'] / (365 * 48)
ind['relative_residual_mae'] = ind['residual_mae'] / ind['mean_interval_kwh']
components = {'cost_exposure': 'annualised_cost_zar', 'demand_intensity': 'annualised_kwh', 'peak_intensity': 'mean_daily_peak_kw', 'demand_variability': 'daily_kwh_cv', 'empirical_irregularity': 'relative_residual_mae'}

# Relative importance tiers are normalised to sum to one.
orig = {'cost_exposure': 0.2, 'demand_intensity': 0.15, 'peak_intensity': 0.15, 'demand_variability': 0.1, 'empirical_irregularity': 0.15}
W = {k: v / sum(orig.values()) for k, v in orig.items()}


def mm(s, lo=None, hi=None):
    """Apply min-max scaling using supplied or sample bounds."""
    lo = s.min() if lo is None else lo
    hi = s.max() if hi is None else hi
    return (s - lo) / (hi - lo)
score = pd.DataFrame({'school_id': ind.school_id})
for k, c in components.items():
    score[k] = mm(ind[c].astype(float))
score['priority_score'] = sum((W[k] * score[k] for k in W))
score['priority_rank'] = score.priority_score.rank(ascending=False, method='min').astype(int)
score['dominant_component'] = score[list(W)].idxmax(axis=1)
score = score.merge(ind[['school_id', 'annualised_kwh', 'annualised_cost_zar', 'mean_daily_peak_kw', 'daily_kwh_cv', 'residual_mae', 'relative_residual_mae']], on='school_id')

# Seeded Dirichlet perturbations preserve the declared concentration.
rng = np.random.default_rng(1041)
keys = list(W)
alpha = np.array([W[k] for k in keys]) * 50
counts = {s: 0 for s in score.school_id}
ranksum = {s: 0.0 for s in score.school_id}
M = score.set_index('school_id')[keys]
for _ in range(5000):
    w = rng.dirichlet(alpha)
    vals = pd.Series(M.values @ w, index=M.index)
    ranks = vals.rank(ascending=False, method='min')
    for sid in ranks.nsmallest(11).index:
        counts[sid] += 1
    for sid, r in ranks.items():
        ranksum[sid] += r
score['top11_probability'] = score.school_id.map({s: counts[s] / 5000 for s in counts})
score['mean_rank_under_weight_perturbation'] = score.school_id.map({s: ranksum[s] / 5000 for s in ranksum})
score.sort_values(['priority_rank', 'school_id']).to_csv(OUT / 'portfolio_priority_ranking.csv', index=False)
ind[['annualised_cost_zar', 'annualised_kwh', 'mean_daily_peak_kw', 'daily_kwh_cv', 'relative_residual_mae']].corr(method='spearman').to_csv(OUT / 'component_spearman_correlation.csv')
print(score.sort_values('priority_rank')[['school_id', 'priority_rank', 'priority_score', 'top11_probability']].head(11).to_string(index=False))
