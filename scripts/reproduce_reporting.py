"""Generate reporting summaries, the break-even interval, cost reconciliation and figures."""

from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
ranking = pd.read_csv(OUT / 'portfolio_priority_ranking.csv').set_index('school_id')
ind = pd.read_csv(ROOT / 'data/school_observed_indicators.csv').set_index('school_id')
base = pd.read_csv(OUT / 'baseline_correlations.csv')
labels = {'annualised_kwh': 'Annual kWh', 'annualised_cost_zar': 'Estimated annual cost', 'mean_daily_peak_kw': 'Mean daily peak'}
rows = []
for field, label in labels.items():
    overlap = len(set(ranking.priority_rank.nsmallest(11).index) & set(ind[field].nlargest(11).index))
    rows.append([label, float(base.set_index('baseline').loc[field, 'rho']), overlap])
pd.DataFrame(rows, columns=['baseline', 'spearman_with_priority_rank', 'top11_overlap']).to_csv(OUT / 'baseline_rank_comparison.csv', index=False)
rob = pd.read_csv(OUT / 'structural_sensitivity.csv').set_index('specification')
labels = {'percentile': 'percentile', 'z-score': 'z-score', 'outlier': 'min-max with S24 excluded from scaling bounds', 'three-domain': 'redundancy-controlled three-domain construction'}
rows = [[label, rob.loc[field, 'rho'], int(rob.loc[field, 'overlap']), int(rob.loc[field, 'max_shift'])] for field, label in labels.items()]
pd.DataFrame(rows, columns=['specification', 'spearman_vs_base', 'top11_overlap', 'max_abs_rank_shift']).to_csv(OUT / 'priority_score_sensitivity.csv', index=False)
be = pd.read_csv(OUT / 'behaviour_break_even.csv').set_index('school_id').reindex(ind.index)
a = ind.annualised_kwh.to_numpy()
b = be.break_even_fixed_cost_multiplier.fillna(10).to_numpy()
ix = np.random.default_rng(1042).integers(0, len(ind), (10000, len(ind)))
ar = rankdata(a[ix], axis=1)
br = rankdata(b[ix], axis=1)
ar -= ar.mean(axis=1)[:, None]
br -= br.mean(axis=1)[:, None]
corr = (ar * br).sum(axis=1) / np.sqrt((ar * ar).sum(axis=1) * (br * br).sum(axis=1))
low, high = np.quantile(corr, [0.025, 0.975])
pd.DataFrame([{'spearman_break_even_vs_annual_kwh': spearmanr(a, b).statistic, 'ci_low': low, 'ci_high': high, 'n_correlation': len(ind), 'n_uncensored': int(be.break_even_fixed_cost_multiplier.notna().sum()), 'n_right_censored_above_10': int(be.break_even_fixed_cost_multiplier.isna().sum()), 'censor_cap': 10, 'bootstrap_draws': 10000, 'bootstrap_seed': 1042}]).to_csv(OUT / 'break_even_size_link.csv', index=False)
hh = pd.read_csv(ROOT / 'data/processed_intervals.csv.gz', usecols=['school_id', 'kwh', 'tariff_zar_per_kwh'])
energy = (hh.kwh * hh.tariff_zar_per_kwh).groupby(hh.school_id).sum().reindex(ind.index)
x = ind[['annualised_cost_zar', 'annualised_kwh', 'mean_daily_peak_kw', 'daily_kwh_cv']].copy()
x['relative_irregularity'] = ind.residual_mae / (ind.annualised_kwh / (365 * 48))
w = np.array([4, 3, 3, 2, 3]) / 15
score_base = (x - x.min()) / (x.max() - x.min()) @ w
x.annualised_cost_zar = energy + 31530.73
score_reconciled = (x - x.min()) / (x.max() - x.min()) @ w
cost = pd.DataFrame({'retained_cost_zar': ind.annualised_cost_zar, 'processed_energy_cost_zar': energy, 'annual_service_charge_zar': 31530.73, 'reconciled_cost_zar': x.annualised_cost_zar, 'retained_minus_reconciled_zar': ind.annualised_cost_zar - x.annualised_cost_zar, 'base_score': score_base, 'reconciled_score': score_reconciled, 'base_rank': score_base.rank(ascending=False, method='min').astype(int), 'reconciled_rank': score_reconciled.rank(ascending=False, method='min').astype(int)})
cost.to_csv(OUT / 'cost_reconciliation.csv')
assert cost.base_rank.equals(cost.reconciled_rank), 'Reconciliation changes ranks'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'font.size': 12})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.png'), dpi=300)
    fig.savefig(FIG / (name + '.pdf'))
    plt.close(fig)
order = ranking.sort_values('priority_rank')
fig, ax = plt.subplots(figsize=(6.8, 2.9))
top = order.head(15)
ax.bar(top.index, top.priority_score, color='#1f77b4')
ax.set(xlabel='School', ylabel='Portfolio priority score')
ax.tick_params(axis='x', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
save(fig, 'figure1_top15_priority_score')
fig, ax = plt.subplots(figsize=(6.8, 3.55))
ax.scatter(order.annualised_kwh / 1000, order.priority_score, s=28, color='#1f77b4')
offsets = {'S24': (-27, 9), 'S48': (-14, 17), 'S43': (19, 8), 'S17': (43, 16), 'S2': (-53, 20), 'S47': (35, 0), 'S3': (-29, 27), 'S49': (31, -13), 'S8': (-18, 16), 'S25': (20, -14), 'S14': (25, -29)}
for sid in order.head(11).index:
    row = order.loc[sid]
    ax.annotate(sid, (row.annualised_kwh / 1000, row.priority_score), xytext=offsets[sid], textcoords='offset points', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#666666', 'lw': 0.5})
ax.set(xlabel='Annual electricity use (MWh)', ylabel='Portfolio priority score')
ax.spines[['top', 'right']].set_visible(False)
save(fig, 'figure2_priority_vs_kwh')
summary = pd.read_csv(OUT / 'behaviour_fixed_cost_summary.csv')
fig, ax = plt.subplots(figsize=(6.8, 3))
ax.plot(summary.fixed_cost_multiplier, summary.viable_school_count, marker='o', color='#1f77b4')
ax.set(xlabel='Behaviour-programme fixed-cost multiplier', ylabel='Schools retaining positive\nmedian saving')
ax.spines[['top', 'right']].set_visible(False)
save(fig, 'figure3_fixed_cost_sensitivity')
versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}
(OUT / 'runtime_versions.json').write_text(json.dumps(versions, indent=2) + '\n')
print('Reporting checks: break-even interval', low, high, '; all 53 reconciled ranks unchanged')
print('Figures regenerated in', FIG)
