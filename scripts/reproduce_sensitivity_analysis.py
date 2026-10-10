"""Compute structural sensitivities, night use, joint results and calendar economics."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
P = Path(__file__).resolve().parents[1]
ind = pd.read_csv(P / 'data/school_observed_indicators.csv').set_index('school_id')
ind['irr'] = ind.residual_mae / (ind.annualised_kwh / (365 * 48))
cols = ['annualised_cost_zar', 'annualised_kwh', 'mean_daily_peak_kw', 'daily_kwh_cv', 'irr']
x = ind[cols]
z = (x - x.min()) / (x.max() - x.min())
w = np.array([4, 3, 3, 2, 3]) / 15

# Structural scores retain the same five indicators and baseline weights.
scores = pd.DataFrame(index=ind.index)
scores['base'] = z @ w
scores['percentile'] = x.rank(pct=True) @ w
scores['z-score'] = (x - x.mean()) / x.std() @ w
sub = x.drop('S24')
scores['outlier'] = (x - sub.min()) / (sub.max() - sub.min()) @ w
scores['three-domain'] = (z.iloc[:, :3].mean(axis=1) + z.iloc[:, 3] + z.iloc[:, 4]) / 3
scores['reverse-variability'] = z @ w - 2 * w[3] * z.iloc[:, 3] + w[3]
scores['reverse-both'] = z @ w - 2 * w[3] * z.iloc[:, 3] - 2 * w[4] * z.iloc[:, 4] + w[3] + w[4]
scores['resource-only'] = z.iloc[:, :3].mean(axis=1)
ranks = scores.rank(ascending=False, method='min').astype(int)
sets = {c: set(ranks[c].nsmallest(11).index) for c in ranks}
cons = set.intersection(*(sets[c] for c in ['base', 'percentile', 'z-score', 'outlier', 'three-domain']))
rng = np.random.default_rng(1042)
ix = rng.integers(0, 53, (10000, 53))


def ci(a, b):
    """Return a paired school-bootstrap interval for a Spearman correlation."""
    ar = rankdata(np.asarray(a)[ix], axis=1)
    br = rankdata(np.asarray(b)[ix], axis=1)
    ar -= ar.mean(1)[:, None]
    br -= br.mean(1)[:, None]
    vals = (ar * br).sum(1) / np.sqrt((ar * ar).sum(1) * (br * br).sum(1))
    return np.quantile(vals, [0.025, 0.975]).tolist()

# Compare structural specifications, audit capacities and resource baselines.
rob = []
for c in scores:
    if c == 'base':
        continue
    low, high = ci(scores.base, scores[c])
    rob.append([c, spearmanr(scores.base, scores[c]).statistic, low, high, len(sets['base'] & sets[c]), int((ranks.base - ranks[c]).abs().max()), ','.join(sorted(sets[c] - sets['base'])), ','.join(sorted(sets['base'] - sets[c]))])
rob = pd.DataFrame(rob, columns=['specification', 'rho', 'ci_low', 'ci_high', 'overlap', 'max_shift', 'enter', 'leave'])
rob.to_csv(P / 'outputs/structural_sensitivity.csv', index=False)
ranks.to_csv(P / 'outputs/structural_ranks.csv')
scores.to_csv(P / 'outputs/structural_scores.csv')
cap = []
for k in [5, 8, 11, 16, 20]:
    b = set(ranks.base.nsmallest(k).index)
    for c in ['percentile', 'z-score', 'outlier', 'three-domain']:
        alt = set(ranks[c].nsmallest(k).index)
        cap.append([k, c, len(b & alt), len(b & alt) / k])
pd.DataFrame(cap, columns=['capacity', 'specification', 'overlap', 'fraction']).to_csv(P / 'outputs/audit_capacity_sensitivity.csv', index=False)
bl = []
for c in cols[:3]:
    low, high = ci(scores.base, ind[c])
    bl.append([c, spearmanr(scores.base, ind[c]).statistic, low, high])
pd.DataFrame(bl, columns=['baseline', 'rho', 'ci_low', 'ci_high']).to_csv(P / 'outputs/baseline_correlations.csv', index=False)

# Reconstruct in-sample calendar profiles and night-use measures.
hh = pd.read_csv(P / 'data/processed_intervals.csv.gz', usecols=['school_id', 'timestamp', 'kwh', 'date', 'season', 'is_school_day', 'tariff_zar_per_kwh'])
hh.timestamp = pd.to_datetime(hh.timestamp)
hh['slot'] = hh.timestamp.dt.hour * 2 + hh.timestamp.dt.minute // 30
med = hh.groupby(['school_id', 'is_school_day', 'season', 'slot']).kwh.transform('median')
mae = (hh.kwh - med).abs().groupby(hh.school_id).mean()
print('max residual discrepancy', float((mae - ind.residual_mae).abs().max()))
meta = hh.groupby(['school_id', 'date']).agg(is_school_day=('is_school_day', 'first')).reset_index()
daily = pd.read_csv(P / 'data/daily_energy_cost.csv')
daily = daily.drop(columns=[c for c in daily if c.startswith('is_school_day')])
daily = daily.merge(meta, on=['school_id', 'date'], validate='one_to_one')
night = hh[hh.timestamp.dt.hour < 6].groupby('school_id').kwh.sum()
night = pd.DataFrame({'night_kwh': night, 'night_share': night / ind.annualised_kwh})
night['night_rank'] = night.night_kwh.rank(ascending=False, method='min').astype(int)
night['night_to_day_mean_ratio'] = 3 * night.night_share / (1 - night.night_share)
night.to_csv(P / 'outputs/night_use.csv')
srv = pd.Series(31530.73, index=ind.index)

# Combine baseline attention ranks with reference programme economics.
fix = pd.read_csv(P / 'outputs/behaviour_fixed_cost_detail.csv')
bef = pd.read_csv(P / 'outputs/behaviour_break_even.csv').set_index('school_id')
joint = ind[['annualised_kwh']].join(ranks[['base', 'three-domain', 'resource-only', 'reverse-both']]).join(bef[['break_even_fixed_cost_multiplier', 'break_even_display']]).join(fix[fix.fixed_cost_multiplier.eq(1)].set_index('school_id')[['saving_median', 'saving_p10', 'saving_p90', 'prob_positive']])
joint['consensus'] = joint.index.isin(cons)
joint['night_rank'] = night.night_rank
joint.to_csv(P / 'outputs/joint_priority_economics.csv')

# Match sampled days to the calendar of each 180-day window.
daily['dt'] = pd.to_datetime(daily.date)
daily['month'] = daily.dt.dt.month
daily['weekday'] = daily.dt.dt.weekday
N = 10000
out = []
rng = np.random.default_rng(20261009)
for sid, g in daily.groupby('school_id'):
    pools = {key: h.daily_kwh.to_numpy() for key, h in g.groupby(['month', 'weekday', 'is_school_day'])}
    eff = rng.triangular(0.02, 0.05, 0.08, N)
    fixed = rng.triangular(0, 1000, 3000, N)
    upfront = rng.triangular(1000, 3000, 6000, N)
    monthly = rng.triangular(0, 100, 300, N)
    for start in ['2022-12-01', '2023-06-01']:
        targets = g.set_index('dt').loc[pd.date_range(start, periods=180)]
        energy = np.empty((N, 180))
        active = targets.is_school_day.to_numpy()
        rates = np.where(targets.index < pd.Timestamp('2023-07-01'), 2.294, 2.6978)
        for j, (_, t) in enumerate(targets.iterrows()):
            energy[:, j] = rng.choice(pools[t.month, t.weekday, t.is_school_day], N) * rates[j]
        for scope in ['all-days', 'school-days']:
            e = energy if scope == 'all-days' else energy * active
            gross = e.sum(1) * eff
            for budget, cost in [('low-incremental', fixed), ('delivery-budget', upfront + 6 * monthly)]:
                sav = gross - cost
                out.append([sid, start, str(targets.index[-1].date()), scope, budget, np.median(sav), np.quantile(sav, 0.1), np.quantile(sav, 0.9), np.mean(sav > 0), np.median(sav) > 0])
            # Linear terminal decay and discounted daily savings.
            factors = np.linspace(1, 0.5, 180) / 1.1 ** (np.arange(1, 181) / 365)
            gross2 = (e * factors).sum(1) * eff
            # One monthly cost draw is held constant across six payments.
            recurring = (monthly[:, None] / 1.1 ** (np.arange(30, 181, 30) / 365)).sum(1)
            sav = gross2 - upfront - recurring
            out.append([sid, start, str(targets.index[-1].date()), scope, 'delivery-decay-discount', np.median(sav), np.quantile(sav, 0.1), np.quantile(sav, 0.9), np.mean(sav > 0), np.median(sav) > 0])
new = pd.DataFrame(out, columns=['school_id', 'start', 'end', 'scope', 'budget', 'saving_median', 'saving_p10', 'saving_p90', 'prob_positive', 'viable'])
new.to_csv(P / 'outputs/calendar_economics_detail.csv', index=False)
summ = new.groupby(['start', 'end', 'scope', 'budget']).agg(median_saving=('saving_median', 'median'), median_p10=('saving_p10', 'median'), median_p90=('saving_p90', 'median'), median_prob=('prob_positive', 'median'), viable=('viable', 'sum')).reset_index()
summ.to_csv(P / 'outputs/calendar_economics_summary.csv', index=False)
print('CONSENSUS', sorted(cons))
print(rob.to_string(index=False))
print(summ.to_string(index=False))
print(joint.loc[['S8', 'S41', 'S18']].to_string())
print('night check', night.loc[['S24', 'S41', 'S18']].to_string())
json.dump({'consensus': sorted(cons), 'service_annual': float(srv.iloc[0]), 'max_residual_discrepancy': float((mae - ind.residual_mae).abs().max())}, open(P / 'outputs/verification.json', 'w'), indent=2)
