"""Simulate reference savings, cost and effect sensitivities, and break-even costs."""

from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / 'data' / 'daily_energy_cost.csv')
ind = pd.read_csv(ROOT / 'data' / 'school_observed_indicators.csv')
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
annual = ind.set_index('school_id')['annualised_kwh']
rank = annual.rank(ascending=False, method='min')
rng = np.random.default_rng(42)
fr = []
pr = []
er = []
br = []

# Reuse sampled daily costs, effects and fixed costs across reference stresses.
for sid, g in d.groupby('school_id'):
    sid = str(sid)
    g = g.reset_index(drop=True)
    idx = rng.integers(0, len(g), size=(10000, 180))
    base = g.daily_energy_cost_zar.to_numpy()[idx].sum(axis=1)
    eff = rng.triangular(0.02, 0.05, 0.08, size=10000)
    fixed = rng.triangular(0, 1000, 3000, size=10000)
    gross = base * eff
    for m in (1, 1.5, 2, 2.5, 3, 4, 5):
        sav = gross - fixed * m
        fr.append([sid, annual.loc[sid], rank.loc[sid], m, np.median(sav), np.quantile(sav, 0.1), np.quantile(sav, 0.9), np.mean(sav > 0), np.median(sav) > 0])
    # Horizon-average factor for linear terminal persistence.
    for remain in (1, 0.75, 0.5):
        sav = base * (eff * ((1 + remain) / 2)) - fixed
        pr.append([sid, remain, np.median(sav), np.quantile(sav, 0.1), np.quantile(sav, 0.9), np.mean(sav > 0)])
    for em in (0.5, 0.75, 1, 1.5, 2):
        sav = base * (eff * em) - fixed
        er.append([sid, em, 0.02 * em, 0.05 * em, 0.08 * em, np.median(sav), np.mean(sav > 0), np.median(sav) > 0])
    # Right-censor above 10; otherwise locate the zero-median break-even cost.
    if np.median(gross - fixed * 10) > 0:
        be = np.nan
        disp = '>10'
    else:
        lo, hi = (0.0, 10.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.median(gross - fixed * mid) > 0:
                lo = mid
            else:
                hi = mid
        be = (lo + hi) / 2
        disp = f'{be:.3f}'
    br.append([sid, annual.loc[sid], rank.loc[sid], be, disp])
fixed_df = pd.DataFrame(fr, columns=['school_id', 'annualised_kwh', 'annual_kwh_rank', 'fixed_cost_multiplier', 'saving_median', 'saving_p10', 'saving_p90', 'prob_positive', 'viable_median_positive'])
fixed_df.to_csv(OUT / 'behaviour_fixed_cost_detail.csv', index=False)
pd.DataFrame(pr, columns=['school_id', 'effect_remaining_day180', 'saving_median', 'saving_p10', 'saving_p90', 'prob_positive']).to_csv(OUT / 'behaviour_persistence_detail.csv', index=False)
pd.DataFrame(er, columns=['school_id', 'effect_multiplier', 'triangular_effect_low', 'triangular_effect_mode', 'triangular_effect_high', 'saving_median', 'prob_positive', 'viable_median_positive']).to_csv(OUT / 'behaviour_effect_size_detail.csv', index=False)
pd.DataFrame(br, columns=['school_id', 'annualised_kwh', 'annual_kwh_rank', 'break_even_fixed_cost_multiplier', 'break_even_display']).to_csv(OUT / 'behaviour_break_even.csv', index=False)
print('Economic stress-test detail outputs written to', OUT)

# Summaries are medians across school-level results.
summary = []
for m, g in fixed_df.groupby('fixed_cost_multiplier'):
    viable = g[g.saving_median > 0]
    summary.append([m, g.saving_median.median(), g.prob_positive.median(), len(viable), viable.annualised_kwh.min()])
pd.DataFrame(summary, columns=['fixed_cost_multiplier', 'median_school_saving', 'median_prob_positive', 'viable_school_count', 'min_annual_kwh_among_viable']).to_csv(OUT / 'behaviour_fixed_cost_summary.csv', index=False)
summary = []
for m, g in pd.DataFrame(er, columns=['school_id', 'effect_multiplier', 'triangular_effect_low', 'triangular_effect_mode', 'triangular_effect_high', 'saving_median', 'prob_positive', 'viable_median_positive']).groupby('effect_multiplier'):
    summary.append([m, g.triangular_effect_low.iloc[0], g.triangular_effect_mode.iloc[0], g.triangular_effect_high.iloc[0], g.saving_median.median(), g.prob_positive.median(), int((g.saving_median > 0).sum())])
pd.DataFrame(summary, columns=['effect_multiplier', 'triangular_effect_low', 'triangular_effect_mode', 'triangular_effect_high', 'median_school_saving', 'median_prob_positive', 'viable_school_count']).to_csv(OUT / 'behaviour_effect_size_summary.csv', index=False)
summary = []
for m, g in pd.DataFrame(pr, columns=['school_id', 'effect_remaining_day180', 'saving_median', 'saving_p10', 'saving_p90', 'prob_positive']).groupby('effect_remaining_day180'):
    summary.append([m, g.saving_median.median(), g.prob_positive.median()])
pd.DataFrame(summary, columns=['effect_remaining_day180', 'median_school_saving', 'median_prob_positive']).to_csv(OUT / 'behaviour_persistence_summary.csv', index=False)
