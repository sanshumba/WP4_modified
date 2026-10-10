# School electricity prioritisation and Monte Carlo stress testing

Analysis code and derived inputs for **Portfolio prioritisation of public-school electricity management under uncertainty: an observed-data index with Monte Carlo stress testing**. The study covers 53 Western Cape schools from 1 December 2022 to 30 November 2023.

## Reproduce the results

The tested environment is Python 3.12.14; package versions are pinned in `requirements.txt` and recorded in `TESTED_ENVIRONMENT.json`. Run these commands from the repository root, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_priority_analysis.py
python scripts/reproduce_economic_stress_test.py
python scripts/reproduce_sensitivity_analysis.py
python scripts/reproduce_component_intervals.py
python scripts/reproduce_reporting.py
```

The execution order is required because later scripts read earlier outputs. The scripts regenerate 23 CSV files and three figures, each in PNG and PDF format. Running them replaces the included reference outputs under `outputs/`; input files are not modified. Actual execution versions are written to `outputs/runtime_versions.json`.

## Data and provenance

Original measurements: Samuels, Michael-Ahile and Booysen, [Mendeley Data, DOI 10.17632/nm2vc4v2br.1](https://doi.org/10.17632/nm2vc4v2br.1), licensed CC BY 4.0. The [Data in Brief article](https://doi.org/10.1016/j.dib.2025.111321) documents the source cleaning and imputation. Supplied files contain anonymised, derived analytic inputs. `INPUT_SHA256.json` records their checksums.

| File under `data/` | Contents |
|---|---|
| `school_observed_indicators.csv` | 53 rows: school identifier, observed days, annual kWh, retained annual cost in ZAR, mean daily peak in kW, daily-kWh coefficient of variation and mean absolute profile residual in kWh per half-hour. |
| `daily_energy_cost.csv` | 19,345 school-day rows: date, daily kWh, variable energy charges in ZAR and school-day status. |
| `processed_intervals.csv.gz` | 928,560 half-hour rows: timestamp, consumption in kWh, date, season, school-day status and energy tariff in ZAR/kWh. |

Secondary processing clips pooled interval consumption at the 0.1st and 99.9th percentiles (0.03 and 27.88 kWh). Consumption, peak and profile indicators and simulated energy savings use these clipped values. Calendar profiles are in-sample medians within school × school-day status × season × half-hour slot. Relative irregularity divides the mean absolute profile residual by mean interval consumption. Night use covers 00:00–05:59; the ratio of mean night to remaining-hours load is `3p/(1-p)`, where `p` is annual night share.

Retained annual cost uses supplied energy charges before secondary clipping plus ZAR 31,530.73 in annual service charges. The cost reconciliation replaces this with clipped energy charges plus the same service charge. Fixed service charges are excluded from simulated savings. Energy rates are ZAR 2.2940/kWh before 1 July 2023 and ZAR 2.6978/kWh thereafter. SPU1 classification is supplied by the panel and has not been verified against invoices.

Pooled clipping may disproportionately reduce consumption, peak indicators and simulated savings at larger schools. School-specific and no-secondary-clipping sensitivities were not performed. Imputation flags and verified enrolment/floor-area linkages are absent; the inputs cannot support raw-data reconstruction, observed-only sensitivity or intensity benchmarking.

## Analysis settings

- **Ranking:** min-max scaling and weights `4:3:3:2:3` for cost, consumption, peak, variability and relative irregularity. Weight sensitivity uses 5,000 Dirichlet draws, concentration 50 and seed 1041. The five-specification consensus comprises base, percentile, z-score, outlier scaling bounds and three-domain weighting; profile-direction reversals are separate tests.
- **Reference economics:** 10,000 draws per school over 180 days, seed 42. Consumption effects follow Triangular(0.02, 0.05, 0.08); fixed costs follow Triangular(0, 1000, 3000) ZAR. Linear terminal persistence is represented by its horizon-average effect factor.
- **Calendar economics:** 1 December 2022–29 May 2023 and 1 June–27 November 2023; 10,000 draws, seed 20261009. Days are sampled within school × month × weekday × school-day status, and target-date tariffs are applied. The illustrative delivery budget is Triangular(1000, 3000, 6000) ZAR upfront plus six payments of one monthly Triangular(0, 100, 300) ZAR draw. The combined case adds linear decay to 50% and a 10% annual discount rate.
- **Intervals:** 10,000 paired school-bootstrap draws, seed 1042. Break-even correlation includes all 53 schools, with S24 capped at 10× while retaining its highest rank. Intervals express conditional school-sampling uncertainty, not meter or imputation uncertainty.

## Outputs and manuscript tables

All filenames below are relative to `outputs/`.

| Outputs | Manuscript location |
|---|---|
| `portfolio_priority_ranking.csv` | Table 2; Supplementary Table S1 |
| `component_spearman_correlation.csv`, `component_correlation_intervals.csv` | Supplementary Tables S2 and S15 |
| `priority_score_sensitivity.csv`, `baseline_rank_comparison.csv`, `structural_sensitivity.csv`, `baseline_correlations.csv` | Table 3; Supplementary Tables S3, S4, S10 and S14 |
| `behaviour_fixed_cost_detail.csv`, `behaviour_fixed_cost_summary.csv`, `behaviour_effect_size_detail.csv`, `behaviour_effect_size_summary.csv`, `behaviour_persistence_detail.csv`, `behaviour_persistence_summary.csv` | Table 4; Supplementary Tables S5–S7 |
| `behaviour_break_even.csv`, `break_even_size_link.csv` | Section 5.5; Supplementary Table S8 |
| `structural_ranks.csv`, `structural_scores.csv`, `audit_capacity_sensitivity.csv` | Supplementary Tables S9 and S11 |
| `joint_priority_economics.csv` | Table 5 |
| `calendar_economics_detail.csv`, `calendar_economics_summary.csv` | Table 6; Supplementary Table S12 |
| `night_use.csv` | Supplementary Table S13 |
| `cost_reconciliation.csv` | Supplementary Method A2 |
| `figures/figure1_top15_priority_score.*`, `figures/figure2_priority_vs_kwh.*`, `figures/figure3_fixed_cost_sensitivity.*` | Figures 1–3 |

`verification.json` records the consensus set, annual service charge and maximum profile-residual discrepancy.

## Verification and AI assistance

A clean-copy run regenerated all 23 CSV outputs and matched the validated results using numeric tolerances of `rtol=1e-10` and `atol=1e-8`; nonnumeric fields were compared exactly. Input checksums were unchanged. Table and figure values were checked against the supplement. ChatGPT (OpenAI) assisted with code preparation and execution, as disclosed in the manuscript; the author remains responsible for the analyses and interpretation.
