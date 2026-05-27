# jacobson_final_project

Analysis code for a longitudinal mobile sensing study of major depressive disorder.  
252 participants, 17,709 nights, GPS + smartphone + EMA data (June 2021 – March 2024).

---

## pipeline

```
pipeline.py  raw data -> nightly_summary/
build_dataset.py   nightly_summary/ -> dataset/dataset.csv
prepare_model_data.py dataset/dataset.csv -> dataset/lme_data.csv + ml_data.csv
run_analysis.py  dataset/lme_data.csv →->results/
```

---

## usage

```bash
# step 1 — run all pipeline stages (0–3)
python pipeline.py

# or run specific stages
python pipeline.py 3  # nightly summary only
python pipeline.py 1 2  # timelines + gps only
python pipeline.py diagnose  # inspect without rerunning

# step 2 — qc + feature engineering
python build_dataset.py

# step 3 — centering, lagging, train/test split
python prepare_model_data.py

# step 4 — analysis
python run_analysis.py
# or point to a specific file
python run_analysis.py dataset/lme_data.csv
```

---

## pipeline stages

| stage | input | output |
|-------|-------|--------|
| 0 | `unlock/` + `running_app_123/` | `unlock_clean/` |
| 1 | `sleep/` + `unlock_clean/` | `merges/` |
| 2 | GPS raw + `merges/` | `GPS Processing/.../parsed_gps/` |
| 3 | `merges/` + `parsed_gps/` + EMA | `nightly_summary/` |

---

## outputs

`results/` after running `run_analysis.py`:

```
lme_nested_results.csv   LRT chi2 + p_fdr for each outcome
lme_primary_coefs.csv  full coefficients for fatigue model
lme_all_outcomes.csv    coefficients across all 7 outcomes
sensitivity_results.csv  4 sensitivity configs
logit_sensitivity.csv  logit-scale check
mlvar_results.csv       bidirectional VAR paths
lgbm_results.csv    R² behavior-only vs all-features
lgbm_shap.png    SHAP feature importance
analysis_results.png   summary figure
```

---

## dependencies

```
python 3.11
pandas
numpy
statsmodels
lightgbm
scikit-learn
matplotlib
scipy
shap
```


---

## data access

Raw data is not included. Access requires IRB approval through Dartmouth College. Contact the Jacobson Lab for details.

