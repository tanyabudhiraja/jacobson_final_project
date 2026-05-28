# jacobson_final_project
Analysis code for a longitudinal mobile sensing study of major depressive disorder.  
252 participants, 17,709 nights, GPS + smartphone + EMA data (June 2021 – March 2024).

---

## pipeline
```
pipeline.py               raw data → nightly_summary/
build_dataset.py         nightly_summary/ → dataset/dataset.csv (QC + feature engineering)
prepare_model_data.py    dataset/dataset.csv → lme_data.csv + ml_data.csv (centering, lagging)
run_analysis.py          lme_data.csv → results/ (all analyses)
```

---

## usage
```bash
# step 1 — run all pipeline stages (0–3)
python pipeline.py

# or run specific stages
python pipeline.py 3        # stage 3 only (nightly summary)
python pipeline.py 1 2      # stages 1 & 2 (sleep merge + GPS parsing)
python pipeline.py diagnose # inspect without rerunning

# step 2 — QC + feature engineering
python build_dataset.py

# step 3 — centering, lagging, min nights enforcement
python prepare_model_data.py

# step 4 — mixed-effects models, VAR, sensitivity analyses
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
| 2 | GPS raw + `merges/` | `parsed_gps/` |
| 3 | `merges/` + `parsed_gps/` + EMA | `nightly_summary/` |

---

## outputs
After running `run_analysis.py`, `results/` contains:
```
lme_nested_results.csv     LRT χ² + p_fdr across 7 outcomes
lme_primary_coefs.csv      full coefficients for fatigue model
lme_all_outcomes.csv       coefficients across all 7 outcomes with p_fdr
sensitivity_results.csv    4 sensitivity configurations
logit_sensitivity.csv      logit-transformed fatigue check
mlvar_results.csv          multilevel VAR bidirectional paths
ols_results.csv            between-person OLS R^2 (behavior-only + all-features)
analysis_results.png       8-panel summary figure
```

---

## dependencies
```
python 3.11
pandas
numpy
statsmodels
scikit-learn
matplotlib
scipy
```

---

## data access
Raw data is not included. Access requires IRB approval through Dartmouth College. Contact the Jacobson Lab for details.

---

## paper
Budhiraja et al. "Evening GPS and Digital Behavior Do Not Predict Next-Morning Fatigue or Mood in Major Depressive Disorder: A Longitudinal Mobile Sensing Study." *In preparation.*
