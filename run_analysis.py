"""
run_analysis.py — nested lme + sensitivities + mlvar + lgbm

m1: outcome ~ outcome_lag1 + day_of_week (baseline)
m2: m1 + total_distance_m + n_locations + minutes_at_home (+ gps)
m3: m2 + phone vars (cwc + pm) (+ digital)

  python run_analysis.py
  python run_analysis.py dataset/lme_data.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/lme_data.csv")
OUT_DIR   = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# all ema outcomes
#fatigue_am is primary
OUTCOMES = ["fatigue_am", "anhedonia_mean", "depression_mean",
            "anxiety_mean", "worry_mean", "somatic_mean", "negative_affect_mean"]

# predictor groups for nested models
# gps now includes behavioral features (n_locations, minutes_at_home) per mindy's
# feedback — total_distance alone wasn't capturing "did they go places"
GPS_VARS   = ["total_distance_m_cwc",  "total_distance_m_pm",
              "n_locations_cwc", "n_locations_pm",
              "minutes_at_home_cwc",   "minutes_at_home_pm"]
PHONE_VARS = ["minutes_on_phone_before_bed_cwc", "minutes_on_phone_before_bed_pm",
              "minutes_social_media_cwc", "minutes_social_media_pm",
              "minutes_communication_cwc"]

# ts vars for lgbm person-level features 
TS_VARS = ["minutes_on_phone_before_bed", "minutes_social_media",
           "minutes_communication", "total_distance_m",
           "n_locations", "minutes_at_home",
           "any_phone", "fatigue_am"]





def fit_lme(data, outcome, preds, group="mlife_id"):
    # nested fits use reml=false so the likelihood ratio test is valid
    formula = f"{outcome} ~ " + " + ".join(preds) if preds else f"{outcome} ~ 1"
    data = data.reset_index(drop=True)
    m = smf.mixedlm(formula, data=data, groups=data[group]).fit(reml=False)
    if not m.converged:
        # fallback 
        m = smf.mixedlm(formula, data=data, groups=data[group]).fit(
            method="lbfgs", reml=False)
    return m


def lrt(m_small, m_big):
    # likelihood ratio test for nested models
    chi2 = 2 * (m_big.llf - m_small.llf)
    dfd  = len(m_big.params) - len(m_small.params)
    p    = stats.chi2.sf(chi2, dfd) if dfd > 0 else 1.0
    return chi2, dfd, p


def lme_nested(d, outcome):
    # m1 base, m2 +gps, m3 +digital
    lag  = f"{outcome}_lag1"
    base = [p for p in [lag, "day_of_week"] if p in d.columns]
    gps  = base + [p for p in GPS_VARS if p in d.columns]
    full = gps  + [p for p in PHONE_VARS if p in d.columns]

    # drop NaN 
    all_preds = list(dict.fromkeys(full))
    d = d.dropna(subset=[outcome] + [p for p in all_preds if p in d.columns])
    d = d.reset_index(drop=True)

    m1 = fit_lme(d, outcome, base)
    m2 = fit_lme(d, outcome, gps)
    m3 = fit_lme(d, outcome, full)

    return {
        "m1": m1, "m2": m2, "m3": m3,
        "lrt_gps":  lrt(m1, m2),
        "lrt_full": lrt(m2, m3),
    }


def icc_from(m):
    # icc = between-person variance / (between + residual)
    return m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale)


def ar1(s):
    s = s.dropna()
    return s.autocorr(lag=1) if len(s) >= 4 else np.nan


def rmssd(s):
    diff = np.diff(s.dropna().values)
    return np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan


def eng_features(grp, var):
    # per person features used for lgbm
    s = grp[var].dropna()
    if len(s) < 5: return {}
    n_w   = len(s) // 4
    win   = [s.iloc[i*n_w:(i+1)*n_w] for i in range(4)]
    means = [w.mean() for w in win if len(w)]
    varis = [w.var()  for w in win if len(w)]
    return {
        f"{var}_mean": s.mean(),
        f"{var}_sd": s.std(),
        f"{var}_rmssd": rmssd(s),
        f"{var}_inertia": ar1(s),
        f"{var}_stability": np.var(means) if means else np.nan,
        f"{var}_lumpiness": np.var(varis) if varis else np.nan,
        f"{var}_pct_zero":  (s == 0).mean(),
    }


def add_cwc_lag(d, var, group="mlife_id"):
    # build cwc + lag-1 columns for var (for mlvar)
    col, lag = f"{var}_cwc", f"{var}_cwc_lag1"
    if col not in d.columns:
        d[col] = d[var] - d.groupby(group)[var].transform("mean")
    d = d.sort_values([group, "calendarDate"]).copy()
    d[lag] = d.groupby(group)[col].shift(1)
    # invalidate lag across gaps > 2 days
    gap = (d["calendarDate"] - d.groupby(group)["calendarDate"].shift(1)).dt.days
    d.loc[gap > 2, lag] = np.nan
    return d



print("loading data")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["calendarDate"] = pd.to_datetime(df["calendarDate"])
print(f"  {len(df)} nights, {df['mlife_id'].nunique()} participants")


# icc from null model 
print("ICC (primary outcome)")
d0  = df.dropna(subset=["fatigue_am"]).reset_index(drop=True).copy()
m0  = smf.mixedlm("fatigue_am ~ 1", data=d0, groups=d0["mlife_id"]).fit(reml=True)
icc = icc_from(m0)
print(f"  ICC = {icc:.3f} ({icc:.0%} between-person, {1-icc:.0%} within-person)")



# nested lme per outcome + fdr
print("nested LME (3 models per outcome)")

fits, summaries = {}, []
for outcome in OUTCOMES:
    if outcome not in df.columns:
        print(f"  {outcome}: missing in dataset, skip")
        continue
    lag = f"{outcome}_lag1"
    d   = df.dropna(subset=[outcome] + ([lag] if lag in df.columns else [])).copy()
    if len(d) < 1000:
        print(f"  {outcome}: too few rows ({len(d)}), skip")
        continue

    r = lme_nested(d, outcome)
    fits[outcome] = r
    summaries.append({
        "outcome": outcome,
        "n_nights": len(d),
        "base_aic": r["m1"].aic,
        "gps_aic": r["m2"].aic,
        "full_aic": r["m3"].aic,
        "lrt_gps_chi2": r["lrt_gps"][0],
        "lrt_gps_df": r["lrt_gps"][1],
        "lrt_gps_p": r["lrt_gps"][2],
        "lrt_full_chi2": r["lrt_full"][0],
        "lrt_full_df": r["lrt_full"][1],
        "lrt_full_p": r["lrt_full"][2],
    })

summ = pd.DataFrame(summaries)
# fdr-bh over outcomes (7 tests for each lrt)
if len(summ) > 1:
    summ["lrt_gps_p_fdr"]  = multipletests(summ["lrt_gps_p"],  method="fdr_bh")[1]
    summ["lrt_full_p_fdr"] = multipletests(summ["lrt_full_p"], method="fdr_bh")[1]
summ.to_csv(OUT_DIR / "lme_nested_results.csv", index=False)

print(f"  {'outcome':24s} {'gps p':>9s} {'gps_fdr':>9s} {'digital p':>11s} {'dig_fdr':>9s}")
for i, r in summ.iterrows():
    print(f"  {r['outcome']:24s} {r['lrt_gps_p']:9.4f} {r['lrt_gps_p_fdr']:9.4f} "
          f"{r['lrt_full_p']:11.4f} {r['lrt_full_p_fdr']:9.4f}")



# primary-outcome coefficients with per-sd effect sizes
print("primary outcome coefficients (effect sizes per within-person SD)")

m_primary = fits["fatigue_am"]["m3"]
d_primary = df.dropna(subset=["fatigue_am", "fatigue_am_lag1"]).copy()

coef_rows = []
for pred in m_primary.params.index:
    if pred in ["Intercept", "Group Var"]: continue
    sd = d_primary[pred].std() if pred in d_primary.columns else np.nan
    coef_rows.append({
        "predictor": pred,
        "beta": m_primary.params[pred],
        "se": m_primary.bse[pred],
        "p": m_primary.pvalues[pred],
        "ci_lo": m_primary.conf_int().loc[pred, 0],
        "ci_hi": m_primary.conf_int().loc[pred, 1],
        "sd": sd,
        "beta_per_sd": m_primary.params[pred] * sd,  # interpretable size
        "sig": m_primary.pvalues[pred] < 0.05,
    })
coef = pd.DataFrame(coef_rows)
coef.to_csv(OUT_DIR / "lme_primary_coefs.csv", index=False)

for i, r in coef.iterrows():
    star = " *" if r["sig"] else ""
    print(f"  {r['predictor']:40s} b={r['beta']:+.4f}  per-sd={r['beta_per_sd']:+.3f}  "
          f"p={r['p']:.4f}{star}")



# all-outcome coefficient table + fdr per predictor
print("all-outcome coefficient table (fdr per predictor across outcomes)")

all_coefs = []
for outcome, fit_pack in fits.items():
    m   = fit_pack["m3"]
    lag = f"{outcome}_lag1"
    d   = df.dropna(subset=[outcome] + ([lag] if lag in df.columns else [])).copy()
    for pred in m.params.index:
        if pred in ["Intercept", "Group Var"]: continue
        sd = d[pred].std() if pred in d.columns else np.nan
        all_coefs.append({
            "outcome": outcome,
            "predictor": pred,
            "beta": m.params[pred],
            "se": m.bse[pred],
            "p": m.pvalues[pred],
            "beta_per_sd": m.params[pred] * sd,
            "sig": m.pvalues[pred] < 0.05,
        })

all_coefs_df = pd.DataFrame(all_coefs)
# fdr per predictor across the 7 outcomes
for pred, grp in all_coefs_df.groupby("predictor"):
    if len(grp) > 1:
        all_coefs_df.loc[grp.index, "p_fdr"] = multipletests(grp["p"], method="fdr_bh")[1]
all_coefs_df.to_csv(OUT_DIR / "lme_all_outcomes.csv", index=False)
print(f"  saved {len(all_coefs_df)} rows to lme_all_outcomes.csv")



# sensitivity: stricter sleep validation + alt min-nights
print("sensitivity (primary outcome only)")

# each config subsets the data, re-runs the nested model, tracks key stats
# strict_sleep = exclude MANUAL/AUTO_MANUAL (rank > 3); ENHANCED + AUTO FINAL/TENTATIVE only
configs = [
    ("main", {}),
    ("strict_sleep", {"sleep_max": 3}),
    ("min_nights_50", {"min_nights": 50}),
    ("min_nights_60", {"min_nights": 60}),
]

sens = []
for label, cfg in configs:
    d = df.dropna(subset=["fatigue_am", "fatigue_am_lag1"]).copy()
    if "sleep_max" in cfg:
        d = d[d["sleep_validation_rank"] <= cfg["sleep_max"]]
    # always re-enforce min nights after subsetting
    cnt  = d.groupby("mlife_id").size()
    keep = cnt[cnt >= cfg.get("min_nights", 30)].index
    d    = d[d["mlife_id"].isin(keep)]

    if len(d) < 1000:
        print(f"  {label:18s} too few rows ({len(d)}), skip")
        continue

    r = lme_nested(d, "fatigue_am")
    sens.append({
        "config": label,
        "n_nights": len(d),
        "n_pids": d["mlife_id"].nunique(),
        "gps_p": r["lrt_gps"][2],
        "digital_p": r["lrt_full"][2],
        # track the one significant predictor from the main analysis
        "comm_b": r["m3"].params.get("minutes_communication_cwc", np.nan),
        "comm_p": r["m3"].pvalues.get("minutes_communication_cwc", np.nan),
    })

sens_df = pd.DataFrame(sens)
sens_df.to_csv(OUT_DIR / "sensitivity_results.csv", index=False)
for i, r in sens_df.iterrows():
    print(f"  {r['config']:18s} n={r['n_nights']:5d}  pids={r['n_pids']:3d}  "
          f"gps_p={r['gps_p']:.3f}  digital_p={r['digital_p']:.3f}  "
          f"comm_b={r['comm_b']:+.3f} comm_p={r['comm_p']:.3f}")


# logit-transform sensitivity 
print("logit-transform sensitivity (primary outcome)")

# build predictor list first so we can drop NaN on all of them before fitting
preds_logit = [p for p in ["fatigue_am_lag1", "day_of_week"] + GPS_VARS + PHONE_VARS
               if p in df.columns]

d_log = df.dropna(subset=["fatigue_am", "fatigue_am_lag1"]).copy()
d_log = d_log.dropna(subset=[p for p in preds_logit if p in d_log.columns])
d_log = d_log.reset_index(drop=True)

# epsilon for log
eps = 0.5
p_arr = (d_log["fatigue_am"] + eps) / (100 + 2 * eps)
d_log["fatigue_logit"] = np.log(p_arr / (1 - p_arr))

m_logit = smf.mixedlm("fatigue_logit ~ " + " + ".join(preds_logit),
                       data=d_log, groups=d_log["mlife_id"]).fit(reml=False)

print(f"  {'predictor':40s} {'raw b':>10s} {'logit b':>10s} {'logit p':>10s}")
logit_rows = []
for pred in m_logit.params.index:
    if pred in ["Intercept", "Group Var"]: continue
    raw_match = coef[coef["predictor"] == pred]["beta"].values
    raw_b = raw_match[0] if len(raw_match) else np.nan
    logit_b = m_logit.params[pred]
    logit_p = m_logit.pvalues[pred]
    logit_rows.append({"predictor": pred, "raw_beta": raw_b,
                       "logit_beta": logit_b, "logit_p": logit_p,
                       "logit_sig": logit_p < 0.05})
    print(f"  {pred:40s} {raw_b:+10.4f} {logit_b:+10.4f} {logit_p:10.4f}")

pd.DataFrame(logit_rows).to_csv(OUT_DIR / "logit_sensitivity.csv", index=False)


# mlvar — model raw outcome with cwc'd lag predictors
print("mlVAR (raw outcome + cwc'd lag predictor)")

df_var = df.copy()
# fatigue_am IS in lme_data.csv but minutes_on_phone_before_bed isn't
if "minutes_on_phone_before_bed" not in df_var.columns:
    df_var["minutes_on_phone_before_bed"] = (
        df_var["minutes_on_phone_before_bed_cwc"] +
        df_var["minutes_on_phone_before_bed_pm"]
    )
df_var = add_cwc_lag(df_var, "fatigue_am")
df_var = add_cwc_lag(df_var, "minutes_on_phone_before_bed")


def run_var(d, outcome, lag_preds, label):
    needed  = [outcome] + lag_preds
    d2  = d.dropna(subset=needed).reset_index(drop=True)
    formula = f"{outcome} ~ " + " + ".join(lag_preds + ["day_of_week"])
    # try default fit; fall back to lbfgs only if that fails.
    try:
        m = smf.mixedlm(formula, data=d2, groups=d2["mlife_id"]).fit(reml=False)
        if not m.converged:
            raise RuntimeError("default optimizer did not converge")
    except Exception:
        m = smf.mixedlm(formula, data=d2, groups=d2["mlife_id"]).fit(
            method="lbfgs", reml=False)

    print(f"  {label} (converged: {m.converged})")
    rows = []
    for pred in m.params.index:
        if pred in ["Intercept", "Group Var"]: continue
        sig = m.pvalues[pred] < 0.05
        print(f"    {pred:42s} b={m.params[pred]:+.4f}  p={m.pvalues[pred]:.4f}"
              + (" *" if sig else ""))
        rows.append({"equation": label, "predictor": pred,
                     "beta": m.params[pred], "p": m.pvalues[pred], "sig": sig})
    return m, pd.DataFrame(rows)


m_fat, c_fat = run_var(df_var, "fatigue_am",
    ["fatigue_am_cwc_lag1", "minutes_on_phone_before_bed_cwc_lag1"], "fatigue")
m_pho, c_pho = run_var(df_var, "minutes_on_phone_before_bed",
    ["minutes_on_phone_before_bed_cwc_lag1", "fatigue_am_cwc_lag1"], "phone")

var_results = pd.concat([c_fat, c_pho], ignore_index=True)
var_results.to_csv(OUT_DIR / "mlvar_results.csv", index=False)

# contemporaneous association (cwc-cwc, same night)
clean = df_var.dropna(subset=["fatigue_am_cwc", "minutes_on_phone_before_bed_cwc"])
contemp_r, contemp_p = stats.pearsonr(
    clean["fatigue_am_cwc"], clean["minutes_on_phone_before_bed_cwc"])
print(f"  contemporaneous r={contemp_r:.4f} p={contemp_p:.4f}")


def get_beta(eq, substr):
    rows = var_results[(var_results["equation"] == eq) &
                       var_results["predictor"].str.contains(substr, na=False)]
    return (rows.iloc[0]["beta"], rows.iloc[0]["p"]) if not rows.empty else (0.0, 1.0)


fat_ar = get_beta("fatigue", "fatigue.*lag1|lag1.*fatigue")
pho_ar = get_beta("phone",   "phone.*lag1|lag1.*phone")
p2f    = get_beta("fatigue", "phone.*lag1|lag1.*phone")
f2p    = get_beta("phone",   "fatigue.*lag1|lag1.*fatigue")


# lgbm person-level features predicting chronic (mean) fatigue
print("LightGBM (person-level)")
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

rows = []
for pid, grp in df.groupby("mlife_id"):
    row = {"mlife_id": pid, "target": grp["fatigue_am"].mean(), "n_nights": len(grp)}
    for var in TS_VARS:
        if var in grp.columns:
            row.update(eng_features(grp, var))
    # cross variable network density = mean |ar(1)| across ts vars
    ar_vals = [abs(ar1(grp[v].dropna())) for v in TS_VARS
               if v in grp.columns and len(grp[v].dropna()) >= 4]
    row["network_density"] = np.mean(ar_vals) if ar_vals else np.nan
    rows.append(row)

person_df = pd.DataFrame(rows).dropna(subset=["target"])
print(f"  {len(person_df)} participants, {len(person_df.columns) - 3} features")

X_all= person_df[[c for c in person_df.columns if c not in ["mlife_id", "target"]]]
X_behav = X_all[[c for c in X_all.columns if "fatigue" not in c]]  # honest: no fatigue features
y = person_df["target"]
X_all = X_all.fillna(X_all.median())
X_behav = X_behav.fillna(X_behav.median())


def cv_lgbm(X, y, label):
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                  num_leaves=15, min_child_samples=10,
                  random_state=42, verbose=-1)
    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof[va] = m.predict(X.iloc[va])
    r2   = r2_score(y, oof)
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  {label}: R2={r2:.3f}  RMSE={rmse:.2f}")
    final = lgb.LGBMRegressor(**params)
    final.fit(X, y)
    return oof, r2, rmse, final


oof_all,   r2_all,   rmse_all,   model_all   = cv_lgbm(X_all,   y, "all features  ")
oof_behav, r2_behav, rmse_behav, model_behav = cv_lgbm(X_behav, y, "behavior only ")

pd.DataFrame({"all_r2": [r2_all], "behav_r2": [r2_behav],
              "all_rmse": [rmse_all], "behav_rmse": [rmse_behav]}
             ).to_csv(OUT_DIR / "lgbm_results.csv", index=False)


# shap on circular model
print("SHAP importance")
try:
    import shap
    expl = shap.TreeExplainer(model_all)
    sv   = expl.shap_values(X_all)
    imp  = (pd.Series(np.abs(sv).mean(axis=0), index=X_all.columns)
            .sort_values(ascending=False).head(20).sort_values())
    cols = ["tomato" if "fatigue" in c else "steelblue" for c in imp.index]

    fig_s, ax_s = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax_s, color=cols)
    ax_s.set_xlabel("mean |SHAP value|", fontsize=9)
    ax_s.set_title("LightGBM feature importance (all-features model)\n"
                   "red = fatigue features  blue = behavior features", fontsize=10)
    for i, v in enumerate(imp.values):
        ax_s.text(v + max(imp) * 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    fig_s.savefig(OUT_DIR / "lgbm_shap.png", dpi=130, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  saved results/lgbm_shap.png")
except ImportError:
    print("  shap not installed — skip (pip install shap)")
except Exception as e:
    print(f"  shap failed: {e}")


# fig

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Evening behavior and next-morning fatigue/mood (MDD cohort)",
             fontsize=13, y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4,
                        top=0.93, bottom=0.06, left=0.08, right=0.97)

# panel 1: study overview
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis("off")
ax0.set_title("Study overview", fontsize=10)
rows_txt = [
    ("Participants", str(df["mlife_id"].nunique())),
    ("Nights", f"{len(df):,}"),
    ("ICC", f"{icc:.3f}"),
    ("Between-person", f"{icc:.0%}"),
    ("Within-person", f"{1-icc:.0%}"),
    ("Fatigue mean", f"{df['fatigue_am'].mean():.1f} / 100"),
]
for i, (lbl, val) in enumerate(rows_txt):
    y_ = 0.9 - i * 0.15
    ax0.text(0.02, y_, lbl, transform=ax0.transAxes, fontsize=9, color="gray", va="top")
    ax0.text(0.98, y_, val, transform=ax0.transAxes, fontsize=9,
             fontweight="bold", va="top", ha="right")

# panel 2: nested-model LRTs across outcomes (the central result)
ax1 = fig.add_subplot(gs[0, 1:])
ax1.set_title("Does each model layer improve fit? (LRT -log10 p_fdr)", fontsize=10)
yp = np.arange(len(summ))
ax1.scatter(-np.log10(summ["lrt_gps_p_fdr"]),  yp - 0.15, color="steelblue",
            s=55, label="GPS over baseline", zorder=3)
ax1.scatter(-np.log10(summ["lrt_full_p_fdr"]), yp + 0.15, color="tomato",
            s=55, label="+ digital over GPS", zorder=3)
ax1.axvline(-np.log10(0.05), color="black", lw=0.8, ls="--", label="p_fdr = .05")
ax1.set_yticks(yp)
ax1.set_yticklabels([o.replace("_", " ") for o in summ["outcome"]], fontsize=8)
ax1.set_xlabel("-log10(p_fdr)", fontsize=9)
ax1.legend(fontsize=7, loc="lower right")
ax1.grid(axis="x", lw=0.4, color="lightgray")

# panel 3: residuals vs fitted (primary outcome)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("LME residuals vs fitted (fatigue_am)", fontsize=10)
ax2.scatter(m_primary.fittedvalues, m_primary.resid, alpha=0.1, s=2,
            color="steelblue", rasterized=True)
ax2.axhline(0, color="red", lw=1, ls="--")
ax2.set_xlabel("fitted", fontsize=9); ax2.set_ylabel("residuals", fontsize=9)
ax2.grid(lw=0.3, color="lightgray")

# panel 4: qq plot
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title("LME residuals Q-Q", fontsize=10)
from scipy.stats import probplot
(osm, osr), (slope, intercept_qq, _) = probplot(m_primary.resid)
ax3.scatter(osm, osr, alpha=0.15, s=2, color="steelblue", rasterized=True)
xl = np.array([min(osm), max(osm)])
ax3.plot(xl, slope * xl + intercept_qq, color="red", lw=1.2, ls="--")
ax3.set_xlabel("theoretical quantiles", fontsize=9)
ax3.set_ylabel("sample quantiles", fontsize=9)
ax3.grid(lw=0.3, color="lightgray")

# panel 5: mlvar network
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title("mlVAR bidirectional paths", fontsize=10)
ax4.set_xlim(0, 10); ax4.set_ylim(0, 8); ax4.axis("off")
node_pos = {
    "Fatigue\n(t-1)": (1.8, 6.2), "Phone\n(t-1)": (1.8, 1.8),
    "Fatigue\n(t)":   (8.2, 6.2), "Phone\n(t)":   (8.2, 1.8),
}
for label, (x, yn) in node_pos.items():
    c = "steelblue" if "Phone" in label else "seagreen"
    ax4.add_patch(plt.Circle((x, yn), 1.0, color=c, alpha=0.2))
    ax4.add_patch(plt.Circle((x, yn), 1.0, fill=False, edgecolor=c, lw=1.5))
    ax4.text(x, yn, label, ha="center", va="center", fontsize=8, fontweight="bold")


def draw_arrow(ax, src, tgt, beta, p, rad=0.0):
    x0, y0 = node_pos[src]; x1, y1 = node_pos[tgt]
    sig = p < 0.05
    col = "black" if sig else "lightgray"
    lw  = max(0.8, min(3.5, abs(beta) * 18))
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))
    mx, my = (x0+x1)/2, (y0+y1)/2
    ax.text(mx, my + rad*2, f"b={beta:+.3f}{'*' if sig else ' ns'}",
            ha="center", fontsize=7.5, color=col,
            bbox=dict(fc="white", ec="none", pad=1))


draw_arrow(ax4, "Fatigue\n(t-1)", "Fatigue\n(t)", fat_ar[0], fat_ar[1])
draw_arrow(ax4, "Phone\n(t-1)",   "Phone\n(t)",   pho_ar[0], pho_ar[1])
draw_arrow(ax4, "Phone\n(t-1)",   "Fatigue\n(t)", p2f[0], p2f[1], rad=0.2)
draw_arrow(ax4, "Fatigue\n(t-1)", "Phone\n(t)",   f2p[0], f2p[1], rad=-0.2)
ax4.text(5, 0.4, f"contemporaneous r={contemp_r:.3f} p={contemp_p:.3f}",
         ha="center", fontsize=7.5, color="gray")

# panel 6: lgbm r2 comparison
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_title("LightGBM R2 (person-level)", fontsize=10)
labels_bar = ["all features\n(circular)", "behavior only\n(honest)"]
vals_bar   = [r2_all, r2_behav]
ax5.barh(labels_bar, vals_bar, color=["lightgray", "steelblue"], height=0.4)
ax5.axvline(0, color="black", lw=0.8)
for j, v in enumerate(vals_bar):
    ax5.text(max(v + 0.01, 0.02), j, f"{v:.3f}", va="center", fontsize=9)
ax5.set_xlabel("R2 (5-fold CV)", fontsize=9)
ax5.set_xlim(-0.1, 1.05)
ax5.grid(axis="x", lw=0.4, color="lightgray")

# panel 7: sensitivity
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title("Sensitivity: communication coef (fatigue_am)", fontsize=10)
yp_s   = np.arange(len(sens_df))
colors = ["tomato" if p < 0.05 else "lightgray" for p in sens_df["comm_p"]]
ax6.barh(yp_s, sens_df["comm_b"], color=colors, height=0.5)
ax6.set_yticks(yp_s)
ax6.set_yticklabels(sens_df["config"], fontsize=8)
ax6.axvline(0, color="black", lw=0.8)
ax6.set_xlabel("communication_cwc beta", fontsize=9)
for j, (b, p) in enumerate(zip(sens_df["comm_b"], sens_df["comm_p"])):
    ax6.text(b + max(abs(sens_df["comm_b"])) * 0.05 + 0.001, j,
             f"p={p:.3f}", va="center", fontsize=7)
ax6.grid(axis="x", lw=0.3, color="lightgray")

# panel 8: key findings
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
ax7.set_title("Key findings", fontsize=10)

findings = []
primary  = summ[summ["outcome"] == "fatigue_am"].iloc[0]

inertia_b = coef[coef["predictor"] == "fatigue_am_lag1"]["beta"].values
if len(inertia_b):
    findings.append(f"fatigue inertia b={inertia_b[0]:+.2f}")

findings.append(
    f"GPS {'helps' if primary['lrt_gps_p_fdr'] < 0.05 else 'does NOT help'} "
    f"(p_fdr={primary['lrt_gps_p_fdr']:.3f})")
findings.append(
    f"digital {'helps' if primary['lrt_full_p_fdr'] < 0.05 else 'does NOT help'} "
    f"(p_fdr={primary['lrt_full_p_fdr']:.3f})")

# comm: was sig in main analysis, does it survive fdr across outcomes?
comm_row = all_coefs_df[
    (all_coefs_df["predictor"] == "minutes_communication_cwc") &
    (all_coefs_df["outcome"] == "fatigue_am")
]
if not comm_row.empty:
    r = comm_row.iloc[0]
    p_fdr = r.get("p_fdr", np.nan)
    if pd.notna(p_fdr) and p_fdr < 0.05:
        findings.append(f"comm -> fatigue holds FDR (p={p_fdr:.3f})")
    elif r["sig"]:
        findings.append(f"comm -> fatigue raw p={r['p']:.3f}, ns after FDR")

if p2f[1] >= 0.05 and f2p[1] >= 0.05:
    findings.append("no bidirectional loop (VAR ns)")
findings.append(f"behavior-only R2 = {r2_behav:.2f}")

for i, txt in enumerate(findings):
    ax7.text(0.02, 0.9 - i * 0.13, f"- {txt}",
             transform=ax7.transAxes, fontsize=8, va="top", wrap=True)

fig.savefig(OUT_DIR / "analysis_results.png", dpi=130, bbox_inches="tight")
plt.close()

print(f"\n  saved results/analysis_results.png")
print(f" csvs: lme_nested_results, lme_primary_coefs, lme_all_outcomes,")
print(f" sensitivity_results, logit_sensitivity, mlvar_results, lgbm_results")
print(f" also: results/lgbm_shap.png")