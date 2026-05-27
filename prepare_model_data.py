"""
prepare_model_data.py
takes dataset/dataset.csv -> two model-ready files

dataset/lme_data.csv for linear mixed effects model
dataset/ml_data.csv for lightgbm / tree models

    python prepare_model_data.py # default
    python prepare_model_data.py dataset dataset  # my path
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# config
MIN_NIGHTS_FINAL = 30
GAP_SPLIT_DAYS = 60
WINDOW_HOURS = 1   # keep in sync with pipeline.py

TARGET = "fatigue_am"  # morning fatigue only 
EMA_PREFIXES = ["fatigue","anhedonia","depression","anxiety","worry","somatic","negative_affect"]

CONTINUOUS_FEATS = [
    "minutes_on_phone_before_bed",
    "minutes_social_media", "minutes_communication",
    "minutes_other", "minutes_system",
    "social_media_pct", "communication_pct",
    "total_distance_m",  # gps_point_count dropped, measurement count, not behavioral
    "n_locations", "minutes_at_home", "max_dist_from_home_m",  # behavioral gps 
]

VALIDATION_RANK = {
    "ENHANCED_FINAL": 0, "AUTO_FINAL": 1,
    "ENHANCED_TENTATIVE": 2, "AUTO_TENTATIVE": 3,
    "MANUAL": 4, "AUTO_MANUAL": 5,
}


def drop_bad_rows(df):
    n = len(df)
    df = df[df["sleep_validation"] != "OFF_WRIST"].copy()
    if len(df) < n:
        print(f"  dropped {n-len(df)} OFF_WRIST rows")
    return df


def split_at_gaps(df):
    # if a participant has a gap > GAP_SPLIT_DAYS, keep biggest seg
    df = df.copy()
    df["_seg"] = 0
    splits = 0
    for pid, grp in df.groupby("mlife_id"):
        grp = grp.sort_values("calendarDate")
        diffs = grp["calendarDate"].diff().dt.days.fillna(0)
        gap_idx = diffs[diffs > GAP_SPLIT_DAYS].index.tolist()
        if not gap_idx: continue
        seg = 0
        for idx in grp.index:
            if idx in gap_idx: seg += 1
            df.loc[idx, "_seg"] = seg
        seg_sizes = df[df["mlife_id"] == pid].groupby("_seg").size()
        keep      = seg_sizes.idxmax()
        drop_mask = (df["mlife_id"] == pid) & (df["_seg"] != keep)
        n_dropped = drop_mask.sum()
        if n_dropped:
            df = df[~drop_mask]
            splits += 1
            print(f"  {pid}: split at {GAP_SPLIT_DAYS}d gap, kept segment {keep} "
                  f"({seg_sizes[keep]} nights), dropped {n_dropped}")
    df = df.drop(columns=["_seg"])
    print(f"total participants split: {splits}")
    return df


def enforce_min_nights(df, min_n):
    counts  = df.groupby("mlife_id").size()
    too_few = counts[counts < min_n].index.tolist()
    if too_few:
        print(f" dropping {len(too_few)} participants with < {min_n} nights: {too_few}")
        df = df[~df["mlife_id"].isin(too_few)]
    return df


def impute_gps_nans(df):
    # nights with no gps distance=0, moving=0 
    gps_zero = df["gps_point_count"] == 0
    for col in ["total_distance_m", "sleep_base_is_moving"]:
        n = df.loc[gps_zero, col].isna().sum()
        if n:
            df.loc[gps_zero, col] = df.loc[gps_zero, col].fillna(0)
            print(f"  imputed {n} NaN → 0 in {col}")
    return df


def center_within_person(df):
    df = df.copy()
    # center all cont feat + ema item means
    ema_mean_cols = [f"{p}_mean" for p in EMA_PREFIXES if f"{p}_mean" in df.columns]
    all_feats = CONTINUOUS_FEATS + ema_mean_cols
    for feat in all_feats:
        if feat not in df.columns: continue
        pm = df.groupby("mlife_id")[feat].transform("mean")
        df[f"{feat}_pm"]  = pm
        df[f"{feat}_cwc"] = df[feat] - pm
    print(f"  added _pm and _cwc for {len([f for f in all_feats if f in df.columns])} features")
    return df


def recompute_centering(df):
    # recenter after dropping rows
    df = df.copy()
    ema_mean_cols = [f"{p}_mean" for p in EMA_PREFIXES if f"{p}_mean" in df.columns]
    for feat in CONTINUOUS_FEATS + ema_mean_cols:
        if feat not in df.columns: continue
        pm = df.groupby("mlife_id")[feat].transform("mean")
        df[f"{feat}_pm"]  = pm
        df[f"{feat}_cwc"] = df[feat] - pm
    return df


def add_lag_features(df):
    df = df.sort_values(["mlife_id", "calendarDate"]).copy()
    ema_mean_cols = [f"{p}_mean" for p in EMA_PREFIXES if f"{p}_mean" in df.columns]
    lag_cols = ema_mean_cols + ["fatigue_am", "minutes_on_phone_before_bed", "any_phone"]

    for col in lag_cols:
        if col not in df.columns: continue
        lagged    = df.groupby("mlife_id")[col].shift(1)
        prev_date = df.groupby("mlife_id")["calendarDate"].shift(1)
        gap_days  = (df["calendarDate"] - prev_date).dt.days
        lagged[gap_days > 2] = np.nan   # gap > 2 days = don't use lag
        df[f"{col}_lag1"] = lagged

    print(f"  added lag-1 for: {lag_cols}")
    return df


def recheck_min(df, min_n):
    counts= df.groupby("mlife_id").size()
    too_few = counts[counts < min_n].index.tolist()
    if too_few:
        print(f"re-enforcing {min_n} nights: dropping {len(too_few)} participants "
              f"who fell below after target filtering: {too_few}")
        df = df[~df["mlife_id"].isin(too_few)]
    return df


def make_lme_data(df):
    lme = df.dropna(subset=[TARGET]).copy()
    lme = recheck_min(lme, MIN_NIGHTS_FINAL)
    lme = recompute_centering(lme)

    id_cols = ["mlife_id", "calendarDate"]
    target_cols = [c for c in lme.columns if any(c.startswith(p) for p in
                    EMA_PREFIXES + ["fatigue_am","fatigue_afternoon","anhedonia_am",
                                    "depression_am","anxiety_am","worry_am","somatic_am",
                                    "negative_affect_am"])]
    cwc_cols = [c for c in lme.columns if c.endswith("_cwc")]
    pm_cols = [c for c in lme.columns if c.endswith("_pm")]
    lag_cols = [c for c in lme.columns if c.endswith("_lag1")]
    # gps_point_count not included — it's measurement frequency, not behavior
    context_cols = [c for c in ["day_of_week","days_in_study","sleep_validation_rank",
                                 "sleep_base_is_moving","any_phone",
                                 "unlock_available"] if c in lme.columns]

    keep = id_cols + list(dict.fromkeys(target_cols + cwc_cols + pm_cols + lag_cols + context_cols))
    lme  = lme[[c for c in keep if c in lme.columns]]
    print(f"  LME: {len(lme)} rows, {len(lme.columns)} cols ({lme['mlife_id'].nunique()} participants)")
    return lme


def make_ml_data(df):
    ml = df.dropna(subset=[TARGET]).copy()
    ml = recheck_min(ml, MIN_NIGHTS_FINAL)
    ml = recompute_centering(ml)

    drop_cols = [
        "fatigue_am", "fatigue_afternoon", # components of target — leaks
        "sleep_start_local",
        "sleep_base_lat", "sleep_base_lon", # raw coords not useful
        "sleep_validation",
        "has_app_data", # pipeline meta col
    ]
    # also drop raw am/pm ema cols 
    for p in EMA_PREFIXES:
        drop_cols += [f"{p}_am", f"{p}_afternoon"]

    ml = ml.drop(columns=[c for c in drop_cols if c in ml.columns])
    print(f"  ML:  {len(ml)} rows, {len(ml.columns)} cols ({ml['mlife_id'].nunique()} participants)")
    return ml


def verify(lme, ml):
    issues = []

    def check(label, ok, detail=""):
        icon = "ok" if ok else "not ok"
        print(f"  [{icon}] {label}" + (f"  {detail}" if detail else ""))
        if not ok: issues.append(label)

    check("lme rows == ml rows", len(lme) == len(ml), f"lme={len(lme)}  ml={len(ml)}")

    leak_cols = {"fatigue_am","fatigue_afternoon"} & set(ml.columns)
    check("no target components in ml", len(leak_cols) == 0, f"found: {leak_cols}" if leak_cols else "clean")

    cwc_nans = lme[[c for c in lme.columns if "_cwc" in c]].isna().sum().sum()
    check("no NaN in _cwc cols", cwc_nans == 0, f"{cwc_nans} NaN" if cwc_nans else "clean")

    t = lme[TARGET]
    check(f"target in 0-100", t.between(0,100).all(), f"min={t.min():.0f}  max={t.max():.0f}")

    dupes = (lme.groupby(["mlife_id","calendarDate"]).size() > 1).sum()
    check("no duplicate nights", dupes == 0, f"{dupes} found" if dupes else "none")

    npp = ml.groupby("mlife_id").size()
    check(f"all participants >= {MIN_NIGHTS_FINAL} nights",
          (npp >= MIN_NIGHTS_FINAL).all(), f"min={npp.min()}  violators={(npp<MIN_NIGHTS_FINAL).sum()}")

    max_dev = lme.groupby("mlife_id")[[c for c in lme.columns if "_cwc" in c]].mean().abs().max(axis=1).max()
    check("_cwc person means ≈ 0", max_dev < 0.001, f"max deviation: {max_dev:.8f}")

    print(f"\n  missingness:")
    for col in [TARGET] + [f"{p}_mean_lag1" for p in EMA_PREFIXES if f"{p}_mean_lag1" in ml.columns] + \
               ["total_distance_m","sleep_base_is_moving","minutes_on_phone_before_bed"]:
        if col in ml.columns:
            print(f"    {col:45s}: {ml[col].isna().mean():.1%}")

    npp_ml = ml.groupby("mlife_id").size()
    print(f"\n nights/participant: min={npp_ml.min()}  median={npp_ml.median():.0f}  "
          f"mean={npp_ml.mean():.1f}  max={npp_ml.max()}")
    print(f"\n  {'all checks passed.' if not issues else f'WARNING: {len(issues)} issue(s) above'}")


def main(in_dir, out_dir):
    in_dir  = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  loading {in_dir /'dataset.csv'} ...")
    df = pd.read_csv(in_dir / "dataset.csv", low_memory=False)
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])
    print(f"  loaded: {len(df)} rows, {df['mlife_id'].nunique()} participants\n")

    print("step 1 cleanup")
    df = drop_bad_rows(df)
    df = split_at_gaps(df)
    df = enforce_min_nights(df, MIN_NIGHTS_FINAL)
    print(f"  after cleanup: {len(df)} rows, {df['mlife_id'].nunique()} participants\n")

    print("step 2 gps nan imputation")
    df = impute_gps_nans(df)
    print()

    print("step 3 within person centering")
    df = center_within_person(df)
    print()

    print("step 4 lag 1 features")
    df = add_lag_features(df)
    print()

    print("step 5 building lme and ml datasets")
    lme = make_lme_data(df)
    ml  = make_ml_data(df)
    print()

    verify(lme, ml)

    lme.to_csv(out_dir/ "lme_data.csv", index=False)
    ml.to_csv(out_dir/ "ml_data.csv",  index=False)
    print(f"\n {out_dir}/lme_data.csv  ({len(lme)} rows, {len(lme.columns)} cols)")
    print(f" {out_dir}/ml_data.csv  ({len(ml)} rows, {len(ml.columns)} cols)\n")

    print(f"lme columns: {list(lme.columns)}")
    print(f"ml columns: {list(ml.columns)}\n")


if __name__ == "__main__":
    ind= sys.argv[1] if len(sys.argv) > 1 else "dataset"
    od = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    main(ind, od)