"""
python build_dataset.py #default                       
python build_dataset.py nightly_summary dataset #my paths

rules:
  1. min nights
  2. EMA coverage >= threshold
  3. GPS coverage >= threshold
  4. no big consecutive GPS gap
  5. phone coverage >= threshold
  6. trailing crop-> trim end once GPS+EMA both missing too long
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# thresholds
MIN_NIGHTS= 14
EMA_MIN_PCT= 0.30
GPS_MIN_PCT= 0.50
GPS_GAP_MAX= 14
PHONE_MIN_PCT= 0.20
TRAIL_GAP_MAX= 14
WINDOW_HOURS= 1 #HAS TO MATCH pipeline.py!!!

VALIDATION_RANK = {
    "ENHANCED_FINAL": 0, "AUTO_FINAL": 1,
    "ENHANCED_TENTATIVE": 2, "AUTO_TENTATIVE": 3,
    "MANUAL": 4, "AUTO_MANUAL": 5,
}
APP_CATS = ["social_media", "communication", "other", "system", "unknown"]
EMA_PREFIXES = ["fatigue","anhedonia","depression","anxiety","worry","somatic","negative_affect"]


def max_run_false(series):
    # max consecutive false
    max_r = cur = 0
    for v in series:
        cur = cur + 1 if not v else 0
        max_r = max(max_r, cur)
    return max_r


def crop_trailing(df):
# trim nights from end once GPS+EMA both missing for TRAIL_GAP_MAX consecutive nights
    df = df.sort_values("calendarDate").reset_index(drop=True)
    ema_cols = [c for c in df.columns if any(c.startswith(p) for p in EMA_PREFIXES)]
    both_missing = (
        (df["gps_point_count"] == 0) &
        (df[ema_cols].isna().all(axis=1) if ema_cols else True)
    )
    good = df.index[~both_missing]
    if len(good) == 0:
        return df.iloc[0:0]
    last_good = good.max()
    if both_missing.iloc[last_good+1:].sum()> TRAIL_GAP_MAX:
        df = df.iloc[:last_good+1]
    return df


def qc_one(df):
    pid = df["mlife_id"].iloc[0]
    report= {"pid": pid, "nights_raw": len(df), "exclusion": None}

    df = df.copy()
    df["calendarDate"]= pd.to_datetime(df["calendarDate"])
    df = df.sort_values("calendarDate").reset_index(drop=True)

# rule 6 first 
    df = crop_trailing(df)
    report["nights_after_crop"] = len(df)
    if len(df) == 0:
        report["exclusion"] = "all_data_trailing"
        return df, report

    has_gps = df["gps_point_count"] > 0
    ema_cols = [c for c in df.columns if any(c.startswith(p) for p in EMA_PREFIXES)]
    has_ema = df[ema_cols].notna().any(axis=1) if ema_cols else pd.Series(False, index=df.index)
    has_phone = df["minutes_on_phone_before_bed"] > 0

    ema_pct = has_ema.mean()
    gps_pct = has_gps.mean()
    phone_pct = has_phone.mean()
    gps_gap = max_run_false(has_gps)

    if ema_pct < EMA_MIN_PCT: report["exclusion"] = f"low_ema_{ema_pct:.0%}"; return df.iloc[0:0], report
    if gps_pct < GPS_MIN_PCT: report["exclusion"] = f"low_gps_{gps_pct:.0%}"; return df.iloc[0:0], report
    if gps_gap > GPS_GAP_MAX: report["exclusion"] = f"gps_gap_{gps_gap}d"; return df.iloc[0:0], report
    if phone_pct < PHONE_MIN_PCT: report["exclusion"] = f"low_phone_{phone_pct:.0%}"; return df.iloc[0:0], report
    if len(df) < MIN_NIGHTS: report["exclusion"] = f"too_few_{len(df)}"; return df.iloc[0:0], report

    report.update({"nights_final": len(df), "ema_pct": round(ema_pct, 3),
                   "gps_pct": round(gps_pct, 3), "phone_pct": round(phone_pct, 3)})
    return df, report


def engineer_features(df):
    df = df.copy()

 # mean for every ema item
    ema_items = EMA_PREFIXES
    for item in ema_items:
        am_c, pm_c = f"{item}_am", f"{item}_afternoon"
        cols = [c for c in [am_c, pm_c] if c in df.columns]
        if cols:
            df[f"{item}_mean"] = df[cols].mean(axis=1)

 # phone binary + cat proportions
    df["any_phone"] = (df["minutes_on_phone_before_bed"] > 0).astype(int)
    total_phone = df["minutes_on_phone_before_bed"].replace(0, np.nan)
    for cat in APP_CATS:
        col = f"minutes_{cat}"
        if col in df.columns:
            df[f"{cat}_pct"] = df[col] / total_phone
    pct_cols = [f"{c}_pct" for c in APP_CATS if f"minutes_{c}" in df.columns]
    df[pct_cols] = df[pct_cols].fillna(0)

    # sleep val rank
    df["sleep_validation_rank"] = df["sleep_validation"].map(VALIDATION_RANK).fillna(99).astype(int)

    # time feat
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])
    df["days_in_study"] = (df["calendarDate"] - df.groupby("mlife_id")["calendarDate"].transform("min")).dt.days
    df["day_of_week"] = df["calendarDate"].dt.dayofweek
    return df


def verify(dataset):
    print("verification")
    issues = []

    def check(label, ok, detail=""):
        icon = "ok" if ok else "not ok"
        print(f"  [{icon}] {label}" + (f"  —  {detail}" if detail else ""))
        if not ok: issues.append(label)

    print(f"participants   : {dataset['mlife_id'].nunique()}")
    print(f"nights : {len(dataset)}")

    dupes = (dataset.groupby(["mlife_id","calendarDate"]).size() > 1).sum()
    check("no duplicate nights", dupes == 0, f"{dupes} found" if dupes else "none")

    #check rules
    rule_fails = []
    for pid, grp in dataset.groupby("mlife_id"):
        if len(grp) < MIN_NIGHTS:
            rule_fails.append(f"{pid}: {len(grp)} nights")
        ema_cols = [c for c in grp.columns if any(c.startswith(p) for p in EMA_PREFIXES)]
        if (grp[ema_cols].notna().any(axis=1).mean() if ema_cols else 0) < EMA_MIN_PCT:
            rule_fails.append(f"{pid}: EMA low")
        if (grp["gps_point_count"] > 0).mean() < GPS_MIN_PCT:
            rule_fails.append(f"{pid}: GPS low")
        if (grp["minutes_on_phone_before_bed"] > 0).mean() < PHONE_MIN_PCT:
            rule_fails.append(f"{pid}: phone low")
    check("all participants pass qc", len(rule_fails) == 0,
          f"{len(rule_fails)} fail(s)" if rule_fails else "all pass")
    for f in rule_fails[:5]: print(f"    - {f}")
    issues.extend(rule_fails)

    # value ranges
    print(f"\n  value ranges:")
    range_checks = [
        ("minutes_on_phone_before_bed", 0, WINDOW_HOURS * 60),
        ("day_of_week", 0, 6),
        ("sleep_validation_rank", 0, 99),
    ]
    # add all ema mean cols
    ema_mean_cols = [c for c in dataset.columns if c.endswith("_mean") and
                     any(c.startswith(p) for p in EMA_PREFIXES)]
    range_checks += [(c, 0, 100) for c in ema_mean_cols]

    for col, lo, hi in range_checks:
        if col not in dataset.columns: continue
        vals = dataset[col].dropna()
        bad = ((vals < lo) | (vals > hi)).sum()
        rng = f"{vals.min():.1f} - {vals.max():.1f}"
        check(f"  {col}", bad == 0, rng + (f"  ({bad} out-of-range)" if bad else ""))
        if bad: issues.append(f"{col}: {bad} out-of-range")

    # missingness
    print(f"\n missingness:")
    check_cols = ema_mean_cols + ["sleep_base_lat","minutes_on_phone_before_bed"]
    for col in check_cols:
        if col in dataset.columns:
            miss = dataset[col].isna().mean()
            print(f"    {col:40s}: {miss:.1%}")

    npp = dataset.groupby("mlife_id").size()
    print(f"\n  nights/participant: min={npp.min()}  median={npp.median():.0f}  mean={npp.mean():.1f}  max={npp.max()}")
    print(f"\n  {'dataset looks clean.' if not issues else f'WARNING: {len(issues)} issue(s) above'}")
    return len(issues)


def main(summary_dir, out_dir):
    summary_dir = Path(summary_dir)
    out_dir     = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(summary_dir.glob("*_summary.csv"))
    print(f"\n  {len(files)} summary files")
    print(f"  qc: min_nights={MIN_NIGHTS}  EMA>={EMA_MIN_PCT:.0%}  GPS>={GPS_MIN_PCT:.0%}  "
          f"gps_gap<={GPS_GAP_MAX}d  phone>={PHONE_MIN_PCT:.0%}  trail_crop>{TRAIL_GAP_MAX}d\n")

    kept, excluded, reports = [], [], []
    excl_reasons = {}

    for i, f in enumerate(files, 1):
        pid = f.stem.replace("_summary", "")
        try:
            raw = pd.read_csv(f, low_memory=False)
            if raw.empty:
                reports.append({"pid": pid, "exclusion": "empty_file"})
                excluded.append(pid); continue

            filtered, report = qc_one(raw)
            reports.append(report)

            n_raw = report["nights_raw"]
            n_crop = report.get("nights_after_crop", n_raw)
            n_final = report.get("nights_final", 0)
            excl = report["exclusion"]
            crop_str = f" [-{n_raw-n_crop} trailing]" if n_raw != n_crop else ""

            if excl:
                excluded.append(pid)
                key = excl.split("_")[0]
                excl_reasons[key] = excl_reasons.get(key, 0) + 1
                print(f"[{i:>3}/{len(files)}] {pid:15s} {n_raw:>3}{crop_str}  ->  excluded ({excl})")
            else:
                kept.append(filtered)
                print(f"  [{i:>3}/{len(files)}] {pid:15s} {n_raw:>3}{crop_str}  ->  {n_final} nights kept")

        except Exception as e:
            reports.append({"pid": pid, "exclusion": f"error: {e}"})
            excluded.append(pid)
            print(f"[{i:>3}/{len(files)}] {pid:15s}  ->  ERROR: {e}")

    print(f"\n Kept: {len(kept)}  |  Excluded: {len(excluded)}  |  Reasons: {excl_reasons}")

    if not kept:
        print("\n no participants passed qc — check thresholds"); return

    dataset = pd.concat(kept, ignore_index=True)
    dataset = engineer_features(dataset)
    verify(dataset)

    dataset.to_csv(out_dir / "dataset.csv", index=False)
    pd.DataFrame(reports).to_csv(out_dir / "qc_report.csv", index=False)
    print(f"\n saved: {out_dir}/dataset.csv")
    print(f" saved: {out_dir}/qc_report.csv\n")


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else "nightly_summary"
    od = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    main(sd, od)