import os
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone
from add_cats import load_app_categorization, categorize_open_app

SLEEP_DIR  = "/Users/tanya/Desktop/code/vscode/jacobson project/sleep"
OUT_DIR = "/Users/tanya/Desktop/code/vscode/jacobson project/merges"
UNLOCK_DIR = "/Users/tanya/Desktop/code/vscode/jacobson project/unlock_clean"
RUNAPP_DIR = "/Users/tanya/Desktop/code/vscode/jacobson project/running_app_123"
    
os.makedirs(OUT_DIR, exist_ok=True)

UNLOCK_CACHE = {}
SESSION_CACHE = {}

VALIDATION_PRIORITY = {
    "ENHANCED_FINAL": 0, "AUTO_FINAL": 1, "ENHANCED_TENTATIVE": 2,
    "AUTO_TENTATIVE": 3, "MANUAL": 4, "AUTO_MANUAL": 5,
}


def convert_to_local_datetime(start_time_seconds, offset_seconds):
    utc_time = datetime.fromtimestamp(int(start_time_seconds), tz=timezone.utc)
    return (utc_time + timedelta(seconds=int(offset_seconds))).replace(tzinfo=None)


def make_time_window(sleep_start_local):
    rows = []
    t = sleep_start_local - timedelta(hours=1)
    while t <= sleep_start_local:
        rel = int((t - sleep_start_local).total_seconds() // 60)
        rows.append((rel, t))
        t += timedelta(minutes=2)
    return rows


def load_unlock_data(mlife_id):
    if mlife_id in UNLOCK_CACHE:
        return UNLOCK_CACHE[mlife_id]

    path = os.path.join(UNLOCK_DIR, f"{mlife_id}_unlock.csv")
    unlock_map = {}
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        if "date" in df.columns and "data" in df.columns:
            for _, row in df.iterrows():
                if pd.isna(row["date"]) or pd.isna(row["data"]):
                    continue
                try:
                    date_key = str(pd.to_datetime(row["date"]).date())
                except Exception:
                    continue
                s = str(row["data"]).strip().strip('"').strip("'").strip("[]")
                if not s:
                    continue
                vals = []
                for x in s.split(","):
                    x = x.strip()
                    if not x:
                        continue
                    try:
                        vals.append(int(float(x)))
                    except ValueError:
                        continue
                if vals:
                    unlock_map[date_key] = vals

    UNLOCK_CACHE[mlife_id] = unlock_map
    return unlock_map


def is_unlocked_at(unlock_map, ts):
    vals = unlock_map.get(str(ts.date()))
    if not vals:
        return False
    n = len(vals)
    secs = ts.hour * 3600 + ts.minute * 60 + ts.second
    idx = min(int(secs * n / 86400), n - 1)
    return vals[idx] == 1


def load_app_sessions(mlife_id):
    if mlife_id in SESSION_CACHE:
        return SESSION_CACHE[mlife_id]

    path = os.path.join(RUNAPP_DIR, f"{mlife_id}.csv")
    sessions = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 500000:
                    break
                if "[" not in line:
                    continue
                chunk = line[line.find("[") + 1: line.find("]")]
                parts = [p.strip().strip('"').strip("'") for p in chunk.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    start_dt = datetime.fromtimestamp(int(parts[1]) / 1000.0)
                except Exception:
                    continue
                dur = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                sessions.append((parts[0], start_dt, start_dt + timedelta(seconds=dur)))

    SESSION_CACHE[mlife_id] = sessions
    return sessions


def match_running_app_data(mlife_id, bins):
    if not bins:
        return [], []

    sessions = load_app_sessions(mlife_id)
    if not sessions:
        return ["NONE"] * len(bins), ["NONE"] * len(bins)

    unlock_map = load_unlock_data(mlife_id)
    use_unlock = bool(unlock_map)

    window_start = bins[0][1]
    window_end = bins[-1][1] + timedelta(minutes=2)
    in_window = [(a, s, e) for (a, s, e) in sessions if e > window_start and s < window_end]
    if not in_window:
        return ["NONE"] * len(bins), ["NONE"] * len(bins)

    new_apps_each_bin = []
    bg_apps_each_bin = []
    for _, ts in bins:
        bin_start = ts
        bin_end = ts + timedelta(minutes=2)
        new_apps = [a for (a, s, e) in in_window if bin_start <= s < bin_end]
        bg_apps = [a for (a, s, e) in in_window if s < bin_end and e > bin_start]
        new_apps_each_bin.append("NONE" if not new_apps else "|".join(sorted(set(new_apps))))
        bg_apps_each_bin.append("NONE" if not bg_apps else "|".join(sorted(set(bg_apps))))

    carried_fg = []
    last_app = "NONE"
    for val in new_apps_each_bin:
        if val not in ("NONE", ""):
            last_app = val
        carried_fg.append(last_app)

    final_fg = []
    final_bg = []
    for (rel_min, ts), fg, bg in zip(bins, carried_fg, bg_apps_each_bin):
        if use_unlock and not is_unlocked_at(unlock_map, ts):
            final_fg.append("NONE")
            final_bg.append("NONE")
        else:
            final_fg.append(fg)
            final_bg.append(bg)

    return final_fg, final_bg


def _ensure_cat_columns(df):
    for col in ["googleCats", "Schoedel", "cats", "reducedCats"]:
        if col not in df.columns:
            df[col] = "NONE"
    return df


def dedup_sleep_records(df):
    df = df.copy()
    df["_val_rank"] = df["validation"].map(VALIDATION_PRIORITY).fillna(99)
    df["durationInSeconds"] = pd.to_numeric(
        df.get("durationInSeconds", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    return (
        df.sort_values(["_val_rank", "durationInSeconds"], ascending=[True, False])
          .drop_duplicates(subset=["calendarDate", "mlife_id"], keep="first")
          .drop(columns=["_val_rank"])
    )


def process_one_sleep_file(path_to_csv):
    try:
        df = pd.read_csv(path_to_csv, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame([])

    df = dedup_sleep_records(df)

    all_rows = []
    for _, row in df.iterrows():
        calendar_date = row.get("calendarDate")
        mlife_id = row.get("mlife_id")
        start_sec = row.get("startTimeInSeconds")
        offset_sec = row.get("startTimeOffsetInSeconds")
        if pd.isna(calendar_date) or pd.isna(mlife_id) or pd.isna(start_sec) or pd.isna(offset_sec):
            continue

        sleep_start_local = convert_to_local_datetime(start_sec, offset_sec)
        timeline_rows = make_time_window(sleep_start_local)
        fg_list, bg_list = match_running_app_data(mlife_id, timeline_rows)

        for i, (rel_min, ts) in enumerate(timeline_rows):
            all_rows.append({
                "calendarDate": calendar_date,
                "mlife_id": mlife_id,
                "sleep_start_local": sleep_start_local.strftime("%Y-%m-%d %H:%M:%S"),
                "relative_time_minutes": rel_min,
                "time_point_local": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open_app": fg_list[i],
                "background_apps": bg_list[i],
            })

    return pd.DataFrame(all_rows)


def main():
    google_map, schoedel_map, cats_map, reduced_map = load_app_categorization()
    for csv_path in glob.glob(os.path.join(SLEEP_DIR, "*.csv")):
        df = process_one_sleep_file(csv_path)
        try:
            df = categorize_open_app(df, google_map, schoedel_map, cats_map, reduced_map)
        except Exception:
            pass
        df = _ensure_cat_columns(df)
        df.to_csv(os.path.join(OUT_DIR, os.path.basename(csv_path)), index=False)


if __name__ == "__main__":
    main()