"""
python pipeline.py -> runs all
python pipeline.py 1 3 -> runs those specific stages
python pipeline.py diagnose ->inspect without rerunning

0  unlock/ + running_app_123/->unlock_clean/
1  sleep/ + unlock_clean/->merges/
2  GPS raw + merges/->parsed_gps/
3  merges/+ parsed_gps/ + EMA ->nightly_summary/
"""

import ast, json, shutil, sys
from datetime import datetime, timedelta, timezone
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import numpy as np
import pandas as pd


# paths 
BASE_DIR = Path(__file__).parent.resolve()
UNLOCK_RAW_DIR = BASE_DIR / "unlock"
UNLOCK_CLEAN_DIR = BASE_DIR / "unlock_clean"
RUNAPP_DIR = BASE_DIR / "running_app_123"
RUNAPP_DIR_ALT = BASE_DIR / "running_app_122"
SLEEP_DIR = BASE_DIR / "sleep"
MERGES_DIR = BASE_DIR / "merges"
GPS_RAW_DIR = BASE_DIR / "GPS Processing/gps_all_240415/raw_gps"
GPS_PARSED_DIR = GPS_RAW_DIR / "parsed_gps"
SUMMARY_DIR = BASE_DIR / "nightly_summary"
EMA_PATH = BASE_DIR / "3._TD_cohort_EMA_FINAL_DATA/3._TD_cohort_EMA_FINAL.csv"

WINDOW_HOURS = 1  # presleep window hours
BIN_MINUTES = 2
MAX_APP_LINES = 500_000 #can adjust later but this num is fine

#garmin 
VALIDATION_PRIORITY = {
    "ENHANCED_FINAL": 0, "AUTO_FINAL": 1,
    "ENHANCED_TENTATIVE": 2, "AUTO_TENTATIVE": 3,
    "MANUAL": 4, "AUTO_MANUAL": 5,
}
APP_CATS = ["Social Media", "Communication", "Other", "System", "UNKNOWN"]

# all ema needed
EMA_ITEMS = {
    "PHQ-4": "fatigue", # tired/little energy
    "PHQ-1": "anhedonia", # little interest or pleasure
    "PHQ-2": "depression", # feeling down/hopeless
    "GAD-1": "anxiety",  # nervous/anxious/on edge
    "GAD-2": "worry", # can't stop worrying
    "Somatic": "somatic",# somatic symptoms
}



def progress(i, total, label=""):
    print(f"  [{i:>3}/{total}] {label}", flush=True)

def dedup_nights(df):
    # one garmin record per night  
    # pick best validation quality then longest duration
    df = df.copy()
    df["_rank"] = df["validation"].map(VALIDATION_PRIORITY).fillna(99)
    df["durationInSeconds"] = pd.to_numeric(
        df.get("durationInSeconds", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    out = (
        df.sort_values(["_rank", "durationInSeconds"], ascending=[True, False])
          .drop_duplicates(subset=["calendarDate", "mlife_id"], keep="first")
          .drop(columns=["_rank"])
    )
    removed = len(df) - len(out)
    if removed:
        print(f"dedup removed {removed} duplicate garmin records")
    return out

def to_local(start_sec, offset_sec):
    utc = datetime.fromtimestamp(int(start_sec), tz=timezone.utc)
    return (utc + timedelta(seconds=int(offset_sec))).replace(tzinfo=None)
#distance form
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    a = sin(radians(lat2-lat1)/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# 0 clean unlock data

def _parse_data_field(val):
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) == 3:
                return parsed
        except (ValueError, SyntaxError):
            pass
    return val

def _align_unlock_to_app(unlock_seq, app_df):
# zeroing out unlock seconds where no app running
# default raw if no valid 
    has_valid = any(isinstance(r["data"], list) and len(r["data"]) == 3
                    for _, r in app_df.iterrows())
    if not has_valid:
        return unlock_seq
    indicator = np.zeros_like(unlock_seq)
    app_sorted = app_df.sort_values("unlock_index").reset_index(drop=True)
    for i, row in app_sorted.iterrows():
        start = int(row["unlock_index"])
        try:
            dur = round(int(row["data"][2]) / 1000)
            end = min(start + dur, len(unlock_seq))
            indicator[start:end] = 1
        except (ValueError, TypeError, IndexError):
            end = int(app_sorted.iloc[i+1]["unlock_index"]) if i+1 < len(app_sorted) else len(unlock_seq)
            indicator[start:end] = unlock_seq[start:end]
    return unlock_seq * indicator

def _clean_one_unlock(uid):
    raw_path = UNLOCK_RAW_DIR / f"{uid}_unlock.csv"
    app_path = RUNAPP_DIR / f"{uid}.csv"
    if not app_path.exists():
        app_path = RUNAPP_DIR_ALT / f"{uid}.csv"
    out_path = UNLOCK_CLEAN_DIR / f"{uid}_unlock.csv"

    if not raw_path.exists(): return "skipped"
    if not app_path.exists():
        shutil.copy(raw_path, out_path)
        return "copied"

    try:
        unlock_df = pd.read_csv(raw_path)
        unlock_df["date"] = pd.to_datetime(unlock_df["date"], errors="coerce")
        unlock_df = unlock_df[unlock_df["date"].dt.year.between(2000, 2100)].copy()
        unlock_df["date"] = unlock_df["date"].dt.normalize()
        unlock_df = unlock_df.sort_values("date").reset_index(drop=True)
        if unlock_df.empty: return "skipped"

        app_df = pd.read_csv(app_path, usecols=lambda c: c != "Unnamed: 0")
        app_df["day"] = pd.to_datetime(app_df["day"], errors="coerce")
        app_df = app_df[app_df["day"].dt.year.between(2000, 2100)].copy()
        app_df["seconds"] = (app_df["day"] - app_df["day"].dt.normalize()).dt.total_seconds().round().astype(int)
        app_df["data"] = app_df["data"].apply(_parse_data_field)

        #one big vector+ track where each day starts
        seqs, starts, day_lens, date_to_start = [], [], [], {}
        cur = 0
        for i, row in unlock_df.iterrows():
            seq = np.array(json.loads(row["data"]))
            seqs.append(seq); starts.append(cur); day_lens.append(len(seq))
            date_to_start[row["date"].date()] = cur
            cur+= len(seq)
        full_seq = np.concatenate(seqs)

        app_df["date_key"] = app_df["day"].dt.date
        app_df["start_index"]= app_df["date_key"].map(date_to_start)
        app_df =app_df.dropna(subset=["start_index"]).copy()
        app_df["unlock_index"] = app_df["start_index"].astype(int) + app_df["seconds"]

        cleaned = _align_unlock_to_app(full_seq, app_df)

        rows = []
        for i, (d_len, start_i) in enumerate(zip(day_lens, starts)):
            day_clean = cleaned[start_i: start_i + d_len]
            raw_day= np.array(json.loads(unlock_df.iloc[i]["data"]))
            # if cleaning zeroed everything use raw
            if day_clean.sum() == 0 and raw_day.sum() > 0:
                day_clean = raw_day
            rows.append({"date": unlock_df.iloc[i]["date"].date(), "data": json.dumps(day_clean.tolist())})

        out_df = pd.DataFrame(rows)
        out_df["uid"] = uid
        out_df.to_csv(out_path, index=False)
        return "cleaned"
    except Exception as e:
        print(f"err: {e}")
        return "error"

def stage0_clean_unlock():
    print("stage 0: clean unlock data")
    UNLOCK_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    uids = sorted(f.name.split("_")[0] for f in UNLOCK_RAW_DIR.iterdir()
                  if f.name.endswith("_unlock.csv"))
    print(f"{len(uids)} participants\n")
    counts= {"cleaned": 0, "copied": 0, "skipped": 0, "error": 0}
    for i, uid in enumerate(uids, 1):
        result= _clean_one_unlock(uid)
        counts[result] += 1
        progress(i, len(uids), f"{uid:20s} → {result}")
    print(f"\n  done: {counts}")


# stage 1 pre sleep timelines

_UNLOCK_CACHE  = {}
_SESSION_CACHE = {}

def _load_unlock(mlife_id):
    if mlife_id in _UNLOCK_CACHE:
        return _UNLOCK_CACHE[mlife_id]
    path = UNLOCK_CLEAN_DIR / f"{mlife_id}_unlock.csv"
    if not path.exists():
        _UNLOCK_CACHE[mlife_id] = {}
        return {}
    df = pd.read_csv(path, dtype=str)
    unlock_map = {}
    for _, row in df.iterrows():
        try:
            date_key = str(pd.to_datetime(row["date"]).date())
        except Exception:
            continue
        s = str(row["data"]).strip().strip("\"'[]")
        vals = [int(float(x)) for x in s.split(",") if x.strip()]
        if vals and sum(vals) > 0: #skip all zero days
            unlock_map[date_key] = vals
    _UNLOCK_CACHE[mlife_id] = unlock_map
    return unlock_map

def _is_unlocked(unlock_map, ts):
    vals = unlock_map.get(str(ts.date()))
    if vals is None: return True # no data = don't filter
    n = len(vals); secs = ts.hour*3600 + ts.minute*60 + ts.second
    return vals[min(int(secs*n/86400), n-1)] == 1

def _load_sessions(mlife_id):
    if mlife_id in _SESSION_CACHE:
        return _SESSION_CACHE[mlife_id]
    path = RUNAPP_DIR / f"{mlife_id}.csv"
    if not path.exists():
        path = RUNAPP_DIR_ALT / f"{mlife_id}.csv"
    if not path.exists():
        _SESSION_CACHE[mlife_id] = []
        return []
    sessions = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > MAX_APP_LINES: break
            if "[" not in line: continue
            chunk = line[line.find("[")+1: line.find("]")]
            parts = [p.strip().strip("\"'") for p in chunk.split(",")]
            if len(parts) < 2: continue
            try:
                start = datetime.fromtimestamp(int(parts[1]) / 1000.0)
            except Exception:
                continue
            dur = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            sessions.append((parts[0], start, start + timedelta(seconds=dur)))
    _SESSION_CACHE[mlife_id] = sessions
    return sessions

def _make_bins(sleep_start):
    bins, t = [], sleep_start - timedelta(hours=WINDOW_HOURS)
    while t <= sleep_start:
        bins.append((int((t - sleep_start).total_seconds() // 60), t))
        t += timedelta(minutes=BIN_MINUTES)
    return bins

def _match_apps(mlife_id, bins):
    sessions   = _load_sessions(mlife_id)
    unlock_map = _load_unlock(mlife_id)
    use_unlock = bool(unlock_map)

    if not sessions or not bins:
        return ["NONE"]*len(bins), ["NONE"]*len(bins)

    window_start = bins[0][1]
    window_end = bins[-1][1] + timedelta(minutes=BIN_MINUTES)
    in_window = [(a,s,e) for a,s,e in sessions if e > window_start and s < window_end]
    if not in_window:
        return ["NONE"]*len(bins), ["NONE"]*len(bins)

    new_each, bg_each = [], []
    for i, ts in bins:
        bin_end = ts + timedelta(minutes=BIN_MINUTES)
        new_each.append("|".join({a for a,s,e in in_window if ts <= s < bin_end}) or "NONE")
        bg_each.append( "|".join({a for a,s,e in in_window if s < bin_end and e > ts}) or "NONE")

    #carry forward with anything already open before the window
    already_open = [a for a,s,e in in_window if s < window_start]
    last = already_open[-1] if already_open else "NONE"
    carried = []
    for val in new_each:
        if val != "NONE": last = val
        carried.append(last)

    fg, bg = [], []
    for (_, ts), fv, bv in zip(bins, carried, bg_each):
        if use_unlock and not _is_unlocked(unlock_map, ts):
            fg.append("NONE"); bg.append("NONE")
        else:
            fg.append(fv); bg.append(bv)
    return fg, bg

def _process_sleep_file(path, cat_maps):
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df = dedup_nights(df)
    rows = []
    for _, row in df.iterrows():
        cal, uid, sec, off = (row.get(c) for c in
            ["calendarDate","mlife_id","startTimeInSeconds","startTimeOffsetInSeconds"])
        if any(pd.isna(v) for v in [cal, uid, sec, off]): continue
        sleep_start = to_local(sec, off)
        bins = _make_bins(sleep_start)
        fg, bg = _match_apps(uid, bins)
        for i, (rel_min, ts) in enumerate(bins):
            rows.append({
                "calendarDate": cal, "mlife_id": uid,
                "sleep_start_local": sleep_start.strftime("%Y-%m-%d %H:%M:%S"),
                "validation": row.get("validation", ""),
                "relative_time_minutes": rel_min,
                "time_point_local": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open_app": fg[i], "background_apps": bg[i],
            })
    result = pd.DataFrame(rows)
    if not result.empty and cat_maps:
        try:
            from add_cats import categorize_open_app
            result = categorize_open_app(result, *cat_maps)
        except Exception as e:
            print(f"      [warn] categorize_open_app: {e}")
    for col in ["googleCats","Schoedel","cats","reducedCats"]:
        if col not in result.columns: result[col] = "NONE"
    return result

def stage1_build_timelines():
    print("stage 1 — build pre-sleep timelines")
    MERGES_DIR.mkdir(parents=True, exist_ok=True)
    sleep_files = sorted(SLEEP_DIR.glob("*.csv"))
    print(f"  {len(sleep_files)} sleep files\n")
    cat_maps = None
    try:
        from add_cats import load_app_categorization
        cat_maps = load_app_categorization()
        print("app categorization loaded ok\n")
    except Exception as e:
        print(f"app cat unavailable: {e}\n")
    for i, path in enumerate(sleep_files, 1):
        progress(i, len(sleep_files), path.name)
        df  = _process_sleep_file(path, cat_maps)
        out = MERGES_DIR / path.name
        df.to_csv(out, index=False)
        nights = df["calendarDate"].nunique() if not df.empty else 0
        print(f" {nights} nights, {len(df)} rows")


# stage 2: gps
def _parse_gps_file(gps_csv):
    df = pd.read_csv(gps_csv)
    if "data" not in df.columns: raise ValueError("missing 'data' column")
    expanded = df["data"].apply(json.loads).apply(pd.Series)
    ts  = pd.to_datetime(df["day"], errors="coerce")
    out = pd.concat([expanded, ts.rename("TIMESTAMP"), df["uid"].rename("UID")], axis=1)
    out.columns = out.columns.str.upper().str.strip()
    return out.dropna(subset=["LATITUDE","LONGITUDE","TIMESTAMP"])

def _load_merge_for_gps(uid):
    path = MERGES_DIR / f"{uid}@mlife.csv"
    if not path.exists(): return None
    df = pd.read_csv(path)
    df["time_point_local"]  = pd.to_datetime(df["time_point_local"],  errors="coerce")
    df["sleep_start_local"] = pd.to_datetime(df["sleep_start_local"], errors="coerce")
    df = df.dropna(subset=["time_point_local"])
    row_counts = (
        df.groupby(["calendarDate","sleep_start_local"]).size().reset_index(name="n")
        .sort_values(["calendarDate","n","sleep_start_local"], ascending=[True,False,False])
    )
    best = row_counts.drop_duplicates("calendarDate", keep="first")[["calendarDate","sleep_start_local"]]
    return df.merge(best, on=["calendarDate","sleep_start_local"], how="inner")

def _align_gps(gps_df, merge_df):
    bw = pd.Timedelta(minutes=BIN_MINUTES)
    merge_df = merge_df.copy()
    merge_df["bin_end"] = merge_df["time_point_local"] + bw
    keys = merge_df[["time_point_local","bin_end","calendarDate",
                      "sleep_start_local","relative_time_minutes"]].sort_values("time_point_local")
    out = pd.merge_asof(
        gps_df.sort_values("TIMESTAMP"),
        keys.rename(columns={"time_point_local":"BIN_START"}),
        left_on="TIMESTAMP", right_on="BIN_START", direction="backward",
    )
    out = out.dropna(subset=["BIN_START"])
    out = out[out["TIMESTAMP"] < out["bin_end"]].drop(columns=["BIN_START","bin_end"])
    ctx = ["calendarDate","sleep_start_local","relative_time_minutes"]
    return out[ctx + [c for c in out.columns if c not in ctx]]

def stage2_preprocess_gps():
    print("stage 2 : gps preprocessing")
    GPS_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    gps_files = sorted(GPS_RAW_DIR.glob("*@mlife_2.csv"))
    print(f" {len(gps_files)} GPS files\n")
    for i, gps_csv in enumerate(gps_files, 1):
        uid  = gps_csv.stem.replace("@mlife_2","")
        out_csv = GPS_PARSED_DIR / f"{uid}_gps_parsed.csv"
        progress(i, len(gps_files), uid)
        try:
            gps_df= _parse_gps_file(gps_csv)
            merge_df= _load_merge_for_gps(uid)
            if merge_df is None: print("no merge file"); continue
            if gps_df.empty: print("no valid GPS rows"); continue
            aligned= _align_gps(gps_df, merge_df)
            if aligned.empty: print("no GPS in any window"); continue
            aligned.to_csv(out_csv, index=False)
            print(f"{len(aligned)} pts /{aligned['calendarDate'].nunique()} nights")
        except Exception as e:
            print(f"err: {e}")


# stage 3: nightly summary
def _load_ema():
    # load all ema items, pivot to one row per person date
    ema = pd.read_csv(EMA_PATH, low_memory=False)
    ema["ema_time"] = ema["ema_time"].astype(str).str.lower().str.strip()
    ema["ema_date"] = pd.to_datetime(ema["date_rep"], errors="coerce").dt.date
    ema["mlife_id"] = ema["mlife_id"].astype(str).str.strip()

    # print ema file cols to verify all items present
    print(f"  EMA file columns: {list(ema.columns)}")
    print(f"  Looking for these items:")
    for col, name in EMA_ITEMS.items():
        found = col in ema.columns
        n = ema[col].notna().sum() if found else 0
        print(f" {col:10s} -> {name:12s} : {'found' if found else 'missing'}"
              + (f" n={n}" if found else " will be skipped"))
    print()

    for col in EMA_ITEMS:
        if col not in ema.columns:
            print(f"{col} not found in EMA file")
            ema[col] = np.nan
        else:
            ema[col] = pd.to_numeric(ema[col], errors="coerce")

    # morning + afternoon only
    ema = ema[ema["ema_time"].isin(["morning","afternoon"])]

    pivots = []
    for col, name in EMA_ITEMS.items():
        p = (
            ema.pivot_table(index=["mlife_id","ema_date"], columns="ema_time",
                            values=col, aggfunc="mean")
               .reset_index()
               .rename(columns={"morning": f"{name}_am", "afternoon": f"{name}_afternoon"})
        )
        pivots.append(p)

    result = pivots[0]
    for p in pivots[1:]:
        result = result.merge(p, on=["mlife_id","ema_date"], how="outer")

    # composite score = mean of all items
    am_cols= [f"{n}_am" for n in EMA_ITEMS.values() if f"{n}_am" in result.columns]
    pm_cols= [f"{n}_afternoon" for n in EMA_ITEMS.values() if f"{n}_afternoon" in result.columns]
    result["negative_affect_am"] = result[am_cols].mean(axis=1)
    result["negative_affect_afternoon"] = result[pm_cols].mean(axis=1)
    return result

def _category_minutes(phone_rows):
    counts = {c: 0 for c in APP_CATS}
    for val in phone_rows["reducedCats"]:
        if pd.isna(val) or val == "NONE": continue
        for cat in {c.strip() for c in str(val).split("|") if c.strip() not in ("NONE","")}:
            counts[cat if cat in counts else "UNKNOWN"] += BIN_MINUTES
    return counts

def _summarize_one(merge_path, gps_path, ema_daily):
    merge = pd.read_csv(merge_path)
    gps = pd.read_csv(gps_path)

    merge["sleep_start_local"] = pd.to_datetime(merge["sleep_start_local"])
    gps["sleep_start_local"] = pd.to_datetime(gps["sleep_start_local"])
    gps["TIMESTAMP"] = pd.to_datetime(gps["TIMESTAMP"], errors="coerce")
    gps["calendarDate"] = gps["calendarDate"].astype(str)
    gps = gps.dropna(subset=["LATITUDE","LONGITUDE","TIMESTAMP"])

    # keep best sleep_start per date
    row_counts = (
        merge.groupby(["calendarDate","sleep_start_local"]).size().reset_index(name="n")
        .sort_values(["calendarDate","n","sleep_start_local"], ascending=[True,False,False])
    )
    best = row_counts.drop_duplicates("calendarDate", keep="first")[["calendarDate","sleep_start_local"]]
    merge = merge.merge(best, on=["calendarDate","sleep_start_local"], how="inner")

    date_to_sleep = dict(zip(best["calendarDate"].astype(str), best["sleep_start_local"]))
    gps["sleep_start_local"] = gps["calendarDate"].map(date_to_sleep)
    gps = gps.dropna(subset=["sleep_start_local"])

    rows = []
    for sleep_time, night in merge.groupby("sleep_start_local", sort=True):
        mlife_id = night["mlife_id"].iloc[0]
        cal_date = night["calendarDate"].iloc[0]

    # dedup phone bins, count if open_app or background_apps is set
        pre = night[night["relative_time_minutes"] < 0].drop_duplicates("relative_time_minutes")
        phone_rows = pre[
            (pre["open_app"] != "NONE") |
            (pre["background_apps"].fillna("NONE") != "NONE")
        ]
        mins_phone = min(len(phone_rows) * BIN_MINUTES, WINDOW_HOURS * 60)
        cat_mins = _category_minutes(phone_rows)

        #gps
        night_gps = gps[gps["sleep_start_local"] == sleep_time].sort_values("TIMESTAMP")
        n_pts = len(night_gps)
        if n_pts > 0:
            closest = night_gps.loc[(night_gps["TIMESTAMP"] - sleep_time).abs().idxmin()]
            lat, lon = float(closest["LATITUDE"]), float(closest["LONGITUDE"])
            coords = list(zip(night_gps["LATITUDE"].astype(float), night_gps["LONGITUDE"].astype(float)))
            total_m = sum(haversine(*coords[i-1], *coords[i]) for i in range(1, len(coords)))
            disp = haversine(*coords[0], *coords[-1]) if len(coords) > 1 else 0
            moving = int(disp > 100)

            #gps features
            # n_locations: 100m grid cells in pre sleep window
            grid = set((round(la, 3), round(lo, 3)) for la, lo in coords)
            n_locations = len(grid)
            # distance from sleep location = closest to sleep start point
            dists_home = [haversine(la, lo, lat, lon) for la, lo in coords]
            max_dist_from_home_m = max(dists_home)
            # fraction of pre sleep window within 100m of home (approx —
            # gps points aren't evenly spaced, so i assumed roughly uniform sampling)
            pct_home = sum(1 for d in dists_home if d <= 100) / len(dists_home)
            minutes_at_home = pct_home * WINDOW_HOURS * 60
        else:
            lat = lon = total_m = np.nan; moving = np.nan
            n_locations = max_dist_from_home_m = minutes_at_home = np.nan

        # match ema on date, pull all cols dynamically
        ema_row = ema_daily[
            (ema_daily["mlife_id"] == str(mlife_id).strip()) &
            (ema_daily["ema_date"] == pd.to_datetime(cal_date).date())
        ]
        ema_cols = [c for c in ema_daily.columns if c not in ["mlife_id","ema_date"]]
        ema_vals = {
            col: ema_row[col].iloc[0] if (not ema_row.empty and col in ema_row.columns) else np.nan
            for col in ema_cols
        }

        #check unlock on both sleep date and previous date
        unlock_map = _load_unlock(mlife_id)
        unlock_ok = (
            bool(unlock_map.get(str(sleep_time.date()))) or
            bool(unlock_map.get(str((sleep_time - pd.Timedelta(days=1)).date())))
        )
#make df
        rows.append({
            "mlife_id": mlife_id,
            "calendarDate": cal_date,
            "sleep_start_local": sleep_time,
            "sleep_validation": night["validation"].iloc[0] if "validation" in night.columns else "",
            "minutes_on_phone_before_bed": mins_phone,
            "sleep_base_lat": lat,
            "sleep_base_lon": lon,
            "sleep_base_is_moving": moving,
            "total_distance_m": total_m,
            "gps_point_count": n_pts,
            "n_locations": n_locations,
            "minutes_at_home": minutes_at_home,
            "max_dist_from_home_m": max_dist_from_home_m,
            **ema_vals,
            "has_app_data": int(len(phone_rows) > 0),
            "unlock_available": int(unlock_ok),
            **{f"minutes_{c.lower().replace(' ','_')}": v for c, v in cat_mins.items()},
        })
    return pd.DataFrame(rows)

def _check_stage3(summary_files):
    all_dfs = []
    for f in summary_files:
        try: all_dfs.append(pd.read_csv(f))
        except Exception: pass
    if not all_dfs:
        print("no summary files found"); return
    df = pd.concat(all_dfs, ignore_index=True)
    total = len(df)

    print(f"total rows : {total}")
    print(f"participants: {df['mlife_id'].nunique()}")

    ema_pfx = ["fatigue","anhedonia","depression","anxiety","worry","somatic","negative_affect"]
    ema_cols = [c for c in df.columns if any(c.startswith(p) for p in ema_pfx)]
    if ema_cols:
        print(f"\nEMA match rate : {100*df[ema_cols].notna().any(axis=1).mean():.1f}%")
        for col in ema_cols:
            vals = df[col].dropna()
            if len(vals):
                bad = (~vals.between(0,100)).sum()
                print(f"  {col:30s}: n={len(vals):5d}  median={vals.median():.1f}"
                      + (" BUG" if bad > 0 else " ok"))

    lats = df["sleep_base_lat"].dropna()
    print(f"\nGPS coverage : {(df['gps_point_count']>0).mean():.1%}")
    if len(lats):
        print(f"lat range: {lats.min():.3f} to {lats.max():.3f}")
        print(f"invalid coords: {(~lats.between(-90,90)).sum()} ok")

    phone = df["minutes_on_phone_before_bed"].dropna()
    over  = (phone > WINDOW_HOURS*60).sum()
    print(f"\nphone > {WINDOW_HOURS*60}m : {over}" + (" BUG" if over > 0 else " ok"))
    print(f"phone range: {phone.min():.0f} to {phone.max():.0f}  median={phone.median():.0f}")
    print(f"zero nights: {(phone==0).sum()} ({100*(phone==0).mean():.1f}%)")

    if "sleep_validation" in df.columns:
        print(f"\n sleep validation:")
        for val, cnt in df["sleep_validation"].value_counts().items():
            print(f" {str(val):28s}: {cnt} ({100*cnt/total:.1f}%)")
        if df["sleep_validation"].str.contains("TENTATIVE", na=False).mean() > 0.5:
            print("majority tentative")

def stage3_build_summary():
    print("stage 3 build nightly summaries")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    print("loading ema ...", end=" ", flush=True)
    ema_daily = _load_ema()
    print(f"{len(ema_daily)} participant-days ok\n")
    merge_files = sorted(MERGES_DIR.glob("*.csv"))
    print(f"  {len(merge_files)} merge files\n")
    for i, merge_path in enumerate(merge_files, 1):
        uid = merge_path.stem.split("@")[0]
        gps_path = GPS_PARSED_DIR / f"{uid}_gps_parsed.csv"
        progress(i, len(merge_files), merge_path.name)
        if not gps_path.exists():
            print(" no GPS file, skipping"); continue
        try:
            df = _summarize_one(merge_path, gps_path, ema_daily)
            out= SUMMARY_DIR / f"{uid}_summary.csv"
            df.to_csv(out, index=False)
            ema_pfx = ["fatigue","anhedonia","depression","anxiety","worry","somatic","negative_affect"]
            ema_cols= [c for c in df.columns if any(c.startswith(p) for p in ema_pfx)]
            ema_ok = df[ema_cols].notna().any(axis=1).mean() if ema_cols else 0
            gps_ok = (df["gps_point_count"] > 0).mean()
            print(f"{len(df)} nights | EMA {100*ema_ok:.0f}% | GPS {100*gps_ok:.0f}%")
        except Exception as e:
            print(f"err: {e}")
    _check_stage3(sorted(SUMMARY_DIR.glob("*_summary.csv")))


def diagnose():
    print("debugging")
    ema = pd.read_csv(EMA_PATH, low_memory=False)
    ema["ema_time"] = ema["ema_time"].astype(str).str.lower().str.strip()
    print(f"columns: {list(ema.columns)}")
    print(f"ema_time values: {sorted(ema['ema_time'].unique())}")
    for col, name in EMA_ITEMS.items():
        vals = pd.to_numeric(ema[col], errors="coerce").dropna()
        print(f"  {col} ({name}): n={len(vals)}  min={vals.min():.0f}  median={vals.median():.0f}  max={vals.max():.0f}")

    uids_123 = {f.stem for f in RUNAPP_DIR.glob("*.csv")} if RUNAPP_DIR.exists() else set()
    uids_122 = {f.stem for f in RUNAPP_DIR_ALT.glob("*.csv")} if RUNAPP_DIR_ALT.exists() else set()
    sleep_uids = {f.stem for f in SLEEP_DIR.glob("*.csv")}
    print(f"in 123 only : {len(uids_123 - uids_122)}")
    print(f"in 122 only : {len(uids_122 - uids_123)}")
    print(f"in both : {len(uids_123 & uids_122)}")
    print(f"in neither: {len(sleep_uids - (uids_123 | uids_122))}")

    _check_stage3(sorted(SUMMARY_DIR.glob("*_summary.csv")))


def main():
    stages = set(sys.argv[1:]) if len(sys.argv) > 1 else {"0","1","2","3"}
    if "diagnose" in stages:
        diagnose(); return
    print(f" base : {BASE_DIR}")
    print(f"window : {WINDOW_HOURS}h | bin: {BIN_MINUTES}min")
    print(f"stages : {sorted(stages)}")
    if "0" in stages: stage0_clean_unlock()
    if "1" in stages: stage1_build_timelines()
    if "2" in stages: stage2_preprocess_gps()
    if "3" in stages: stage3_build_summary()
    print("\ndone\n")

if __name__ == "__main__":
    main()