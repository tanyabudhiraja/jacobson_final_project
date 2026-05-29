import pandas as pd

CATS_PATH = "/Users/tanya/Desktop/code/vscode/jacobson project/Apps_categorization_final.csv"


def load_app_categorization():
    cats = pd.read_csv(CATS_PATH, dtype=str).fillna("UNKNOWN")
    cats["app_name"] = cats["app_name"].astype(str).str.strip().str.lower()
    google_map   = dict(zip(cats["app_name"], cats["googleCats"]))
    schoedel_map = dict(zip(cats["app_name"], cats["Schoedel"]))
    cats_map     = dict(zip(cats["app_name"], cats["cats"]))
    reduced_map  = dict(zip(cats["app_name"], cats["reducedCats"]))
    return google_map, schoedel_map, cats_map, reduced_map


def categorize_open_app(df, google_map, schoedel_map, cats_map, reduced_map):
    if df.empty or "open_app" not in df.columns:
        for col in ["googleCats", "Schoedel", "cats", "reducedCats"]:
            df[col] = []
        return df

    def map_one(val):
        if pd.isna(val) or str(val).strip() in ("NONE", ""):
            return ("NONE", "NONE", "NONE", "NONE")
        names = [x.strip().lower() for x in str(val).split("|") if x.strip()]
        g = [google_map.get(x, "UNKNOWN") for x in names]
        s = [schoedel_map.get(x, "UNKNOWN") for x in names]
        c = [cats_map.get(x, "UNKNOWN") for x in names]
        r = [reduced_map.get(x, "UNKNOWN") for x in names]
        return ("|".join(g), "|".join(s), "|".join(c), "|".join(r))

    cat_values = df["open_app"].apply(map_one)
    df[["googleCats", "Schoedel", "cats", "reducedCats"]] = pd.DataFrame(
        cat_values.tolist(), index=df.index
    )
    return df