"""
Input: Raw SportsDataIO team-level files for each season (2022-2025)

NOTE: 2026+ (and 2025 week 18) come from download_nflverse.py, which writes
straight to processed/clean_team_stats_season/. If you re-run this script for
2025, re-run download_nflverse.py --season 2025 afterwards to restore week 18.

Renames some column names and computes statistics like 
point differential (target variable) and win (bool).

Returns: cleaned team-level file for each season saved to: 
    processed/clean_team_stats_season/
"""
import pandas as pd
from nfl_prediction.config import PROC_DIR, RAW_STATS_CSV, CLEAN_STATS_DIR

PROC_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_STATS_DIR.mkdir(parents=True, exist_ok=True)

def clean_season(season: str):
    """
    Clean team stats dataset for a given season, rename some features, and 
    compute some basic features.
    """
    file_path = RAW_STATS_CSV / f"team_stats_{season}REG_all_weeks.csv"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\nCleaning {file_path}...")


    df = pd.read_csv(file_path)

    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.lower()
    )

    rename_map = {
        "score": "points_for",
        "opponentscore": "points_against",
        "team": "team",
        "opponent": "opponent",
        "homeoraway": "home_away"
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [
        col for col in df.columns
        if col not in ["team", "opponent", "home_away", "stadium", "dayofweek", "date"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    df = df.dropna(subset=["points_for", "points_against"])

    df["point_diff"] = df["points_for"] - df["points_against"]
    df["win"] = (df["point_diff"] > 0).astype(int)

    if "takeaways" in df.columns and "giveaways" in df.columns:
        df["turnover_diff"] = df["takeaways"] - df["giveaways"]

    if "thirddownattempts" in df.columns and "thirddownconversions" in df.columns:
        df["third_down_pct"] = df["thirddownconversions"] / df["thirddownattempts"].replace(0, 1)

    df["home"] = (df["home_away"].str.upper() == "HOME").astype(int)

    out_file = CLEAN_STATS_DIR / f"clean_team_stats_{season}.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved cleaned file to {out_file} ({len(df)} rows)")

def main():
    seasons = ["2022", "2023", "2024", "2025"]
    for season in seasons:
        clean_season(season)

if __name__ == "__main__":
    main()