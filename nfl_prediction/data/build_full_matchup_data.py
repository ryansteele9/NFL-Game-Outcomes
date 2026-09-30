"""
Input: Matchup-level files
Combines matchup-level files into one big file for all seasons. Adds elo rating
features to quantify team strength.
File saved to: processed/matchups
"""
import pandas as pd

from nfl_prediction.config import MATCHUPS_DIR, ODDS_PROC_DIR, SEASON_STRS
from nfl_prediction.data.team_ratings import add_elo_features



OUTPUT_PATH = MATCHUPS_DIR / "matchups_all_seasons.csv"

def load_all_odds() -> pd.DataFrame:
    """
    Load all processed odds files.
    """
    files = sorted(ODDS_PROC_DIR.glob("odds_*_week*.csv"))
    if not files:
        print("[WARNING] No odds files found; vegas lines will be missing.")
        return pd.DataFrame()
    
    frames = [pd.read_csv(file) for file in files]
    odds = pd.concat(frames, ignore_index=True)
    
    return odds
    
def combine_matchups():
    """
    Combines all season matchups datasets into one big dataset for seasons 
    in config.SEASONS. Adds elo ratings and vegas odds to dataset.
    """
    all_dfs = []
    for season in SEASON_STRS:
        file_path = MATCHUPS_DIR / f"matchups_{season}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["season"] = int(season)
            
            if "point_diff" in df.columns:
                before = len(df)
                df = df[df["point_diff"].notna()].copy()
                after = len(df)
                print(f"""Loaded {file_path} with {before} rows.
                      {before - after} rows with NaN point_diff dropped.
                      """)
            else:
                print(
                    f"WARNING: 'point_diff' not in {file_path} columns; "
                    "no filtering of dummy rows applied."
                )
            all_dfs.append(df)
        else:
            print(f"Missing file: {file_path}, skipping...")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        if "gamekey" not in full_df.columns:
            full_df = full_df.sort_values(["season", "week"])
        else:
            full_df = full_df.sort_values(["season", "week", "gamekey"])
        
        full_df = add_elo_features(full_df)
        
        odds_all = load_all_odds()
        if not odds_all.empty:
            odds_for_merge = odds_all.rename(
                columns={
                    "home_team": "team",
                    "away_team": "opponent"
                }
            )
        
            full_df = full_df.merge(
                odds_for_merge, 
                on=["season", "week", "team", "opponent"],
                how="left",
            )
            
            vegas_cols = [
                "home_moneyline",
                "away_moneyline",
                "home_implied_prob",
                "vegas_spread",
                "vegas_total",
            ]
            
            for col in vegas_cols:
                if col in full_df.columns:
                    if "moneyline" in col:
                        full_df[col] = full_df[col].clip(-300, 300)
                    
        else:
            print("No odds merged (no odds data found).")
        
        full_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nCombined dataset saved to: {OUTPUT_PATH}")
        print(f"Total rows: {len(full_df)}")
    else:
        print("No matchup files found!")

if __name__ == "__main__":
    combine_matchups()
