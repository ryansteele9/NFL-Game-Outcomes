"""
Input:
    All team files for each season
    nflfastR EPA metrics file (all seasons 2022-2025)

Adds computes and adds features to team files such as rolling averages,
EPA-related metrics (from nflfastR) and cumulative stats. Lags data so that for
a given game, statistics columns contain data for the next game. This is so that
when training model to predict point differential, model doesn't have access to
stats from the game(s) it is trying to predict. Prevents leakage. Adds dummy 
week so that when lagging the statistics columns, statistics from final week in
season isn't lost and can be used to predict future games.

Returns: 
    Team files for each season with new features, saved to: processed/features/
"""
import pandas as pd
import numpy as np

from nfl_prediction.config import TEAMS_DIR, FEATURES_DIR, SEASON_STRS
from nfl_prediction.data.nflfastr_epa import load_nflfastr_team_epa

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

NFLFASTR_EPA = load_nflfastr_team_epa()

EPA_MERGE_COLS = [
    "off_epa_per_play", "off_success_rate", "off_dropback_epa", "off_rush_epa",
    "def_epa_per_play", "def_success_rate", "def_dropback_epa_against",
    "def_rush_epa_against",
]

# Early-season carry-over: the last PRIOR_GAMES games of the previous season are
# prepended (pulled PRIOR_REGRESSION of the way toward that season's league
# average) so week 1-4 rolling/lagged features aren't empty or based on 1-2
# games. Set EARLY_SEASON_CARRYOVER = False for the old per-season reset.
EARLY_SEASON_CARRYOVER = True
PRIOR_GAMES = 5            # longest rolling window
PRIOR_REGRESSION = 1.0 / 3.0

# Columns never regressed/treated as stats
ID_COLS = {
    "gamekey", "season", "week", "date", "team", "opponent", "home_away",
    "stadium", "dayofweek", "teamgameid", "teamid", "opponentid", "scoreid",
    "seasontype", "home", "nflverse_game_id",
}

def add_future_dummy_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dummy week for upcoming week. Because data is shifted, dummy future week
    will house data from the most recent week. NaNs for columns whose data 
    doesn't exist yet for future dummy week.
    """
    required = {"season", "week", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for dummy creation: {missing}")
    
    groups = []
    for (season, team), g in df.groupby(["season", "team"]):
        g = g.sort_values("week").copy()
        max_week = int(g["week"].max())
        
        dummy = g.iloc[-1].copy()
        dummy["week"] = max_week + 1
        
        unknown_features = {
            "gamekey", "points_for", "points_against", "totalscore", "scorequarter1", 
            "scorequarter2", "scorequarter3", "scorequarter4", "scoreovertime", 
            "opponentscorequarter1", "opponentscorequarter2", "opponentscorequarter3", 
            "opponentscorequarter4", "opponentscoreovertime", "point_diff", "date"
        }
        
        for feature in unknown_features:
            if feature in dummy.index:
                dummy[feature] = np.nan
        
        groups.append(pd.concat([g, pd.DataFrame([dummy])], ignore_index=True))
    
    return pd.concat(groups, ignore_index=True)

def _merge_epa(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        NFLFASTR_EPA[["season", "week", "team", *EPA_MERGE_COLS]],
        how="left",
        on=["season", "week", "team"],
    )


def league_means(season: str) -> pd.Series | None:
    """League-average value of every stat column for a season (teams + EPA)."""
    season_dir = TEAMS_DIR / season
    files = list(season_dir.glob("*.csv")) if season_dir.exists() else []
    if not files:
        return None
    games = _merge_epa(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    numeric = games.select_dtypes("number").drop(columns=list(ID_COLS), errors="ignore")
    return numeric.mean()


def prior_season_games(team: str, season: int, means: pd.Series | None) -> pd.DataFrame | None:
    """Last PRIOR_GAMES games of the team's previous season, regressed to the mean."""
    if not EARLY_SEASON_CARRYOVER or means is None:
        return None
    path = TEAMS_DIR / str(season - 1) / f"{team}_{season - 1}.csv"
    if not path.exists():
        return None

    prior = _merge_epa(pd.read_csv(path).sort_values("week")).tail(PRIOR_GAMES).copy()
    stat_cols = [c for c in means.index if c in prior.columns]
    prior[stat_cols] = means[stat_cols] + (1.0 - PRIOR_REGRESSION) * (prior[stat_cols] - means[stat_cols])
    prior["_is_prior"] = True
    return prior


def add_team_features(team_file_path, prior_means: pd.Series | None = None):
    """
    Adds rolling and cumulative features to a single team-season file.
    """
    df = pd.read_csv(team_file_path)
    df = df.sort_values(by="week").reset_index(drop=True)

    # Add the future dummy week BEFORE lagging, so that after the shift the
    # dummy row holds stats through the most recent game. (Previously it was
    # added after lagging, so it was a copy of the last row and ignored the
    # most recent game entirely.)
    df = add_future_dummy_week(df)
    
    season = int(df["season"].iloc[0])
    team = df["team"].iloc[0]
    
    df = _merge_epa(df)
    df["_is_prior"] = False

    # Prepend regressed prior-season games so early-season windows have data
    prior = prior_season_games(team, season, prior_means)
    if prior is not None:
        df = pd.concat([prior, df], ignore_index=True)
    df = df.copy()  # de-fragment before adding many feature columns

    df["rolling_points_for_3"] = df["points_for"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_points_against_3"] = df["points_against"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_point_diff_3"] = df["point_diff"].shift(1).rolling(window=3, min_periods=1).mean()

    if "offensiveyards" in df.columns:
        df["rolling_yards_total_3"] = df["offensiveyards"].shift(1).rolling(window=3, min_periods=1).mean()
    if "opponentoffensiveyards" in df.columns:
        df["rolling_yards_allowed_3"] = df["opponentoffensiveyards"].shift(1).rolling(window=3, min_periods=1).mean()
    if "turnover_diff" in df.columns:
        df["rolling_turnover_diff_3"] = df["turnover_diff"].shift(1).rolling(window=3, min_periods=1).mean()
    if "third_down_pct" in df.columns:
        df["rolling_third_down_pct_3"] = df["third_down_pct"].shift(1).rolling(window=3, min_periods=1).mean()

    df["rolling_win_rate_5"] = df["win"].shift(1).rolling(window=5, min_periods=1).mean()
    # Cumulative stats stay season-to-date (no carry-over)
    def season_cumsum(col: str) -> pd.Series:
        return df.groupby("season")[col].transform(lambda x: x.shift(1).cumsum())

    df["cumulative_points_for"] = season_cumsum("points_for")
    df["cumulative_points_against"] = season_cumsum("points_against")
    df["cumulative_wins"] = season_cumsum("win")
    
    EPA_COLS = [
        "off_epa_per_play",
        "off_success_rate",
        "off_dropback_epa",
        "off_rush_epa",
        "def_epa_per_play",
        "def_success_rate",
        "def_dropback_epa_against",
        "def_rush_epa_against",
    ]
    
    for col in EPA_COLS:
        if col in df.columns:
            df[f"{col}_rolling_3"] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
    
    df["off_epa_per_play_rolling_5"] = df["off_epa_per_play"].shift(1).rolling(window=5, min_periods=1).mean()
    df["def_epa_per_play_rolling_5"] = df["def_epa_per_play"].shift(1).rolling(window=5, min_periods=1).mean()
    
    do_not_lag = {
        "gamekey", "season", "week", "date", "team", "opponent", "home_away", 
        "stadium", "dayofweek", "teamgameid", "teamid", "opponentid", "win", 
        "points_for", "points_against", "point_diff", "totalscore", 
        "scorequarter1", "scorequarter2", "scorequarter3", "scorequarter4", 
        "scoreovertime", "opponentscorequarter1", "opponentscorequarter2", 
        "opponentscorequarter3", "opponentscorequarter4", "opponentscoreovertime",
        "seasontype", "_is_prior"
    }
    
    already_shifted = {
        "rolling_points_for_3", "rolling_points_against_3", 
        "rolling_point_diff_3", "rolling_yards_total_3", "rolling_yards_allowed_3", 
        "rolling_turnover_diff_3", "rolling_third_down_pct_3", "rolling_win_rate_5", 
        "cumulative_points_for", "cumulative_points_against", "cumulative_wins", 
        "off_epa_per_play_rolling_3", "off_success_rate_rolling_3",
        "off_dropback_epa_rolling_3", "off_rush_epa_rolling_3",
        "def_epa_per_play_rolling_3", "def_success_rate_rolling_3",
        "def_dropback_epa_against_rolling_3", "def_rush_epa_against_rolling_3",
        "epa_def_diff_rolling_3", "epa_off_diff_rolling_3", 
        "def_epa_per_play_rolling_5", "off_epa_per_play_rolling_5"
    }
    
    feature_cols = [
        col for col in df.columns 
        if col not in do_not_lag 
        and col not in already_shifted 
        and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    if feature_cols:
        df[feature_cols] = df.groupby("team")[feature_cols].shift(1)
        # df = df.dropna(subset=feature_cols) If need to drop rows without data from lag
        
    round_cols_1 = [
        "rolling_points_for_3", "rolling_points_against_3", 
        "rolling_point_diff_3", "rolling_yards_total_3", "rolling_yards_allowed_3"
    ]
    round_cols_3 = [
        "rolling_turnover_diff_3", "rolling_win_rate_5", "third_down_pct", 
        "rolling_third_down_pct_3"
    ]

    for col in round_cols_1:
        if col in df.columns:
            df[col] = df[col].round(1)

    for col in round_cols_3:
        if col in df.columns:
            df[col] = df[col].round(3)

    df = df[~df["_is_prior"].astype(bool)].drop(columns="_is_prior").reset_index(drop=True)
    
    return df

def process_season(season: str):
    """
    Process all teams in a given season.
    """
    season_team_dir = TEAMS_DIR / season
    season_output_dir = FEATURES_DIR / season
    season_output_dir.mkdir(parents=True, exist_ok=True)

    prior_means = league_means(str(int(season) - 1)) if EARLY_SEASON_CARRYOVER else None

    for team_file in season_team_dir.glob("*.csv"):
        team_name = team_file.stem.split("_")[0]
        print(f"Processing {team_name} ({season})")

        df_features = add_team_features(team_file, prior_means)

        out_path = season_output_dir / f"{team_name}_{season}_features.csv"
        df_features.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")

def main():
    for season in SEASON_STRS:
        print(f"\nBuilding features for season {season}...")
        process_season(season)

if __name__ == "__main__":
    main()