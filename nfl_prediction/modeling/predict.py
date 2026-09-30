"""
Predicts the point differentials for given future week. Pulls NFl schedule for
given season and week from nflverse and builds matchup DataFrame for
these future matchups. Adds adds features to ensure future matchup df has
same structure as the DataFrame used for training model. Loads trained model and
adds model's predictions to df. Then, adjusts for injuries using injury
adjustment functions and prints adjusted predicted point differentials for
each matchup in week.

Args:
    season (int): NFL season to predict
    week (int): NFL week to predict
"""
from pathlib import Path
import pickle
from typing import Optional, List
import os

import numpy as np
import pandas as pd
from loguru import logger
import typer

from nfl_prediction.data.build_matchup_data import add_matchup_strength_features
from nfl_prediction.data.team_ratings import get_elo_ratings_to_week
from nfl_prediction.data.injury_adjust import compute_team_injury_adjustments, apply_injury_adjustments
from nfl_prediction.data.build_matchup_data import add_matchup_strength_features
from nfl_prediction.config import MODELS_DIR, MATCHUPS_DIR, FEATURES_DIR, RAW_DIR, ODDS_PROC_DIR

app = typer.Typer(help="Predict NFL game outcomes using trained XGBoost model.")

SCHEDULES_DIR = RAW_DIR / "schedules"
SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)

VEGAS_SCALE_MAP = {
    "home_moneyline": 0,
    "away_moneyline": 0,
    "home_implied_prob": 1,
    "vegas_spread": 1,
}

def scale_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent scaling to Vegas-related columns (in-place)."""
    for col, factor in VEGAS_SCALE_MAP.items():
        if col in df.columns:
            df[col] = df[col] * factor
        else:
            logger.debug(f"Vegas column '{col}' not found in dataframe during scaling.")
    return df

def default_model_path() -> Path:
    return MODELS_DIR / "xgb_point_diff.pkl"

def load_model_and_features(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError
    
    logger.info(f"Loading Model from {model_path}")
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    
    model = data["model"]
    if "feature_cols" in data:
        feature_cols = data["feature_cols"]
    elif "features" in data:
        feature_cols = data["features"]
    else:
        raise ValueError(
            "Model pickle must contain 'feature_cols' or 'features' "
            "listing the feature column names."
        )
    logger.info(f"Loaded model with {len(feature_cols)} features.")
    return model, feature_cols

def load_historical_matchups() -> pd.DataFrame:
    """
    Load full historical matchup dataset to compute elos.
    """
    
    path = MATCHUPS_DIR / "matchups_all_seasons.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_full_matchup_data.py before predicting.")
    
    return pd.read_csv(path)

def load_odds_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Load processed odds for a given season/week.
    """
    odds_path = ODDS_PROC_DIR / f"odds_{season}_week{week:02d}.csv"
    if not odds_path.exists():
        logger.warning(f"No odds file found at {odds_path}; vegas features will be missing.")
        return pd.DataFrame()
    
    odds = pd.read_csv(odds_path)
    
    rename_map = {
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    }
    
    for old, new in rename_map.items():
        if old in odds.columns and new not in odds.columns:
            odds = odds.rename(columns={old: new})
    
    required = {"season", "week", "home_team", "away_team"}
    missing = required - set(odds.columns)
    if missing:
        logger.warning(f"Odds file missing required columns {missing}; will not merge odds.")
        return pd.DataFrame()
    
    odds = (
        odds.groupby(["season", "week", "home_team", "away_team"], as_index=False)
        .agg(
            {
                "vegas_spread": "mean",
                "vegas_total": "mean",
                "home_moneyline": "mean",
                "away_moneyline": "mean",
                "home_implied_prob": "mean",
            }
        )
    )
    
    return odds

def fetch_schedules(season: int, week: int) -> pd.DataFrame:
    """
    Pull the weekly schedule from nflverse (free; replaces the SportsDataIO
    ScoresByWeek call).

    Returns DataFrame with information about future matchups.
    """
    from nfl_prediction.data.download_nflverse import load_schedule

    sched = load_schedule([season])
    df = sched[sched["week"] == week]
    if df.empty:
        raise ValueError(f"No games returned for season={season}, week={week}")

    df = df.assign(season=season, week=week)
    return df[["season", "week", "game_id", "home_team", "away_team", "gameday"]]

def load_schedule(season: int, week: int, season_type: str = "REG") -> pd.DataFrame:
    csv_path = SCHEDULES_DIR / f"schedule_{season}{season_type}_{week:02d}.csv"
    if csv_path.exists():
        logger.info(f"Loading schedule from cache: {csv_path}")
        return pd.read_csv(csv_path)

    df = fetch_schedules(season, week)
    df.to_csv(csv_path, index=False)
    logger.info(f"Cached schedule to {csv_path}")
    return df

def team_features_path(season: int, team: str) -> Path:
    season_dir = FEATURES_DIR / str(season)
    return season_dir / f"{team}_{season}_features.csv"

def get_features_for_future_week(season: int, future_week: int, team: str) -> pd.Series:
    path = team_features_path(season, team)
    if not path.exists():
        raise FileNotFoundError(f"Feature file for team {team} not found at {path}")
    
    df = pd.read_csv(path)
    if "week" not in df.columns:
        raise ValueError(f"'week' column missing in {path}")
    
    df_here = df.loc[df["week"] == future_week]
    if df_here.empty:
        max_week = df["week"].max()
        logger.warning(
            f"No features for {team} week {future_week} in {path}; "
            f"falling back to week {max_week}"
        )
        df_here = df.loc[df["week"] == max_week]
    
    return df_here.sort_values("week").iloc[-1]

def build_future_matchups(season: int, week: int, season_type: str = "REG") -> pd.DataFrame:
    """
    For each given future week, build a single-row matchup using data from dummy rows
    """
    schedule_df = load_schedule(season, week, season_type)
    hist = load_historical_matchups()
    elo_ratings = get_elo_ratings_to_week(hist, season, week)
    
    def elo_lookup(team: str) -> float:
        # Previously returned elo_ratings.get("home_elo_pre", 1500.0), which is
        # always 1500 because the dict is keyed by team.
        return elo_ratings.get(team, 1500.0)
    
    rows = []
    logger.info(f"Building future matchups for {len(schedule_df)} games...")
    
    for _, game in schedule_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        home_feat = get_features_for_future_week(season, week, home)
        away_feat = get_features_for_future_week(season, week, away)
        
        row = {
            "season": season,
            "week": week,
            "game_id": game["game_id"],
            "home_team": home,
            "away_team": away,
        }
        
        for col, val in home_feat.items():
            row[col] = val
        
        for col, val in away_feat.items():
            row[f"opp_{col}"] = val
        
        rows.append(row)
        
    df_matchups = pd.DataFrame(rows)
    df_matchups = add_matchup_strength_features(df_matchups)
    
    df_matchups["home_elo_pre"] = df_matchups["home_team"].map(elo_lookup)
    df_matchups["away_elo_pre"] = df_matchups["away_team"].map(elo_lookup)
    df_matchups["diff_elo_pre"] = df_matchups["home_elo_pre"] - df_matchups["away_elo_pre"]
    
    odds = load_odds_for_week(season, week)
    if not odds.empty:
        before = len(df_matchups)
        df_matchups = df_matchups.merge(
            odds,
            on=["season", "week", "home_team", "away_team"],
            how="left",
        )
        logger.info(
            f"Merged vegas odds for {df_matchups['vegas_spread'].notna().sum()} "
            f"of {before} games."
        )
    else:
        for col in [
            "vegas_spread",
            "vegas_total",
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
        ]:
            if col not in df_matchups.columns:
                df_matchups[col] = np.nan
    
    df_matchups = scale_vegas_features(df_matchups)
    
    logger.info(f"Constructed {len(df_matchups)} future matchup rows.")
    return df_matchups

def add_predictions(df_games: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df_games.columns]
    if missing:
        raise ValueError(f"Future matchup data missing feature columns: {missing}")
    
    df_games[feature_cols] = df_games[feature_cols].fillna(0)
    
    X = df_games[feature_cols].to_numpy()
    preds = model.predict(X)
    
    df_games["pred_point_diff"] = preds
    df_games["pred_winner"] = np.where(preds > 0, "home", "away")
    
    spread = np.abs(preds)
    win_prob = 0.5 + 0.4 * np.tanh(spread / 10.0)
    df_games["pred_home_win_prob"] = np.where(preds > 0, win_prob, 1.0 - win_prob)
    
    return df_games

def print_pred_results(df_pred: pd.DataFrame):
    cols = [col for col in ["season", "week", "game_id", "home_team", "away_team"] if col in df_pred.columns]
    if "point_diff_adj" in df_pred:
        cols += ["point_diff_adj", "inj_adj_team", "inj_adj_opp"]
    
    cols += ["pred_point_diff"]
    cols += ["pred_winner", "pred_home_win_prob"]
    
    output = df_pred[cols].copy()
    if "point_diff_adj" in df_pred:
        output["point_diff_adj"] = output["point_diff_adj"].round(2)
    else:
        output["pred_point_diff"] = output["pred_point_diff"].round(2)
    output["pred_point_diff"] = output["pred_point_diff"].round(2)
    output["pred_home_win_prob"] = output["pred_home_win_prob"].round(3)

    print()
    print(output.to_string(index=False))
    print()

@app.command()
def main(
    season: int = typer.Option(..., help="Season year, e.g. 2025"), 
    week: int = typer.Option(..., help="Future week number, e.g. 13"), 
    season_type: str = typer.Option(
        "REG", help="Season type: REG, PRE, or POST"
    ),
    model_path: Optional[Path] = typer.Option(
        None, help="Optional path to model pickle (defaults to MODELS_DIR)."
    ),
    home_team: Optional[str] = typer.Option(
        None, help="Optional filter: only this home team code (e.g. BAL)."
    ),
    away_team: Optional[str] = typer.Option(
        None, help="Optional filter: only this away team code (e.g. NYJ)."
    ),
    game_id: Optional[str] = typer.Option(
        None, help="Optional filter: only this specific game_id/GameKey."
    ),
    save_matchups: bool = typer.Option(
        False,
        help="If set, save synthetic future matchup rows to data/matchups/.",
    ),
):
    """
    Predict point differential for all games in a scheduled future week.
    """
    model_path = model_path or default_model_path()
    model, feature_cols = load_model_and_features(model_path)
    
    logger.info(f"Predicting season={season}{season_type}, week={week}")
    df_matchups = build_future_matchups(season, week, season_type=season_type)
    
    # optional filters
    mask = pd.Series(True, index=df_matchups.index)
    if home_team:
        mask &= df_matchups["home_team"] == home_team
    if away_team:
        mask &= df_matchups["away_team"] == away_team
    if game_id:
        mask &= df_matchups["game_id"] == game_id
    
    df_matchups = df_matchups.loc[mask].copy()
    if df_matchups.empty:
        raise ValueError("No games match the given filters.")
    
    if save_matchups:
        out_path = MATCHUPS_DIR / f"future_matchups_{season}{season_type}_week{week:02d}.csv"
        df_matchups.to_csv(out_path, index=False)
        logger.info(f"Saved future matchups to {out_path}")
    
    df_pred = add_predictions(df_matchups, model, feature_cols)
    
    injuries_dir = Path("data/processed/injuries")
    inj_path = injuries_dir / f"injuries_week{week:02d}_curated.csv"
    
    if inj_path.exists():
        logger.info(f"Loading injury adjustments from {inj_path}")
        inj_df = pd.read_csv(inj_path)
        
        team_adj = compute_team_injury_adjustments(inj_df)
        df_pred = apply_injury_adjustments(df_pred, team_adj)
    else:
        logger.warning(f"No injury file found at {inj_path}; predictions are unadjusted.")
    
    print_pred_results(df_pred)
    
    logger.success("Prediction Complete.")


if __name__ == "__main__":
    app()
