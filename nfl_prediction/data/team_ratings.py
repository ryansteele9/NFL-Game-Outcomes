"""
Functions that compute elo features given a matchup-level dataset. Functions
implemented in build_full_matchup_data.py and predict.py.

Ratings carry over between seasons with partial regression toward the mean:
at the start of each season every team's rating is pulled `season_regression`
of the way back to `base_rating` (1/3 is the FiveThirtyEight convention). Set
season_regression=1.0 to reproduce the old behavior (full reset to 1500 every
season).
"""
import pandas as pd
from typing import Dict, List, Optional

BASE_RATING = 1500.0
K = 20.0
HOME_FIELD_ADVANTAGE = 2.5
SEASON_REGRESSION = 1.0 / 3.0


def _regress(ratings: Dict[str, float], base_rating: float, season_regression: float) -> Dict[str, float]:
    return {
        team: base_rating + (1.0 - season_regression) * (rating - base_rating)
        for team, rating in ratings.items()
    }


def _update(rating_home: float, rating_away: float, margin: float,
            k: float, home_field_advantage: float) -> float:
    """Return the Elo change for the home team (away team gets the negative)."""
    rating_diff = (rating_home - rating_away) + home_field_advantage
    expected_home_win = 1.0 / (1.0 + 10 ** (-(rating_diff) / 400.0))

    if margin > 0:
        actual_home_win = 1.0
    elif margin < 0:
        actual_home_win = 0.0
    else:
        actual_home_win = 0.5

    # margin of victory
    mov_scale = max(0.5, (abs(margin) / 14.0) ** 0.5)
    return k * mov_scale * (actual_home_win - expected_home_win)


def _run_elo(
    df: pd.DataFrame,
    base_rating: float,
    k: float,
    home_field_advantage: float,
    season_regression: float,
    stop_at: Optional[tuple] = None,
):
    """
    Walk through games in (season, week, gamekey) order.

    Returns (home_pre, away_pre, ratings) where ratings is the dict of current
    ratings. If stop_at=(season, week) is given, stops before the first game of
    that week and returns ratings as of that point (regressed if the season
    just rolled over).
    """
    sort_cols = [c for c in ["season", "week", "gamekey"] if c in df.columns]
    df = df.sort_values(sort_cols)

    ratings: Dict[str, float] = {}
    current_season = None
    home_pre: List[float] = []
    away_pre: List[float] = []
    index: List = []

    for idx, row in df.iterrows():
        season = int(row["season"])
        week = int(row["week"])

        if stop_at is not None and (season, week) >= stop_at:
            break

        if current_season is not None and season != current_season:
            ratings = _regress(ratings, base_rating, season_regression)
        current_season = season

        home = row["team"]
        away = row["opponent"]
        rating_home = ratings.get(home, base_rating)
        rating_away = ratings.get(away, base_rating)
        home_pre.append(rating_home)
        away_pre.append(rating_away)
        index.append(idx)

        delta = _update(rating_home, rating_away, float(row["point_diff"]), k, home_field_advantage)
        ratings[home] = rating_home + delta
        ratings[away] = rating_away - delta

    if stop_at is not None and current_season is not None and stop_at[0] != current_season:
        # Target week is the first of a new season: apply the offseason regression
        ratings = _regress(ratings, base_rating, season_regression)

    return home_pre, away_pre, index, ratings


def add_elo_features(
    df_matchups: pd.DataFrame,
    base_rating: float = BASE_RATING,
    k: float = K,
    home_field_advantage: float = HOME_FIELD_ADVANTAGE,
    season_regression: float = SEASON_REGRESSION,
) -> pd.DataFrame:
    """
    Given a matchup-level dataframe (one row per game, home perspective),
    return a copy with columns:
        - home_elo_pre
        - away_elo_pre
        - diff_elo_pre
    Elo ratings updated game-by-game, carried across seasons with regression.
    """
    df = df_matchups.sort_values(["season", "week", "gamekey"]).copy()
    home_pre, away_pre, index, _ = _run_elo(
        df, base_rating, k, home_field_advantage, season_regression
    )
    df.loc[index, "home_elo_pre"] = home_pre
    df.loc[index, "away_elo_pre"] = away_pre
    df["diff_elo_pre"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df


def get_elo_ratings_to_week(
    df_matchups: pd.DataFrame,
    season: int,
    up_to_week: int,
    base_rating: float = BASE_RATING,
    k: float = K,
    home_field_advantage: float = HOME_FIELD_ADVANTAGE,
    season_regression: float = SEASON_REGRESSION,
) -> Dict[str, float]:
    """
    Elo ratings for every team going into (season, up_to_week), using all
    completed games before that point (including prior seasons).
    Teams missing from the result should be treated as base_rating.
    """
    df = df_matchups[df_matchups["point_diff"].notna()]
    _, _, _, ratings = _run_elo(
        df, base_rating, k, home_field_advantage, season_regression,
        stop_at=(season, up_to_week),
    )
    return ratings
