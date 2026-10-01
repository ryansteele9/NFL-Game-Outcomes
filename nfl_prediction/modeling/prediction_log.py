"""
Live prediction log: a running, tracked record of every pre-game prediction,
so the current season is a true out-of-sample test.

predict.py appends each week's predictions here automatically, together with
the Vegas line at the moment of prediction and a model fingerprint. After the
games, `update` fills in final scores and the closing line, and `summary`
reports season-to-date accuracy against Vegas.

Rules that keep the record honest:
    - Rows are only ever appended; earlier predictions are never overwritten.
    - Each row stores the time it was logged and the game's kickoff time.
      Rows logged at or after kickoff are flagged (post_kickoff = True) and
      ignored by `summary`.
    - If a game was predicted more than once before kickoff, `summary` scores
      the LAST pre-kickoff prediction (the one you'd actually have used).

Log file: reports/live/predictions_log.csv (tracked in git, unlike data/).

Usage:
    python -m nfl_prediction.modeling.prediction_log update
    python -m nfl_prediction.modeling.prediction_log summary [--season 2026]
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer
from loguru import logger

from nfl_prediction.config import PROJ_ROOT

LOG_PATH = PROJ_ROOT / "reports" / "live" / "predictions_log.csv"
EASTERN = ZoneInfo("America/New_York")

LOG_COLUMNS = [
    # when / what
    "logged_at", "kickoff", "post_kickoff", "model_version",
    "season", "week", "game_id", "home_team", "away_team",
    # market at prediction time
    "vegas_spread", "vegas_total", "home_implied_prob",
    # prediction
    "pred_home_margin", "pred_home_margin_inj_adj", "pred_winner", "pred_home_win_prob",
    "ats_pick",
    # filled in by `update`
    "home_score", "away_score", "actual_home_margin", "closing_spread",
]

app = typer.Typer(help="Live prediction log: record, score and summarize predictions.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_eastern() -> datetime:
    return datetime.now(EASTERN)


def model_fingerprint(model_path: Path) -> str:
    """Short hash of the model file, so you can tell which model made a pick."""
    try:
        return hashlib.sha1(Path(model_path).read_bytes()).hexdigest()[:10]
    except OSError:
        return "unknown"


def _load_schedule(seasons: list[int]) -> pd.DataFrame:
    from nfl_prediction.data.download_nflverse import load_schedule
    return load_schedule(seasons)


def _kickoffs(schedule: pd.DataFrame) -> pd.Series:
    """Kickoff timestamps (Eastern) indexed by game_id. nflverse times are ET."""
    ts = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("00:00").astype(str),
        errors="coerce",
    ).dt.tz_localize(EASTERN)
    return pd.Series(ts.array, index=schedule["game_id"].values)


def read_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)


def _write_log(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[LOG_COLUMNS].to_csv(LOG_PATH, index=False)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def log_predictions(df_pred: pd.DataFrame, season: int, week: int, model_path: Path) -> pd.DataFrame:
    """Append predictions from predict.py to the log. Returns the new rows."""
    now = _now_eastern()

    try:
        kickoff = _kickoffs(_load_schedule([season]))
    except Exception as exc:  # network trouble shouldn't block predicting
        logger.warning(f"Couldn't load kickoff times ({exc}); post_kickoff check skipped.")
        kickoff = pd.Series(dtype="datetime64[ns, America/New_York]")

    pred = df_pred["pred_point_diff"].astype(float)
    spread = df_pred["vegas_spread"].astype(float) if "vegas_spread" in df_pred else np.nan
    model_minus_vegas = pred + spread  # vegas home margin = -spread

    rows = pd.DataFrame({
        "logged_at": now.isoformat(timespec="seconds"),
        "kickoff": df_pred["game_id"].map(lambda g: kickoff.get(g, pd.NaT)),
        "model_version": model_fingerprint(model_path),
        "season": season,
        "week": week,
        "game_id": df_pred["game_id"].values,
        "home_team": df_pred["home_team"].values,
        "away_team": df_pred["away_team"].values,
        "vegas_spread": spread,
        "vegas_total": df_pred.get("vegas_total", np.nan),
        "home_implied_prob": df_pred.get("home_implied_prob", np.nan),
        "pred_home_margin": pred.round(2),
        "pred_home_margin_inj_adj": df_pred.get("point_diff_adj", pd.Series(np.nan, index=df_pred.index)).astype(float).round(2),
        "pred_winner": np.where(pred > 0, df_pred["home_team"], df_pred["away_team"]),
        "pred_home_win_prob": df_pred["pred_home_win_prob"].astype(float).round(3),
        "ats_pick": np.where(model_minus_vegas > 0, df_pred["home_team"],
                             np.where(model_minus_vegas < 0, df_pred["away_team"], "none")),
    })
    rows["post_kickoff"] = rows["kickoff"].map(lambda k: bool(pd.notna(k) and now >= k))
    rows["kickoff"] = rows["kickoff"].map(lambda k: k.isoformat() if pd.notna(k) else "")
    for col in ["home_score", "away_score", "actual_home_margin", "closing_spread"]:
        rows[col] = np.nan

    late = int(rows["post_kickoff"].sum())
    if late:
        logger.warning(f"{late} game(s) already kicked off; logged but excluded from scoring.")

    _write_log(pd.concat([read_log(), rows[LOG_COLUMNS]], ignore_index=True))
    logger.info(f"Logged {len(rows)} predictions to {LOG_PATH}")
    return rows


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def update_results() -> pd.DataFrame:
    """Fill in final scores and closing lines for every finished game in the log."""
    log = read_log()
    if log.empty:
        logger.warning("Prediction log is empty.")
        return log

    sched = _load_schedule(sorted(log["season"].astype(int).unique()))
    final = sched[sched["home_score"].notna()].set_index("game_id")

    done = log["game_id"].isin(final.index)
    ids = log.loc[done, "game_id"]
    log.loc[done, "home_score"] = ids.map(final["home_score"]).values
    log.loc[done, "away_score"] = ids.map(final["away_score"]).values
    log.loc[done, "actual_home_margin"] = (log.loc[done, "home_score"] - log.loc[done, "away_score"]).values
    log.loc[done, "closing_spread"] = (-ids.map(final["spread_line"])).values

    _write_log(log)
    logger.info(f"Results filled for {int(done.sum())} logged rows ({ids.nunique()} games).")
    return log


def scored_predictions(log: pd.DataFrame, season: Optional[int] = None) -> pd.DataFrame:
    """Last pre-kickoff prediction per finished game, with per-game scoring columns."""
    df = log[(~log["post_kickoff"].astype(bool)) & log["actual_home_margin"].notna()].copy()
    if season is not None:
        df = df[df["season"] == season]
    df = df.sort_values("logged_at").groupby("game_id", as_index=False).tail(1)

    actual = df["actual_home_margin"].astype(float)
    pred = df["pred_home_margin"].astype(float)
    spread = df["vegas_spread"].astype(float)
    closing = df["closing_spread"].astype(float)

    df["su_correct"] = np.where(actual == 0, np.nan, ((pred > 0) == (actual > 0)).astype(float))
    df["abs_err"] = (pred - actual).abs()
    df["vegas_abs_err"] = (-spread - actual).abs()
    # ATS vs the line you had at prediction time; pushes and no-pick games excluded
    cover = actual + spread
    pick = pred + spread
    df["ats_correct"] = np.where((cover == 0) | (pick == 0), np.nan, ((pick > 0) == (cover > 0)).astype(float))
    # ATS vs the closing line (the tougher, standard benchmark)
    cover_c = actual + closing
    pick_c = pred + closing
    df["ats_close_correct"] = np.where((cover_c == 0) | (pick_c == 0), np.nan, ((pick_c > 0) == (cover_c > 0)).astype(float))
    # Did the line move toward the model after it was logged? (closing line value)
    df["clv_points"] = np.sign(pick) * (spread - closing)
    return df


def summarize(log: pd.DataFrame, season: Optional[int] = None) -> pd.DataFrame:
    df = scored_predictions(log, season)
    if df.empty:
        return pd.DataFrame()

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "games": len(g),
            "su_acc": g["su_correct"].mean(),
            "mae": g["abs_err"].mean(),
            "vegas_mae": g["vegas_abs_err"].mean(),
            "ats_acc": g["ats_correct"].mean(),
            "ats_acc_close": g["ats_close_correct"].mean(),
            "avg_clv": g["clv_points"].mean(),
        })

    by_week = df.groupby(["season", "week"]).apply(agg, include_groups=False)
    total = agg(df).to_frame().T
    total.index = pd.MultiIndex.from_tuples([("all", "all")], names=["season", "week"])
    return pd.concat([by_week, total])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def update():
    """Fill in final scores and closing lines for finished games."""
    update_results()


@app.command()
def summary(season: Optional[int] = typer.Option(None, help="Limit to one season")):
    """Season-to-date accuracy of logged pre-kickoff predictions."""
    table = summarize(read_log(), season)
    if table.empty:
        print("No finished games with pre-kickoff predictions yet. Run `update` after games finish.")
        return
    fmt = table.copy()
    for col in ["su_acc", "ats_acc", "ats_acc_close"]:
        fmt[col] = (fmt[col] * 100).round(1).astype(str) + "%"
    for col in ["mae", "vegas_mae", "avg_clv"]:
        fmt[col] = fmt[col].astype(float).round(2)
    fmt["games"] = fmt["games"].astype(int)
    print()
    print(fmt.to_string())
    print()
    print("su_acc: picked the winner | mae / vegas_mae: avg miss in points (model / Vegas line at prediction)")
    print("ats_acc: right side of the line you had | ats_acc_close: vs the closing line")
    print("avg_clv: points the line moved toward the model's side after logging (positive = good)")


if __name__ == "__main__":
    app()
