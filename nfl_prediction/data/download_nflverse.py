"""
Free replacement for the SportsDataIO download steps (download_team_stats.py,
download_odds.py, the SDIO schedule call in predict.py) and for
nflfastr_build_advanced_stats.R. Everything comes from nflverse via nflreadpy.

Builds, for the requested seasons:
    1. Team-game box scores in the *cleaned* SportsDataIO format
       (same lowercase column names as clean_team_stats.py output), upserted into
       processed/clean_team_stats_season/clean_team_stats_{season}.csv.
       Existing SDIO rows are kept; only (week, team) rows that are missing are
       added, so running this for 2025 just fills in week 18.
    2. Weekly odds files in the download_odds.py format, saved to
       raw/odds/ and processed/odds/ (only weeks that have lines posted).
       Existing odds files are left alone unless --overwrite-odds is passed.
    3. Weekly schedules (for predict.py) saved to raw/schedules/.
    4. Team-game EPA metrics (Python port of nflfastr_build_advanced_stats.R),
       saved to external/nflfastr/team_game_advanced.csv for all seasons given
       to --epa-seasons.

Only box-score columns that could be validated against the 2025 SDIO data are
filled; the rest of the SDIO schema is left empty for nflverse rows. Every
column the model uses matches the 2025 SDIO data exactly (see
validate_against_sdio()).

Conventions (checked against 2025):
    - team code "LA" (nflverse) -> "LAR" (SDIO)
    - vegas_spread = -spread_line (nflverse spread_line is positive when the home
      team is favored; SDIO HomePointSpread is negative when home is favored)
    - gamekey = f"{season}1{week:02d}{home SDIO team id:02d}", same as SDIO

Usage:
    python -m nfl_prediction.data.download_nflverse --season 2026
    python -m nfl_prediction.data.download_nflverse --season 2025 --season 2026
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import nflreadpy as nfl

from nfl_prediction.config import (
    CLEAN_STATS_DIR,
    EXTERNAL_DIR,
    ODDS_PROC_DIR,
    ODDS_RAW_DIR,
    RAW_DIR,
    SEASONS,
)

SCHEDULES_DIR = RAW_DIR / "schedules"
EPA_PATH = EXTERNAL_DIR / "nflfastr" / "team_game_advanced.csv"

TEAM_MAP = {"LA": "LAR"}

# SportsDataIO TeamIDs, used to build SDIO-style gamekeys
SDIO_TEAM_IDS = {
    "ARI": 1, "ATL": 2, "BAL": 3, "BUF": 4, "CAR": 5, "CHI": 6, "CIN": 7,
    "CLE": 8, "DAL": 9, "DEN": 10, "DET": 11, "GB": 12, "HOU": 13, "IND": 14,
    "JAX": 15, "KC": 16, "MIA": 19, "MIN": 20, "NE": 21, "NO": 22, "NYG": 23,
    "NYJ": 24, "LV": 25, "PHI": 26, "PIT": 28, "LAC": 29, "SEA": 30, "SF": 31,
    "LAR": 32, "TB": 33, "TEN": 34, "WAS": 35,
}

PBP_COLS = [
    "season", "season_type", "week", "game_id", "posteam", "defteam", "epa",
    "pass", "rush", "qb_scramble", "first_down_rush", "first_down_pass",
    "first_down_penalty", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed", "fumble_lost", "fumbled_1_team",
]


def _to_pandas(df) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)


def _map_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_MAP)
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_schedule(seasons: list[int]) -> pd.DataFrame:
    sched = _to_pandas(nfl.load_schedules(seasons))
    sched = sched[sched["game_type"] == "REG"].copy()
    return _map_teams(sched, ["home_team", "away_team"])


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    pbp = nfl.load_pbp(seasons)
    pbp = pbp.select([c for c in PBP_COLS if c in pbp.columns])
    pbp = _to_pandas(pbp)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    return _map_teams(pbp, ["posteam", "defteam", "fumbled_1_team"])


def load_team_week_stats(seasons: list[int]) -> pd.DataFrame:
    ts = _to_pandas(nfl.load_team_stats(seasons, summary_level="week"))
    ts = ts[ts["season_type"] == "REG"].copy()
    return _map_teams(ts, ["team", "opponent_team"])


# ---------------------------------------------------------------------------
# Box scores
# ---------------------------------------------------------------------------
def _team_side_stats(ts: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with the team's own box-score stats."""
    out = pd.DataFrame({
        "season": ts["season"].astype(int),
        "week": ts["week"].astype(int),
        "team": ts["team"],
        "passingattempts": ts["attempts"],
        "passingcompletions": ts["completions"],
        "passingyards": ts["passing_yards"] + ts["sack_yards_lost"],  # net of sacks
        "passingtouchdowns": ts["passing_tds"],
        "passinginterceptions": ts["passing_interceptions"],
        "rushingattempts": ts["carries"],
        "rushingyards": ts["rushing_yards"],
        "rushingtouchdowns": ts["rushing_tds"],
        "timessacked": ts["sacks_suffered"],
        "timessackedyards": -ts["sack_yards_lost"],  # nflverse stores this as negative
        "quarterbackhits": ts["def_qb_hits"],         # SDIO: hits BY this team's defense
        "sacks": ts["def_sacks"],
        "tacklesforloss": ts["def_tackles_for_loss"],
        "passesdefended": ts["def_pass_defended"],
        "fumblesforced": ts["def_fumbles_forced"],
        "penalties": ts["penalties"],
        "penaltyyards": ts["penalty_yards"],
        "fieldgoalattempts": ts["fg_att"],
        "fieldgoalsmade": ts["fg_made"],
    })
    out["rushingyardsperattempt"] = (
        out["rushingyards"] / out["rushingattempts"].replace(0, np.nan)
    ).round(1)
    out["offensiveyards"] = out["passingyards"] + out["rushingyards"]

    # Down / first-down counts come from play-by-play
    downs = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(
            firstdownsbyrushing=("first_down_rush", "sum"),
            firstdownsbypassing=("first_down_pass", "sum"),
            firstdownsbypenalty=("first_down_penalty", "sum"),
            thirddownconversions=("third_down_converted", "sum"),
            third_failed=("third_down_failed", "sum"),
            fourthdownconversions=("fourth_down_converted", "sum"),
            fourth_failed=("fourth_down_failed", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    downs["firstdowns"] = downs[
        ["firstdownsbyrushing", "firstdownsbypassing", "firstdownsbypenalty"]
    ].sum(axis=1)
    downs["thirddownattempts"] = downs["thirddownconversions"] + downs.pop("third_failed")
    downs["fourthdownattempts"] = downs["fourthdownconversions"] + downs.pop("fourth_failed")
    downs[["season", "week"]] = downs[["season", "week"]].astype(int)

    # Fumbles lost from play-by-play (team stats miss special-teams fumbles)
    fumbles = (
        pbp[pbp["fumble_lost"] == 1]
        .groupby(["season", "week", "fumbled_1_team"]).size()
        .rename("fumbleslost").reset_index()
        .rename(columns={"fumbled_1_team": "team"})
    )
    fumbles[["season", "week"]] = fumbles[["season", "week"]].astype(int)

    out = out.merge(downs, on=["season", "week", "team"], how="left")
    out = out.merge(fumbles, on=["season", "week", "team"], how="left")
    out["fumbleslost"] = out["fumbleslost"].fillna(0).astype(int)
    out["giveaways"] = out["passinginterceptions"] + out["fumbleslost"]
    return out


def build_team_games(season: int, sched: pd.DataFrame, ts: pd.DataFrame,
                     pbp: pd.DataFrame) -> pd.DataFrame:
    """Cleaned SDIO-format team-game rows for all completed games in a season."""
    games = sched[(sched["season"] == season) & sched["home_score"].notna()].copy()
    if games.empty:
        return pd.DataFrame()

    side = _team_side_stats(ts[ts["season"] == season], pbp[pbp["season"] == season])

    rows = []
    for _, g in games.iterrows():
        gamekey = int(f"{season}1{int(g['week']):02d}{SDIO_TEAM_IDS[g['home_team']]:02d}")
        date = f"{g['gameday']}T{g['gametime']}:00" if pd.notna(g["gametime"]) else g["gameday"]
        for team, opp, is_home in [(g["home_team"], g["away_team"], True),
                                   (g["away_team"], g["home_team"], False)]:
            pf = g["home_score"] if is_home else g["away_score"]
            pa = g["away_score"] if is_home else g["home_score"]
            rows.append({
                "gamekey": gamekey,
                "date": date,
                "seasontype": 1,
                "season": season,
                "week": int(g["week"]),
                "team": team,
                "opponent": opp,
                "home_away": "HOME" if is_home else "AWAY",
                "points_for": float(pf),
                "points_against": float(pa),
                "totalscore": float(pf + pa),
                "stadium": g.get("stadium"),
                "dayofweek": g.get("weekday"),
                "teamid": SDIO_TEAM_IDS[team],
                "opponentid": SDIO_TEAM_IDS[opp],
                "nflverse_game_id": g["game_id"],
            })
    df = pd.DataFrame(rows)

    df = df.merge(side, on=["season", "week", "team"], how="left")

    # Mirror opponent's stats as opponent* columns (SDIO naming)
    stat_cols = [c for c in side.columns if c not in ("season", "week", "team")]
    opp = side.rename(columns={c: f"opponent{c}" for c in stat_cols})
    opp = opp.rename(columns={"team": "opponent"})
    df = df.merge(opp, on=["season", "week", "opponent"], how="left")

    df["takeaways"] = df["opponentgiveaways"]
    df["turnoverdifferential"] = df["takeaways"] - df["giveaways"]

    # Same derived columns as clean_team_stats.py
    df["point_diff"] = df["points_for"] - df["points_against"]
    df["win"] = (df["point_diff"] > 0).astype(int)
    df["turnover_diff"] = df["takeaways"] - df["giveaways"]
    df["third_down_pct"] = df["thirddownconversions"] / df["thirddownattempts"].replace(0, 1)
    df["home"] = (df["home_away"] == "HOME").astype(int)

    return df.sort_values(["week", "gamekey", "home_away"], ascending=[True, True, False])


def upsert_clean_season(season: int, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Add nflverse rows for (week, team) pairs not already in the clean file."""
    CLEAN_STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = CLEAN_STATS_DIR / f"clean_team_stats_{season}.csv"

    if path.exists():
        existing = pd.read_csv(path)
        have = set(zip(existing["week"], existing["team"]))
        add = new_rows[[(w, t) not in have for w, t in zip(new_rows["week"], new_rows["team"])]]
        combined = pd.concat([existing, add], ignore_index=True)
        print(f"[{season}] {len(existing)} existing rows kept, {len(add)} nflverse rows added")
    else:
        combined = new_rows
        print(f"[{season}] created with {len(combined)} nflverse rows")

    combined = combined.sort_values(["week", "gamekey"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Odds + schedules
# ---------------------------------------------------------------------------
def _american_to_prob(odds: pd.Series) -> pd.Series:
    odds = odds.astype(float)
    return np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0))


def build_odds(season: int, sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[(sched["season"] == season) & sched["spread_line"].notna()].copy()
    p_home = _american_to_prob(s["home_moneyline"])
    p_away = _american_to_prob(s["away_moneyline"])
    return pd.DataFrame({
        "season": season,
        "week": s["week"].astype(int),
        "game_id": s["game_id"],
        "home_team": s["home_team"],
        "away_team": s["away_team"],
        "vegas_spread": -s["spread_line"],
        "vegas_total": s["total_line"],
        "home_moneyline": s["home_moneyline"],
        "away_moneyline": s["away_moneyline"],
        "home_implied_prob": p_home / (p_home + p_away),
    })


def save_odds(odds: pd.DataFrame, overwrite: bool = False) -> None:
    ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    ODDS_PROC_DIR.mkdir(parents=True, exist_ok=True)
    for (season, week), wk in odds.groupby(["season", "week"]):
        name = f"odds_{season}_week{week:02d}.csv"
        if (ODDS_PROC_DIR / name).exists() and not overwrite:
            continue
        wk.to_csv(ODDS_RAW_DIR / name, index=False)
        wk.to_csv(ODDS_PROC_DIR / name, index=False)
        print(f"Saved odds: {name} ({len(wk)} games)")


def save_schedules(season: int, sched: pd.DataFrame) -> None:
    SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)
    s = sched[sched["season"] == season]
    for week, wk in s.groupby("week"):
        out = pd.DataFrame({
            "season": season,
            "week": int(week),
            "game_id": wk["game_id"],
            "home_team": wk["home_team"],
            "away_team": wk["away_team"],
            "gameday": wk["gameday"],
        })
        out.to_csv(SCHEDULES_DIR / f"schedule_{season}REG_{int(week):02d}.csv", index=False)


# ---------------------------------------------------------------------------
# EPA (port of nflfastr_build_advanced_stats.R)
# ---------------------------------------------------------------------------
def build_team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["epa"].notna()].copy()
    p["dropback"] = (p["pass"] == 1) | (p["qb_scramble"] == 1)
    p["is_rush"] = p["rush"] == 1

    def summarize(side: str, prefix: str, against: str) -> pd.DataFrame:
        d = p[p[side].notna()]
        keys = ["season", "week", side, "game_id"]
        base = d.groupby(keys).agg(
            plays=("epa", "size"),
            epa_per_play=("epa", "mean"),
            success_rate=("epa", lambda x: (x > 0).mean()),
        )
        db = d[d["dropback"]].groupby(keys)["epa"].agg(["size", "mean"])
        ru = d[d["is_rush"]].groupby(keys)["epa"].agg(["size", "mean"])
        out = base.join(db.add_prefix("db_"), how="left").join(ru.add_prefix("ru_"), how="left")
        out = out.reset_index().rename(columns={side: "team"})
        return pd.DataFrame({
            "season": out["season"], "week": out["week"], "team": out["team"],
            "game_id": out["game_id"],
            f"{prefix}_plays": out["plays"],
            f"{prefix}_epa_per_play": out["epa_per_play"],
            f"{prefix}_success_rate": out["success_rate"],
            f"{prefix}_dropbacks{against}": out["db_size"].fillna(0).astype(int),
            f"{prefix}_dropback_epa{against}": out["db_mean"],
            f"{prefix}_rushes{against}": out["ru_size"].fillna(0).astype(int),
            f"{prefix}_rush_epa{against}": out["ru_mean"],
        })

    off = summarize("posteam", "off", "")
    dfn = summarize("defteam", "def", "_against")
    # R script required >1 rush for offense (kept for consistency)
    off.loc[off["off_rushes"] <= 1, "off_rush_epa"] = np.nan

    epa = off.merge(dfn, on=["season", "week", "team", "game_id"], how="outer")
    epa[["season", "week"]] = epa[["season", "week"]].astype(int)
    return epa.sort_values(["season", "week", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
MODEL_INPUT_COLS = [
    "points_for", "points_against", "passingattempts", "rushingyardsperattempt",
    "quarterbackhits", "timessackedyards", "opponentpenaltyyards", "offensiveyards",
    "opponentoffensiveyards", "turnover_diff", "third_down_pct",
]


def validate_against_sdio(season: int, nv: pd.DataFrame) -> pd.DataFrame:
    """Compare nflverse-built rows with the SDIO clean file for a season."""
    sdio = pd.read_csv(CLEAN_STATS_DIR / f"clean_team_stats_{season}.csv")
    m = sdio.merge(nv, on=["week", "team"], suffixes=("_sdio", "_nv"))
    rows = []
    for col in [c for c in nv.columns if f"{c}_sdio" in m.columns]:
        a, b = m[f"{col}_sdio"], m[f"{col}_nv"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            continue
        diff = (a - b).abs()
        rows.append({
            "column": col,
            "model_input": col in MODEL_INPUT_COLS,
            "exact_match": float((diff < 0.0015).mean()),
            "mae": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
        })
    return pd.DataFrame(rows).sort_values(["model_input", "exact_match"], ascending=[False, True])


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, action="append",
                        help="Season(s) to download box scores/odds for (default: latest in config.SEASONS)")
    parser.add_argument("--epa-seasons", type=int, nargs="*", default=None,
                        help="Seasons for the EPA file (default: all of config.SEASONS)")
    parser.add_argument("--overwrite-odds", action="store_true",
                        help="Overwrite existing processed odds files")
    parser.add_argument("--validate", type=int, default=None,
                        help="Instead of saving, compare nflverse box scores with SDIO for this season")
    args = parser.parse_args()

    if args.validate:
        s = args.validate
        nv = build_team_games(s, load_schedule([s]), load_team_week_stats([s]), load_pbp([s]))
        report = validate_against_sdio(s, nv)
        print(report.to_string(index=False))
        return

    seasons = args.season or [max(SEASONS)]
    epa_seasons = args.epa_seasons or list(SEASONS)

    sched = load_schedule(seasons)
    ts = load_team_week_stats(seasons)
    pbp_all = load_pbp(sorted(set(seasons) | set(epa_seasons)))

    for season in seasons:
        rows = build_team_games(season, sched, ts, pbp_all)
        if not rows.empty:
            upsert_clean_season(season, rows)
        save_odds(build_odds(season, sched), overwrite=args.overwrite_odds)
        save_schedules(season, sched)

    epa = build_team_game_epa(pbp_all[pbp_all["season"].isin(epa_seasons)])
    EPA_PATH.parent.mkdir(parents=True, exist_ok=True)
    epa.to_csv(EPA_PATH, index=False)
    print(f"Saved EPA for seasons {epa_seasons} to {EPA_PATH} ({len(epa)} team-games)")


if __name__ == "__main__":
    main()
