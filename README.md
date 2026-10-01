# NFL Game Outcome Predictive Modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicts NFL game point differentials (home margin) for upcoming weeks of the
2026 NFL season with an XGBoost regression model. The model uses weekly team box
scores, play-by-play EPA efficiency metrics, Elo ratings and Vegas odds.

This started as a course project for INST414 at the University of Maryland
(2025 season). So that instructors could reproduce the results, the processed
datasets were originally committed to the repo, and the raw data came from a paid
API (SportsDataIO). As of the 2026 season, the whole pipeline runs on
**free, public nflverse data**. No API key is needed and no data is stored in
the repo: a fresh clone rebuilds every dataset from 2022 to the present,
retrains the model and predicts the upcoming week.

## Project Statement
NFL games are unpredictable. Outcomes depend on team strength, matchups,
injuries, strategy and a lot of randomness.

### Goal
Build a data-driven model that predicts NFL game outcomes through point
differentials, trained on past seasons and updated weekly during the current
season. The original target was to pick the winner of future games with >55%
accuracy.

## Datasets
All data is pulled at runtime and written to the gitignored `data/` folders.

| Data | Source | Granularity | Used for |
|:-----|:-------|:------------|:---------|
| Schedules, final scores, stadiums | nflverse schedules (`nflreadpy.load_schedules`) | 1 row per game | Scores/target, schedule for upcoming week |
| Vegas odds (spread, total, moneylines) | nflverse schedules | 1 row per game | `vegas_spread`, `home_implied_prob` features |
| Weekly team box scores | nflverse team stats (`nflreadpy.load_team_stats`) | 1 row per team-game | Box-score and rolling features |
| Play-by-play (EPA, success rate, downs, fumbles) | nflverse play-by-play / nflfastR (`nflreadpy.load_pbp`) | 1 row per play, aggregated to team-game | EPA features, third/fourth downs, first downs, fumbles lost |
| Current injury report (optional) | CBS Sports (https://www.cbssports.com/nfl/injuries/) | 1 row per player | Post-prediction injury adjustment only; not used in training |

Only regular-season games are used.

### A note on the original SportsDataIO data
The 2022–2025 box scores and odds were originally downloaded from SportsDataIO.
`download_nflverse.py` rebuilds the same columns in the same format. Its output
was checked against the SportsDataIO data for every model input:

| Season | Model inputs matching SportsDataIO exactly |
|:------:|:------------------------------------------:|
| 2022 | 97.2–100% of team-games |
| 2023 | 98.9–100% |
| 2024 | 99.6–100% |
| 2025 | 99.6–100% |

The rest of the ~160 SportsDataIO box-score columns aren't used by the model and
are left empty. There is one other difference: SportsDataIO odds were averaged
across sportsbooks, while nflverse provides a single line per game. Spreads agree
within 1 point for 98% of games. To rerun the check:
`python -m nfl_prediction.data.download_nflverse --validate 2025` (needs the old
SportsDataIO-based files locally).

## Technologies Used
- Python 3.11+
- nflreadpy (nflverse): free NFL data (schedules, odds, team stats, play-by-play)
- pandas, numpy: data engineering
- XGBoost: model
- scikit-learn: evaluation metrics
- Typer: command-line options for training and prediction
- loguru: logging
- matplotlib, seaborn: visualizations (notebooks)

## Modeling Methods
- **Target:** home-team point differential (one row per game, home perspective)
- **Model:** XGBoost regressor (600 trees, depth 1, learning rate 0.05). The
  hyperparameters were chosen by grid search (`tune_xgb.py`) and are fixed in
  `train.py`.
- **Features (29):** Vegas spread and home implied probability; Elo ratings
  (home, away, difference); the prior game's passing attempts, rushing yards per
  attempt, QB hits, sack yards and opponent penalty yards; 3-game rolling points
  for/against and yards for both teams; the opponent's 5-game win rate; 3- and
  5-game rolling EPA metrics (offense, defense, dropback, rush, success rate) and
  offense/defense EPA differentials. The feature list is the result of
  importance-based selection.
- **No leakage:** every stat feature is shifted so that a row only sees games
  played before it. Each team's upcoming week is a "dummy" row holding stats
  through its most recent game.
- **Elo:** updated game by game with a margin-of-victory multiplier. Ratings
  carry over between seasons, pulled 1/3 of the way back toward 1500 each
  offseason (`team_ratings.SEASON_REGRESSION`).
- **Validation:** rolling-origin splits. For test season *s*, the model trains
  on seasons < *s*, validates on weeks 1–5 of *s* and tests on weeks 6+ of *s*.
- **Final model:** trained on all completed games from 2024 onward.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ryansteele9/NFL-Game-Outcomes.git
cd NFL-Game-Outcomes
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

No API key or `.env` file is needed.

## Pipeline

Every step is run from the repo root as a module (`python -m ...`). The
seasons are set once in `nfl_prediction/config.py` (`SEASONS`); add the
new year there at the start of each season.

### First run (fresh clone): build everything from scratch
```bash
python -m nfl_prediction.data.download_nflverse --all
python -m nfl_prediction.data.clean_team_stats_by_team
python -m nfl_prediction.data.feature_engineering_team
python -m nfl_prediction.data.build_matchup_data
python -m nfl_prediction.data.build_full_matchup_data
python -m nfl_prediction.modeling.train
python -m nfl_prediction.modeling.predict --season 2026 --week 4
```
This takes a few minutes, mostly for downloading play-by-play data.

### Weekly update (during the season)
After the week's games finish (nflverse usually updates within a day), run the
same steps without `--all`. That downloads only the current season, and
existing seasons are left untouched:
```bash
python -m nfl_prediction.data.download_nflverse
python -m nfl_prediction.data.clean_team_stats_by_team
python -m nfl_prediction.data.feature_engineering_team
python -m nfl_prediction.data.build_matchup_data
python -m nfl_prediction.data.build_full_matchup_data
python -m nfl_prediction.modeling.train
python -m nfl_prediction.modeling.predict --season 2026 --week <next week>
```

### Steps, inputs and parameters

| # | Script | What it does | Parameters | Output |
|:-:|:-------|:-------------|:-----------|:-------|
| 1 | `data/download_nflverse.py` | Downloads schedules, odds, team stats and play-by-play; builds team-game box scores, weekly odds files, weekly schedules and team-game EPA | `--all` (every season in `config.SEASONS`)<br>`--season YEAR` (repeatable; default: latest season)<br>`--epa-seasons YEAR ...` (default: all)<br>`--overwrite-odds` (refresh past seasons' odds; current season always refreshes)<br>`--validate YEAR` (compare with SportsDataIO data; saves nothing) | `data/processed/clean_team_stats_season/`, `data/processed/odds/`, `data/raw/odds/`, `data/raw/schedules/`, `data/external/nflfastr/team_game_advanced.csv` |
| 2 | `data/clean_team_stats_by_team.py` | Splits each season file into one file per team | none | `data/processed/teams/<season>/` |
| 3 | `data/feature_engineering_team.py` | Merges EPA; adds lagged rolling/cumulative features; adds each team's upcoming-week dummy row | none | `data/processed/features/<season>/` |
| 4 | `data/build_matchup_data.py` | Joins each team to its opponent (one row per game, home perspective); adds strength and EPA differentials | none | `data/processed/matchups/matchups_<season>.csv` |
| 5 | `data/build_full_matchup_data.py` | Stacks all seasons, drops unplayed games, adds Elo and Vegas odds | none | `data/processed/matchups/matchups_all_seasons.csv` |
| 6 | `modeling/train.py` | Rolling-split evaluation, then trains and saves the final model | `--features-path` (default: `matchups_all_seasons.csv`)<br>`--model-path` (default: `models/xgb_point_diff.pkl`) | `models/xgb_point_diff.pkl`, `reports/predictions_{train,test}_<season>.csv` |
| 7 | `modeling/predict.py` | Builds matchup rows for an upcoming week from the dummy rows, current Elo and odds; predicts; applies injury adjustments if an injury file exists | `--season` (required)<br>`--week` (required)<br>`--season-type` (default `REG`)<br>`--model-path`<br>`--home-team`, `--away-team`, `--game-id` (filters)<br>`--save-matchups` (writes the rows to `data/processed/matchups/`) | Printed table of predicted margins, winners and win probabilities |

### Optional: injury adjustments
Injuries only adjust predictions after the fact; they aren't model features.
This part is semi-manual:
1. `python -m nfl_prediction.data.download_cbs_injury_report` scrapes the
   current CBS report to `data/external/injuries/CBS_injuries.csv`.
2. Filter it by hand to starters (see `notebooks/cbs_injury_report.ipynb`)
   and save it as `data/processed/injuries/cbs_injuries_cleaned_week<NN>.csv`.
3. `python -m nfl_prediction.data.build_injury_adjustment_file` writes
   `injuries_week<NN>_curated.csv`. The week number is currently hard-coded in
   the script, so edit it first.
4. `predict.py` picks up `injuries_week<NN>_curated.csv` automatically when
   it exists.

### Optional: hyperparameter tuning
`modeling/tune_xgb.py` runs a grid search over rolling season splits. To
re-tune, uncomment the `tune_xgb_hyperparams` call in `train.py` and pass
`**best_params` to the model.

### Legacy SportsDataIO scripts
`download_team_stats.py`, `download_odds.py`, `clean_team_stats.py` and
`nflfastr_build_advanced_stats.R` are the original paid-API and R versions of
step 1. They are kept for reference only, need a `SPORTSDATAIO_API_KEY`, and
aren't part of the pipeline.

## Results
All results come from a fresh clone running the free pipeline above (nflverse
data). Test sets are weeks 6–18 of each season (194 games).

### Test-split results
| Test season | MAE | RMSE | R² | Model win accuracy | Vegas favorite win accuracy |
|:-----------:|:---:|:----:|:--:|:------------------:|:---------------------------:|
| 2023 | 10.00 | 12.52 | 0.127 | 60.3% | 69.6% |
| 2024 | 9.50 | 12.25 | 0.314 | 71.6% | 75.3% |
| 2025 | 10.92 | 13.12 | 0.150 | 60.8% | 64.9% |
| 2026 wk 1–3* | 11.27 | — | — | 60.4% | — |

\*2026 so far (48 games) comes from a model trained on 2022–2025 and scored on
the games played to date, before any week 6+ test set exists.

### Compared with Vegas
| Test season | Beat-Vegas % | Avg. edge vs Vegas (MAE diff) | ATS accuracy (model) |
|:-----------:|:------------:|:-----------------------------:|:--------------------:|
| 2023 | 42.3% | +0.70 | 52.6% |
| 2024 | 52.6% | +0.08 | 57.7% |
| 2025 | 37.1% | +1.05 | 43.3% |

- **Beat-Vegas %:** share of test games where the model's predicted margin was
  closer to the actual result than the Vegas spread was.
- **Avg. edge vs Vegas:** model MAE minus Vegas MAE (negative = better than Vegas).
- **ATS accuracy:** share of games where the model picked the right side of
  the spread (e.g. the model has BAL by 10, Vegas has BAL −4.5, and BAL wins by 7).

### Conclusions
- The model clears the original 55% target every season (60–72%) but doesn't
  beat simply picking the Vegas favorite, which won 65–75% of the same games.
- Its margins are less accurate than the Vegas spread in every season
  (positive edge), and ATS accuracy is roughly a coin flip. 2024 was the one
  season close to Vegas.
- The Vegas features (implied probability and spread) account for over half of
  the final model's total gain. Most of what the model knows comes from the
  market, which explains why it trails Vegas: it's largely re-learning the line.
  After those, the most important features are sack yards, rolling points,
  Elo and dropback EPA.

### Changes from the original (2025) version
- Data source switched from SportsDataIO to nflverse (see the validation table above).
- Fixed: `predict.py` gave every team an Elo of 1500 at prediction time.
- Fixed: each team's upcoming-week feature row left out its most recent game.
- Elo now carries over between seasons with regression to the mean instead of
  resetting to 1500.
- The earlier README reported 66.7% win accuracy for 2025. That number came from
  a partial season; the full-season figure is about 61%.

## Project Organization

```
NFL-Game-Outcomes/
├── README.md
├── LICENSE
├── Makefile                 ← Cookiecutter convenience commands
├── pyproject.toml
├── requirements.txt
├── .gitignore               ← Ignores all of data/, venv, .env, etc.
│
├── data/                                 ← Generated by the pipeline (gitignored)
│   ├── external/
│   │   ├── nflfastr/                     ← team_game_advanced.csv (team-game EPA)
│   │   └── injuries/                     ← Raw CBS injury report
│   ├── raw/
│   │   ├── odds/                         ← Weekly odds (copy of processed)
│   │   └── schedules/                    ← Weekly schedules
│   └── processed/
│       ├── clean_team_stats_season/      ← One team-game box score file per season
│       ├── teams/<season>/               ← One file per team-season
│       ├── features/<season>/            ← Team files with EPA, rolling and dummy-week rows
│       ├── matchups/                     ← Per-season and all-season matchup files
│       ├── odds/                         ← Weekly Vegas odds
│       └── injuries/                     ← Curated injury files (optional)
│
├── nfl_prediction/                       ← Main project code
│   ├── config.py                         ← Paths and SEASONS
│   ├── data/
│   │   ├── download_nflverse.py          ← Step 1: downloads and builds all raw inputs (free)
│   │   ├── clean_team_stats_by_team.py   ← Step 2: splits season files by team
│   │   ├── nflfastr_epa.py               ← Helper: loads the team-game EPA file
│   │   ├── feature_engineering_team.py   ← Step 3: rolling, cumulative and EPA features
│   │   ├── build_matchup_data.py         ← Step 4: team + opponent matchup rows
│   │   ├── team_ratings.py               ← Helper: Elo ratings
│   │   ├── build_full_matchup_data.py    ← Step 5: all seasons + Elo + odds
│   │   ├── download_cbs_injury_report.py ← Optional: scrapes CBS injury report
│   │   ├── build_injury_adjustment_file.py ← Optional: curates injury file
│   │   ├── injury_adjust.py              ← Helper: injury weights and adjustments
│   │   ├── download_team_stats.py        ← Legacy (SportsDataIO)
│   │   ├── download_odds.py              ← Legacy (SportsDataIO)
│   │   ├── clean_team_stats.py           ← Legacy (SportsDataIO)
│   │   └── nflfastr_build_advanced_stats.R ← Legacy (R version of EPA build)
│   └── modeling/
│       ├── train.py                      ← Step 6: rolling evaluation + final model
│       ├── tune_xgb.py                   ← Grid-search hyperparameter tuning
│       └── predict.py                    ← Step 7: predicts an upcoming week
│
├── models/
│   └── xgb_point_diff.pkl                ← Trained model + feature list
│
├── notebooks/
│   ├── univariate_analysis.ipynb         ← EDA: univariate
│   ├── multivariate_analysis.ipynb       ← EDA: multivariate
│   ├── baseline_model.ipynb              ← Baseline models for comparison
│   ├── model_comparison.ipynb            ← XGBoost vs. RandomForest
│   ├── feature_importance.ipynb          ← Feature importance / selection
│   ├── regression_diagnostics.ipynb      ← Residuals and error distributions
│   ├── model_2025_predict_check.ipynb    ← Next-week prediction performance (2025)
│   ├── verify_odds.ipynb                 ← Odds debugging
│   └── cbs_injury_report.ipynb           ← Injury report filtering
│
├── reports/
│   ├── predictions_test_<season>.csv     ← Test-split predictions (from train.py)
│   ├── predictions_train_<season>.csv    ← Train-split predictions (from train.py)
│   ├── variable_inventory.py             ← Builds variable inventory (original SDIO data)
│   └── variable_inventory.csv
│
├── docs/                                 ← MkDocs scaffold (Cookiecutter)
└── tests/                                ← Test scaffold (Cookiecutter)
```
