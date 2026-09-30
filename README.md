# NFL Game Outcome Predictive Modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## NFL Game Outcome Prediction Pipeline 2026
Predict NFL game point differentials for upcoming NFL weeks in the 2026 NFL season. 
Model uses weekly team data, play-by-play analytics, Vegas odds, and other 
engineered features. 

This project implements an end-to-end machine learning pipeline following the
Cookiecutter Data Science structure. As of 2026 the pipeline runs entirely on
free nflverse data (no API key). 2022-2025 box scores were originally pulled
from SportsDataIO and are kept as-is; 2026 box scores, all odds from 2026 on,
schedules, and EPA come from nflverse via `download_nflverse.py`. Every model
input rebuilt from nflverse matches the 2025 SportsDataIO data on 99.6-100% of
team-games.

## Project Statement
NFL games are unpredictable——game outcomes are affected by a host of factors, 
including, but not limited to, team-strength, opponent matchups, injuries, 
strategy, and a lot of randomness.
### Goal:
Build a data-driven model that predicts NFL game outcomes via game point 
differentials. Specifically, this project aims to predict future games in the
2025 season, using historical and 2025 season data. The objective is to predict
the outcome of any future NFL game with >55% accuracy.

## Datasets
| Data Type | Source | Size | Notes |
|----------:|:------:|:----:|:-----:|
| Weekly NFL data (2026+, 2025 wk 18) | nflverse via nflreadpy | 32 teams x weeks | Free; `download_nflverse.py` |
| Vegas odds + schedules (2026+) | nflverse via nflreadpy | 1 row per game | Free; `vegas_spread = -spread_line` |
| Weekly NFL data (2022-2025) | SportsDataIO (https://discoverylab.sportsdata.io/) | Records: 2024; Columns: 159 | API key required |
| Play-by-play EPA metrics | nflfastR (https://nflfastr.com/) | Records: 2019; Columns: 18 | Used to derive matchup strength features and other advanced metrics |
| Vegas Odds | SportsDataIO (https://discoverylab.sportsdata.io/) | Records: 1160; Columns: 10 | API key required |
| Current Injury Report | CBS Sports (https://www.cbssports.com/nfl/injuries/) | Records: 452; Columns: 4 | Only used for predicting future outcomes, not used in training model |
| Current Week NFL Schedule | SportsDataIO (https://discoverylab.sportsdata.io/) | Records: 15; Columns: 5 | API key required; only used for predicting games in upcoming week |

## Technologies Used
- Python 3.11
- Typer: CLI tooling for some pipeline steps
- pandas, numpy: data engineering
- XGBoost: predictive model used
- scikit-learn: model metrics
- nflfastR, nflreadr: EPA data source
- requests: pull API data
- loguru: structured logging
- matplotlib, seaborn: visualizations

## Modeling Methods
- XGBoost regression
- Rolling time-series train/validation/test splits
- Hyperparamter tuning using grid-search
- Feature importance for feature selection, 31 features used
- Final model tested on all available data

## Setup

### API Key Required
This project uses the SportsData Discovery Lab API to download:
- Game statistics
- Vegas Odds
- Weekly NFL schedules
To reproduce the **data collection pipeline**, you must register for an API key.

Store API key in a `.env` file.

## Installation
Follow these steps to set up the project locally:

### 1. Clone repository
```bash
git clone https://github.com/ryansteele9/NFL_Predict_Repo_INST414.git
cd NFL_Predict_Repo_INST414
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
```

Activate it:

Mac/Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
All required packages are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. API Key (required for downloading data)
Store key in a `.env` file:
```bash
SPORTSDATAIO_API_KEY=your_key_here
```

### 5. Running the pipeline (no API key required)
If you do not have API key, you can still run:
- Feature Engineering
- Model Training/Evaluation

All processed datasets required for feature engineering and model 
training/evaluation are included in `data/processed`.

This allows full reproducibility of ML pipeline without an API key.

### 6. Optional: Download data if you have API key:
```bash
python -m nfl_prediction.data.download_team_stats
python -m nfl_prediction.data.download_odds
```
These build raw datasets using live API endpoints.

## Usage Examples
Add engineered features to season-team files:
```bash
python -m nfl_prediction.data.feature_engineering_team
```
Build full matchup dataset across all seasons (2022-2025):
```bash
python -m nfl_prediction.data.build_full_matchup_data
```
Train XGBoost Model:
```bash
python -m nfl_prediction.modeling.train
```
None of the steps in the pipeline require parameters.

## Full Pipeline (2026, no API key)
1. download_nflverse.py --season 2026 (box scores, odds, schedules, EPA)
2. clean_team_stats_by_team.py
3. feature_engineering_team.py
4. build_matchup_data.py
5. build_full_matchup_data.py
6. train.py
7. predict.py --season 2026 --week N

Seasons are set in `config.SEASONS`. Each week: re-run steps 1-7.

The original SportsDataIO steps (download_team_stats.py, download_odds.py,
clean_team_stats.py, nflfastr_build_advanced_stats.R) are only needed to rebuild
2022-2025 from raw SDIO data.

### 2026 changes
- Elo ratings now carry over between seasons, regressed 1/3 toward 1500
  (`team_ratings.SEASON_REGRESSION`; set to 1.0 for the old full reset).
- Fixed: predict.py used Elo 1500 for every team.
- Fixed: the upcoming-week feature row skipped each team's most recent game.
- Fixed: `home` column was always 0 ("Home" vs "HOME").

## Results
### Test split results for seasons 2023-2025
| season | MAE | RMSE | R² | Win Accuracy |
|:------:|:---:|:----:|:--:|:------------:|
| 2023 | 9.935 | 12.436 | 0.139 | 62.9% |
| 2024 | 9.607 | 12.493 | 0.286 | 72.2% |
| 2025 | 9.902 | 11.987 | 0.231 | 66.7% |

Uses rolling time-series splits, where for season s:
- Train on seasons < s
- Validate on s weeks 1-5
- Test on s weeks 6+

**Conclusion:**
The model correctly predicts the winner of a given game in 2025 at ~66.7% 
accuracy. This is signifigantly higher than the target of >55%. Factors that
signifigantly improved model performance inlude EPA-based efficiency metrics,
Elo-ratings for each team, and rolling point differentials.

### Test split results compared to Vegas predictions
| season | Beat-Vegas % | Avg. Edge vs Vegas (MAE diff) | ATS Acc % (model) |
|:------:|:---------------:|:-----------------------------:|:---------------:|
| 2023 | 41.8% | 0.657 | 51.5% |
| 2024 | 50.5% | 0.190 | 56.2% |
| 2025 | 47.9% | 0.287 | 52.1% |

Test splits for Vegas evaluation metrics.
- Beat-Vegas %: Percentage of test point differentials with smaller error than 
    vegas spread
- Avg. Edge vs. Vegas: Difference in mean absolute error between test point
    differentials and vegas spreads (negative = better than Vegas)
- ATS Acc % (model): Percentage of predicted point differentials that correctly
    predict against the spread outcome. (e.g. if model predicts BAL to win by 
    10 and Vegas spread has BAL at -4.5, model correctly predicts ATS outcome)

**Conclusion:**
Model predictions are slightly worse than Vegas spreads. 2023 is signifigantly
worse, most likely due to rolling time-series splits, where 2023 would have
least amount of training data. 2024 is the best by a good margin. 2025 is 
slightly worse, most likely due to either there being less testing data (season
not over yet) and/or increased variation in 2025 outcomes.


## Project Organization

```
my_repo2/
├── README.md
├── .env                     ← Stores API keys (SportsData.io), NOT tracked by git
├── .gitignore               ← Ignores raw data, virtual env, etc.
│
├── data/                             ← All data used/generated by pipeline
|   ├── external/                     ← External raw data
|   |    ├── injuries/                ← Raw injuries csv
|   |    └── nflfastr/                ← EPA metrics csv
|   |
|   ├── processed/                    ← Data processed by pipeline
|   |    ├── clean_team_stats_season/ ← Cleaned season data
|   |    ├── features/                ← Season-team matchups csv with EPA, Elo, Rolling features
|   |    ├── matchups/                ← Season files with all matchups from season
|   |    ├── injuries/                ← Processed Injuries
|   |    ├── teams/                   ← Season-team matchups csv
|   |    └── odds/                    ← Processed Odds
|   |
|   └── raw/                          ← Raw data
|        ├── odds/                    ← Weekly vegas odds
|        ├── sdio_json/               ← Raw JSON from SportsDataIO
|        ├── team_stats_csv/          ← Raw CSV from SportsDataIO
|        └── schedules/               ← NFL week schedules 
|
├── nfl_prediction/          ← Main project code
│   ├── config.py            ← API keys and paths
│   │
│   ├── data/                               ← All data used/generated by pipeline
│   |   ├── download_team_stats.py          ← Downloads raw weekly stats from SportsDataIO
│   |   ├── clean_team_stats.py             ← Cleans API responses into structured CSVs
│   |   ├── clean_team_stats_by_team.py     ← Splits by team + season
|   |   ├── nflfastr_build_advanced_stats.R ← Builds EPA metrics csv
|   |   ├── nflfastr_epa.py                 ← Helper function to load EPA metrics csv
│   |   ├── feature_engineering_team.py     ← Adds rolling averages & cumulative features
│   |   ├── build_team_dataset.py           ← Combines all weekly CSVs into season-level files
│   |   ├── build_matchup_data.py           ← Merges opponent stats into matchup rows
|   |   ├── download_odds.py                ← Downloads vegas odds data from SportsDataIO
|   |   ├── team_ratings.py                 ← Helper functions to compute and add Elos to DataFrame
|   |   ├── build_full_matchup_data.py      ← Merges all matchup data files into one for EDA
|   |   ├── download_cbs_injury_report.py   ← Pulls injury report from CBS Sports
|   |   ├── build_injury_adjustment_file.py ← Processes raw injury report file
|   |   └── injury_adjust.py                ← Helper functions to compute and apply injury weights
|   |
|   └── modeling/
|       ├── train.py    ← Makes splits and trains model
|       ├── tune_xgb.py ← Hyperparameter tuning function
|       └── predict.py  ← Creates matchups for upcoming week and makes point spread predictions
│
├── notebooks/
|   ├── multivariate_aalysis.ipynb      ← EDA: Multivariate Analysis
│   ├── univariate_analysis.ipynb       ← EDA: Univariate Analysis
|   ├── verify_odds.ipynb               ← Debug vegas odds
|   ├── cbs_injury_report.ipynb         ← Debug injury report
|   ├── baseline_model.ipynb            ← Baseline models for comparison
|   ├── model_comparison.ipynb          ← Comparing XGBoost vs. RandomForest
|   ├── feature_importance.ipynb        ← Feature importance to XGBoost model
|   ├── model_2025_predict_check.ipynb  ← Model next-week prediction performance
|   └── regression_diagnostics.ipynb    ← Residual/Errors plots & distributions
│
├── reports/                          ← Final reports, EDA summaries
|   ├── variable_inventory.py         ← Creates CSV file with variable information
|   ├── variable_inventory.csv        ← CSV file with variable information
|   ├── predictions_test*.csv         ← prediction test set data
|   └── predictions_train*.csv        ← prediction train set data
|
├── models/                   ← Trained models
|   └── xgb_point_diff.pkl    ← Trained XGBoost model for point differential
├── requirements.txt          ← Python dependencies
├── pyproject.toml
└── venv/                     ← Virtual environment (ignored by Git)
```

--------
