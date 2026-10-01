"""
Model choice: XGBoost Regressor

Scales Vegas odds (scaled very low to limit impact on model), makes rolling
time-splits for training, selects features for training, and 
trains/validates/tests on rolling time-splits. Rolling time splits works as
follows: for season s, train: seasons < s, validate: s weeks 1-5, test: s weeks 6+.
Hyperparameters are determined via leak-free grid-search. 

Evaluation metrics used are mean absolute error, root mean squared error, r2, 
and win accuracy (classification). Win accuracy is determined by checking if the
predicted point differential picks the correct winner, where a positive
point differential indicates the home team is predicted to win, negative: away.
Percentage of wins correctly predicted = win accuracy.

Hyperparameters are tuned per split using only games before that split's test
weeks (tune_xgb.py). The final XGBoost model is trained on all completed games,
with params tuned on all completed games, and saved to models/.
"""
from pathlib import Path
import pickle

import pandas as pd
from loguru import logger
import typer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import numpy as np

from nfl_prediction.config import MODELS_DIR, MATCHUPS_DIR
from nfl_prediction.modeling.tune_xgb import tune_xgb_hyperparams

app = typer.Typer()

BASE_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    eval_metric="rmse",
    random_state=34,
    n_jobs=1,
)

# Used when tuning is turned off (--no-tune); the original hand-picked set.
DEFAULT_TREE_PARAMS = dict(
    max_depth=1,
    min_child_weight=14,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_lambda=3.0,
    reg_alpha=0.0,
)

VEGAS_SCALE_MAP = {
    "home_moneyline": 0.0,
    "away_moneyline": 0.0,
    "home_implied_prob": 1,
    "vegas_spread": 1,
}

def scale_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent scaling to Vegas-related columns.
    """
    for col, factor in VEGAS_SCALE_MAP.items():
        if col in df.columns:
            raw_col = f"{col}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[col]
            df[col] = df[col] * factor
        else:
            logger.warning(f"Vegas column '{col}' not found in dataframe during scaling.")
    return df

def make_rolling_splits(df: pd.DataFrame, start_season: int = 2022) -> list[dict]:
    """
    Rolling-origin train/val/test splits.
    
    For each season szn:
        - Train on seasons < szn
        - Val on szn weeks 1-5
        - Test on szn weeks 6+
    """
    
    seasons = sorted(df["season"].unique())
    
    splits = []
    
    for szn in seasons:
        if szn == start_season:
            continue
        
        train_split = df["season"] < szn
        val_split = (df["season"] == szn) & (df["week"].between(1, 5))
        test_split = (df["season"] == szn) & (df["week"] >= 6)
        
        if test_split.sum() == 0:
            # In-progress season without week 6+ yet (e.g. 2026 early on)
            continue
        
        splits.append({
            "test_season": szn,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
        })
        
    return splits

"""
def make_splits(df: pd.DataFrame):
    '''
    Time-based train/val/test splits
    
    — Train: 2022 - 2025 (week 5)
    — Validate: 2025 weeks 6-8
    — Test: 2025 weeks 9-12
    '''
    season_col = "season"
    week_col = "week"
    
    train_set = (df[season_col] < 2025) | ((df[season_col] == 2025) & df[week_col] <= 5)
    val_set = (df[season_col] == 2025) & (df[week_col].between(6, 8))
    test_set = (df[season_col] == 2025) & (df[week_col] >= 9)
    
    return train_set, val_set, test_set
"""
def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Return select features to be used in model training.
    """
    selected_features = [
        # Vegas (scaled)
        "vegas_spread",
        "home_implied_prob",

        # Elo ratings
        "home_elo_pre",
        "away_elo_pre",
        "diff_elo_pre",

        # Some standard NFL stats
        "passingattempts",
        "rushingyardsperattempt",
        "quarterbackhits",
        "opp_rolling_win_rate_5",
        "timessackedyards",
        "opponentpenaltyyards",

        # Rolling stability features
        "rolling_yards_total_3",
        "rolling_points_for_3",
        "rolling_points_against_3",
        "opp_rolling_yards_total_3",
        "opp_rolling_points_for_3",
        "opp_rolling_points_against_3",
        
        # Rolling EPA metrics (from nflfastR)
        "off_epa_per_play_rolling_3",
        "off_dropback_epa_rolling_3",
        "def_epa_per_play_rolling_3",
        "def_dropback_epa_against_rolling_3",
        "def_success_rate_rolling_3",
        "def_rush_epa_against_rolling_3",
        "off_success_rate_rolling_3",
        "off_rush_epa_rolling_3",
        "epa_off_diff_rolling_3",
        "epa_def_diff_rolling_3",
        "off_epa_per_play_rolling_5",
        "def_epa_per_play_rolling_5",
    ]
    
    final_cols = [c for c in selected_features if c in df.columns]
    
    missing = set(selected_features) - set(final_cols)
    if missing:
        print(f"[WARNING] Missing expected features: {missing}")
    
    return final_cols

@app.command()
def main(features_path: Path = MATCHUPS_DIR / "matchups_all_seasons.csv", 
         model_path: Path = MODELS_DIR / "xgb_point_diff.pkl",
         tune: bool = typer.Option(True, help="Grid-search params per split using only pre-test games")):
    logger.info(f"Loading data from {features_path}")
    df = pd.read_csv(features_path)
    
    target_col = "point_diff"
    
    df = df.dropna(subset=[target_col])
    
    df = scale_vegas_features(df)

    feature_cols = select_features(df, target_col)
    logger.info(f"Using {len(feature_cols)} features")
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df[target_col]
    
    df["next_game_win"] = (df[target_col] > 0).astype(int)
    win = df["next_game_win"]
    
    splits = make_rolling_splits(df)
    
    all_metrics = []
    
    for split in splits:
        test_season = split["test_season"]
        train_set = split["train_split"]
        val_set = split["val_split"]
        test_set = split["test_split"]
        
        X_train, y_train = X[train_set], y[train_set]
        X_val, y_val = X[val_set], y[val_set]
        X_test, y_test = X[test_set], y[test_set]
        
        win_train = win[train_set]
        win_val = win[val_set]
        win_test = win[test_set]
        
        logger.info(
            f"\n==== Rolling split: Test season {test_season} ====\n"
            f"Train: {len(X_train)} rows | Val: {len(X_val)} rows | Test: {len(X_test)} rows"
        )
    
        # Tune only on games before this season's test weeks (no test leakage)
        tree_params = (
            (tune_xgb_hyperparams(df, feature_cols, target_col, cutoff=(int(test_season), 6))
             or DEFAULT_TREE_PARAMS)
            if tune else DEFAULT_TREE_PARAMS
        )
        model = xgb.XGBRegressor(**BASE_PARAMS, **tree_params)

        logger.info(f"[Season {test_season}] Training XGBoost model...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
    
        def report_split(name: str, X_split, y_split, win_split):
            predictions = model.predict(X_split)
            mae = mean_absolute_error(y_split, predictions)
            rmse = root_mean_squared_error(y_split, predictions)
            r2 = r2_score(y_split, predictions)
            
            pred_win = (predictions > 0).astype(int)
            win_acc = (pred_win == win_split).mean()
            
            logger.info(
                f"[Season {test_season}] {name} "
                f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f} | WIN ACCURACY={win_acc:.3f}"
            )
            return mae, rmse, r2, win_acc
    
        report_split("Train", X_train, y_train, win_train)
        report_split("Val", X_val, y_val, win_val)
        mae_test, rmse_test, r2_test, winacc_test = report_split("Test",  X_test,  y_test,  win_test)
        
        preds_test = model.predict(X_test)
        model_err = preds_test - y_test
        spread_mae = np.mean(np.abs(model_err))
        spread_bias = np.mean(model_err)

        # --- Vegas comparison metrics ---
        if "vegas_spread_raw" in df.columns:
            vegas_spread_raw = df.loc[test_set, "vegas_spread_raw"].values
            vegas_err = (- vegas_spread_raw) - y_test.values

            # 1) Beat-Vegas rate
            model_abs = np.abs(model_err)
            vegas_abs = np.abs(vegas_err)
            beat_vegas = (model_abs < vegas_abs)
            beat_vegas_rate = beat_vegas.mean()

            # 2) Average edge vs Vegas (negative = better than Vegas)
            avg_edge_vs_vegas = float(np.mean(model_abs - vegas_abs))

            # 3) ATS accuracy (model vs actual cover)
            # Home covers if actual margin > -spread
            home_covers_actual = (y_test.values > -vegas_spread_raw).astype(int)
            home_covers_model  = (preds_test > -vegas_spread_raw).astype(int)
            ats_acc_model = (home_covers_actual == home_covers_model).mean()

            logger.info(
                f"[Season {test_season}] Beat-Vegas rate: {beat_vegas_rate:.3f} | "
                f"Avg edge vs Vegas (MAE diff): {avg_edge_vs_vegas:.3f} | "
                f"ATS accuracy (model): {ats_acc_model:.3f}"
            )
        else:
            beat_vegas_rate = np.nan
            avg_edge_vs_vegas = np.nan
            ats_acc_model = np.nan

        logger.info(f"[Season {test_season}] Test Spread MAE: {spread_mae:.3f}")
        logger.info(f"[Season {test_season}] Spread Bias (positive = overpredicting home): {spread_bias:.3f}\n")
        
        all_metrics.append({
            "test_season": test_season,
            "test_MAE": mae_test,
            "test_RMSE": rmse_test,
            "test_R2": r2_test,
            "test_WIN_ACC": winacc_test,
            "test_Spread_MAE": spread_mae,
            "test_Spread_Bias": spread_bias,
            "Beat_Vegas_Rate": beat_vegas_rate,
            "Edge_vs_Vegas_MAE": avg_edge_vs_vegas,
            "ATS_ACC_Model": ats_acc_model,
        })
        
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        result_cols = [
            "season",
            "week",
            "team",
            "opponent",
            "home",
            "vegas_spread",
            "home_implied_prob",
        ]

        result_cols = [c for c in result_cols if c in df.columns]

        # ---------- TEST RESULTS ----------
        test_base = df.loc[test_set, result_cols].copy()

        test_base["y_true"] = y_test.values
        test_base["y_pred"] = model.predict(X_test)
        test_base["residual"] = test_base["y_pred"] - test_base["y_true"]
        test_base["abs_residual"] = test_base["residual"].abs()

        test_base.to_csv(reports_dir / f"predictions_test_{test_season}.csv", index=False)

        # ---------- TRAIN RESULTS ----------
        train_base = df.loc[train_set, result_cols].copy()

        train_base["y_true"] = y_train.values
        train_base["y_pred"] = model.predict(X_train)
        train_base["residual"] = train_base["y_pred"] - train_base["y_true"]
        train_base["abs_residual"] = train_base["residual"].abs()

        train_base.to_csv(reports_dir / f"predictions_train_{test_season}.csv", index=False)
        
        logger.info(f"Saved predictions for season {test_season}\n")
    

    # Final model is trained exactly like the evaluated ones: all completed
    # games, with params tuned the same way (now using every completed game).
    X_final = X
    y_final = y
    
    final_params = (
        (tune_xgb_hyperparams(df, feature_cols, target_col, cutoff=None) or DEFAULT_TREE_PARAMS)
        if tune else DEFAULT_TREE_PARAMS
    )
    final_model = xgb.XGBRegressor(**BASE_PARAMS, **final_params)
        
    logger.info(f"Training FINAL model on all completed games...")
    final_model.fit(X_final, y_final, verbose=False)
        
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump({"model": final_model, "features": feature_cols, "params": final_params}, file)
    
    logger.success(f"Saved FINAL trained model to {model_path}")

if __name__ == "__main__":
    app()