"""
Helper function that tunes hyperparameters for model. Uses grid-search to train
model on different sets of parameters. Returns parameters that give best mean
absolute error on validation folds that end before the test games. Used by
train.py for every rolling split and for the final model.
"""
import pandas as pd
from loguru import logger

from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np



def tune_xgb_hyperparams(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cutoff: tuple[int, int] | None = None,
    min_val_season: int = 2023,
) -> dict | None:
    """
    Grid-search XGBoost hyperparameters using only games played BEFORE `cutoff`
    (season, week), so the test games of a split never influence the choice.

    Folds: for each season t >= min_val_season, train on seasons < t and
    validate on season t's games that fall before the cutoff. For test season s
    (cutoff = (s, 6)) that means every earlier season in full plus weeks 1-5 of
    s. cutoff=None uses every completed game (for the final model).

    Parameters are ranked by average validation MAE across folds.
    """
    if cutoff is None:
        before = pd.Series(True, index=df.index)
    else:
        c_season, c_week = cutoff
        before = (df["season"] < c_season) | ((df["season"] == c_season) & (df["week"] < c_week))

    candidate_val_seasons = sorted(
        s for s in df.loc[before, "season"].unique() if s >= min_val_season
    )

    logger.info(f"Hyperparameter tuning (cutoff={cutoff}) across validation seasons: {candidate_val_seasons}")

    param_grid = [
        {"max_depth": 1, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 2.0, "reg_alpha": 0.0},
        {"max_depth": 1, "min_child_weight": 14, "subsample": 0.9, "colsample_bytree": 0.6, "reg_lambda": 3.0, "reg_alpha": 0.0},

        {"max_depth": 2, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 2.0, "reg_alpha": 0.2},
        {"max_depth": 2, "min_child_weight": 14, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 3.0, "reg_alpha": 0.3},

        {"max_depth": 2, "min_child_weight": 8,  "subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 2.0, "reg_alpha": 0.5},
        {"max_depth": 3, "min_child_weight": 12, "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 3.0, "reg_alpha": 0.5},
    ]


    best_params: dict | None = None
    best_score = np.inf

    X_full = df[feature_cols]
    y_full = df[target_col]

    for i, params in enumerate(param_grid, start=1):
        logger.debug(f"Testing param set {i}/{len(param_grid)}: {params}")
        fold_maes: list[float] = []

        for val_season in candidate_val_seasons:
            train_mask = df["season"] < val_season
            val_mask   = (df["season"] == val_season) & before

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_train = X_full[train_mask]
            y_train = y_full[train_mask]
            X_val   = X_full[val_mask]
            y_val   = y_full[val_mask]

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=600,
                learning_rate=0.05,
                eval_metric="rmse",
                random_state=34,
                n_jobs=1,
                **params,
            )

            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            fold_maes.append(mae)

        if not fold_maes:
            continue

        avg_mae = float(np.mean(fold_maes))
        logger.debug(f"Param set {i}: avg validation MAE across seasons = {avg_mae:.3f}")

        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params

    if best_params is None:
        logger.warning(f"No validation games before cutoff={cutoff}; caller should use default params.")
        return None

    logger.info(f"Best params (cutoff={cutoff}): {best_params} with avg val MAE={best_score:.3f}")
    return best_params