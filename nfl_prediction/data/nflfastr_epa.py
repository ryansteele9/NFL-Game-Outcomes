"""
Helper function that loads EPA metrics from nflfastR. Function implemented in 
feature_engineering_team.py.
"""
import pandas as pd

from nfl_prediction.config import EXTERNAL_DIR

def load_nflfastr_team_epa() -> pd.DataFrame:
    """
    Load team-game EPA metrics exported from nflfastR.
    """
    new_path = EXTERNAL_DIR / "nflfastr" / "team_game_advanced.csv"   # download_nflverse.py
    old_path = EXTERNAL_DIR / "nflfastr" / "team_game_advanced_2022_2025.csv"  # R script
    NFLFASTR_EPA_PATH = new_path if new_path.exists() else old_path
    df = pd.read_csv(NFLFASTR_EPA_PATH)
    
    df["season"] =df["season"].astype(int)
    df["week"] =df["week"].astype(int)
    
    team_map = {
        "LA": "LAR"
    }
    
    df["team"] = df["team"].replace(team_map)
    
    return df