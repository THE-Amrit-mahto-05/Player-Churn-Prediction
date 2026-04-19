import pandas as pd
import joblib
import os

# Load the list of expected columns trained by the ML model
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))
try:
    expected_columns = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
except Exception:
    expected_columns = []

def validate_and_prepare_player_data(player_dict: dict) -> dict:
    """
    Validates a player dictionary, fills missing values with safe defaults, 
    and ensures numerical types are correct.
    """
    safe_player = player_dict.copy()
    
    # Safe defaults
    defaults = {
        'PlayTimeHours': 0.0,
        'SessionsPerWeek': 0,
        'AvgSessionDurationMinutes': 0.0,
        'PlayerLevel': 1,
        'AchievementsUnlocked': 0,
        'InGamePurchases': 0,
        'GameGenre': 'Unknown',
        'GameDifficulty': 'Medium'
    }
    
    # Fill missing or null values
    for key, default_val in defaults.items():
        if key not in safe_player or pd.isna(safe_player[key]):
            safe_player[key] = default_val
            
    # Guarantee type casting
    try:
        safe_player['PlayTimeHours'] = float(safe_player['PlayTimeHours'])
        safe_player['SessionsPerWeek'] = int(safe_player['SessionsPerWeek'])
        safe_player['AvgSessionDurationMinutes'] = float(safe_player['AvgSessionDurationMinutes'])
        safe_player['PlayerLevel'] = int(safe_player['PlayerLevel'])
        safe_player['AchievementsUnlocked'] = int(safe_player['AchievementsUnlocked'])
        safe_player['InGamePurchases'] = int(safe_player['InGamePurchases'])
    except ValueError:
        pass # Fallback if data is too garbled
        
    return safe_player

def dict_to_ml_dataframe(player_dict: dict) -> pd.DataFrame:
    """
    Converts a single player dict into a pandas DataFrame 
    with perfectly aligned columns for the Logistic Regression model.
    """
    df = pd.DataFrame([player_dict])
    
    # Apply one-hot encoding exactly as the training script did
    df = pd.get_dummies(df, drop_first=True)
    
    # Add any missing expected columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
            
    # Ensure correct column order
    if expected_columns:
        df = df[expected_columns]
        
    return df
