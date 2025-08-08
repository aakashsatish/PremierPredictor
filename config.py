#!/usr/bin/env python3
"""
EPL Tracker - Configuration Settings
Centralized configuration for the prediction system
"""

import os
from typing import Dict, Any

# API Configuration
SCRAPER_API_KEY = "ddfd01475e78ecc08703ba3677251cec"
FBREF_BASE_URL = "https://fbref.com"

# Model Configuration
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_depth': 15,
    'random_state': 42
}

# Feature Engineering Configuration
ROLLING_WINDOW = 3  # Number of matches for rolling averages
CONFIDENCE_THRESHOLDS = {
    'high': 0.65,
    'medium': 0.55,
    'low': 0.45
}

# Team Name Mappings
TEAM_MAPPINGS = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd", 
    "Newcastle United": "Newcastle Utd", 
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'ham Forest"
}

# File Paths
DATA_FILES = {
    'matches': 'matches.csv',
    'production_predictions': '2025_2026_production_predictions.csv',
    'improved_predictions': '2025_2026_improved_predictions.csv'
}

# Model Performance Metrics
MODEL_ACCURACY = {
    'random_forest': 0.645,  # 64.5%
    'enhanced_statistical': 0.559,  # 55.9%
    'high_confidence_threshold': 0.833  # 83.3%
}

# Prediction Configuration
PREDICTION_SETTINGS = {
    'home_advantage': 0.10,  # 10% home advantage
    'draw_probability_base': 0.15,  # Base draw probability
    'min_probability': 0.10,  # Minimum probability bounds
    'max_probability': 0.80   # Maximum probability bounds
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'epl_tracker.log'
}

# Feature Columns for Model Training
FEATURE_COLUMNS = [
    "gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga"
]

# Additional Calculated Features
CALCULATED_FEATURES = [
    "xg_diff", "goals_per_xg", "shots_accuracy"
]

# Model Predictors
MODEL_PREDICTORS = [
    "venue_code", "opp_code", "hour", "day_code"
] + [f"{col}_rolling" for col in FEATURE_COLUMNS + CALCULATED_FEATURES]

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        'api': {
            'scraper_api_key': SCRAPER_API_KEY,
            'fbref_base_url': FBREF_BASE_URL
        },
        'model': {
            'random_forest_params': RANDOM_FOREST_PARAMS,
            'accuracy': MODEL_ACCURACY,
            'predictors': MODEL_PREDICTORS
        },
        'features': {
            'rolling_window': ROLLING_WINDOW,
            'base_columns': FEATURE_COLUMNS,
            'calculated_features': CALCULATED_FEATURES
        },
        'predictions': {
            'confidence_thresholds': CONFIDENCE_THRESHOLDS,
            'settings': PREDICTION_SETTINGS
        },
        'files': DATA_FILES,
        'teams': TEAM_MAPPINGS,
        'logging': LOGGING_CONFIG
    }

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        config = get_config()
        
        # Check required files exist
        for file_type, file_path in config['files'].items():
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: {file_type} file not found: {file_path}")
        
        # Validate model parameters
        rf_params = config['model']['random_forest_params']
        if rf_params['n_estimators'] <= 0:
            raise ValueError("n_estimators must be positive")
        
        # Validate thresholds
        thresholds = config['predictions']['confidence_thresholds']
        if not (0 < thresholds['low'] < thresholds['medium'] < thresholds['high'] < 1):
            raise ValueError("Confidence thresholds must be in ascending order between 0 and 1")
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_config() 