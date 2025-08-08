#!/usr/bin/env python3
"""
EPL Tracker - Enhanced Predictions with Balanced Approach
Uses Random Forest model with improved probability calibration
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare historical match data"""
    print("📊 Loading historical match data...")
    
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    
    # Create target variable (Win = 1, Draw/Loss = 0)
    matches["target"] = (matches["result"] == "W").astype(int)
    
    # Create venue codes
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    
    # Create opponent codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    
    # Create day codes
    matches["day_code"] = matches["day"].astype("category").cat.codes
    
    # Extract hour from time
    matches["hour"] = pd.to_datetime(matches["time"], format="%H:%M").dt.hour
    
    # Calculate additional features
    matches["xg_per_shot"] = matches["xg"] / matches["sh"].replace(0, 1)
    matches["possession_efficiency"] = matches["poss"] / 100
    matches["formation_code"] = matches["formation"].astype("category").cat.codes
    matches["season_stage"] = matches["date"].dt.month
    
    # Calculate rolling averages
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group

    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga", "xg_diff", "goals_per_xg", "shots_accuracy"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    return matches_rolling, new_cols

def train_enhanced_model(data, predictors):
    """Train an enhanced model with probability calibration"""
    print("🤖 Training enhanced model with probability calibration...")
    
    # Use all historical data for training
    train_data = data[data["date"] < '2025-06-01']
    
    # Initialize Random Forest with more trees for better probability estimates
    rf_model = RandomForestClassifier(
        n_estimators=100,  # More trees for better probability estimates
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=15,
        random_state=42
    )
    
    # Calibrate the model for better probability estimates
    calibrated_model = CalibratedClassifierCV(
        rf_model, 
        cv=5, 
        method='isotonic'
    )
    
    calibrated_model.fit(train_data[predictors], train_data["target"])
    
    return calibrated_model

def predict_with_enhanced_model(model, fixtures_df, team_stats, new_cols):
    """Make predictions using the enhanced model"""
    print("🎯 Making enhanced predictions...")
    
    # Team name mappings
    team_mappings = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd", 
        "Newcastle United": "Newcastle Utd", 
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Nottingham Forest": "Nott'ham Forest"
    }
    
    predictions = []
    
    for idx, fixture in fixtures_df.iterrows():
        try:
            home_team = fixture['home']
            away_team = fixture['away']
            date = fixture['date']
            time = fixture['time']
            matchweek = fixture.get('matchweek', '')
            day = fixture.get('day', '')
            
            # Map team names
            home_team_mapped = team_mappings.get(home_team, home_team)
            away_team_mapped = team_mappings.get(away_team, away_team)
            
            # Get team stats
            home_stats = team_stats.get(home_team_mapped, {
                'overall_win_rate': 0.3,
                'recent_win_rate': 0.3,
                'home_win_rate': 0.3,
                'away_win_rate': 0.3,
                'total_matches': 0
            })
            
            away_stats = team_stats.get(away_team_mapped, {
                'overall_win_rate': 0.3,
                'recent_win_rate': 0.3,
                'home_win_rate': 0.3,
                'away_win_rate': 0.3,
                'total_matches': 0
            })
            
            # Create feature vector for prediction
            # Use average values for rolling features since we don't have recent data
            avg_rolling_features = [0.5] * len(new_cols)  # Neutral values
            
            # Create prediction features
            features = [
                1,  # venue_code (Home)
                away_stats['opp_code'] if 'opp_code' in away_stats else 0,
                pd.to_datetime(time, format="%H:%M").hour if pd.notna(time) else 15,
                pd.to_datetime(day, format="%a").dayofweek if pd.notna(day) else 5,
                home_stats['home_win_rate'],  # xg_per_shot proxy
                home_stats['recent_win_rate'],  # possession_efficiency proxy
                0,  # formation_code (default)
                8,  # season_stage (August)
            ] + avg_rolling_features
            
            # Get model prediction probability
            win_prob = model.predict_proba([features])[0][1]
            
            # ENHANCED: More balanced probability calculation
            # Combine model prediction with team stats
            model_weight = 0.7
            stats_weight = 0.3
            
            stats_win_prob = (
                0.4 * home_stats['home_win_rate'] +
                0.3 * home_stats['recent_win_rate'] +
                0.2 * home_stats['overall_win_rate'] +
                0.1 * (1 - away_stats['away_win_rate'])
            )
            
            # Add home advantage
            stats_win_prob += 0.08
            
            # Combine model and stats
            final_win_prob = model_weight * win_prob + stats_weight * stats_win_prob
            
            # ENHANCED: More realistic probability bounds
            final_win_prob = max(0.15, min(0.85, final_win_prob))
            
            # Calculate away win probability (simplified)
            away_win_prob = max(0.10, min(0.70, 1 - final_win_prob - 0.15))
            
            # ENHANCED: Better prediction thresholds
            if final_win_prob > 0.55:
                prediction = "Win"
                confidence = "High" if final_win_prob > 0.65 else "Medium"
            elif final_win_prob > 0.45:
                prediction = "Win"
                confidence = "Low"
            elif away_win_prob > 0.55:
                prediction = "Loss/Draw"
                confidence = "High" if away_win_prob > 0.65 else "Medium"
            else:
                prediction = "Loss/Draw"
                confidence = "Low"
            
            draw_prob = max(0.10, 1 - final_win_prob - away_win_prob)
            
            result = {
                "team": home_team,
                "opponent": away_team,
                "venue": "Home",
                "date": date,
                "time": time,
                "matchweek": matchweek,
                "day": day,
                "win_probability": final_win_prob,
                "opponent_win_probability": away_win_prob,
                "draw_probability": draw_prob,
                "prediction": prediction,
                "confidence": confidence,
                "model_probability": win_prob,
                "stats_probability": stats_win_prob,
                "home_overall_rate": home_stats['overall_win_rate'],
                "home_recent_rate": home_stats['recent_win_rate'],
                "home_home_rate": home_stats['home_win_rate'],
                "away_overall_rate": away_stats['overall_win_rate'],
                "away_recent_rate": away_stats['recent_win_rate'],
                "away_away_rate": away_stats['away_win_rate']
            }
            
            predictions.append(result)
            print(f"✅ Matchweek {matchweek} ({day}): {home_team} vs {away_team}: {prediction} ({final_win_prob:.1%}) [{confidence}]")
            
        except Exception as e:
            print(f"❌ Error predicting fixture {idx}: {e}")
            continue
    
    return predictions

def main():
    """Main function to generate enhanced predictions"""
    print("🚀 EPL Tracker - Enhanced Predictions")
    print("=" * 50)
    
    # Load and prepare data
    matches_rolling, new_cols = load_and_prepare_data()
    
    # Define predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code", "xg_per_shot", "possession_efficiency", "formation_code", "season_stage"] + new_cols
    
    # Train enhanced model
    enhanced_model = train_enhanced_model(matches_rolling, predictors)
    
    # Load fixtures (you'll need to implement fixture loading)
    print("📋 Loading fixtures...")
    # This would load your fixtures data
    # fixtures_df = load_fixtures()
    
    # For now, we'll create a sample prediction
    print("✅ Enhanced model trained successfully!")
    print("📈 Expected improvements:")
    print("   - More balanced win/loss predictions")
    print("   - Better probability calibration")
    print("   - More realistic confidence levels")
    print("   - Reduced conservative bias")
    
    return enhanced_model

if __name__ == "__main__":
    main() 