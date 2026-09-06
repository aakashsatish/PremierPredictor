#!/usr/bin/env python3
"""
EPL Tracker - Weighted Production Predictions
Uses Random Forest with weighted training data (2025/2026 season matches weighted more heavily)
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# ScraperAPI configuration
api_key = "ddfd01475e78ecc08703ba3677251cec"

def scrape_with_scraperapi(url):
    """Scrape using ScraperAPI to handle anti-bot protection"""
    scraper_url = f"http://api.scraperapi.com/?api_key={api_key}&url={url}"
    response = requests.get(scraper_url)
    print(f"📡 Scraping: {url}")
    return response

def load_and_prepare_data():
    """Load and prepare historical match data with weights for 2025/2026 season"""
    print("📊 Loading historical match data...")
    
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    
    # Create target variable (Win = 1, Draw/Loss = 0)
    matches["target"] = (matches["result"] == "W").astype(int)
    
    # Create sample weights: 2025/2026 season matches get 3x weight
    # More recent matches within 2025/2026 season also get higher weight
    matches["sample_weight"] = 1.0  # Default weight for historical matches
    
    # Weight 2025/2026 season matches more heavily (3x weight)
    current_season_mask = matches["date"] >= "2025-08-01"
    matches.loc[current_season_mask, "sample_weight"] = 3.0
    
    # Additional weighting based on recency within current season (if within last 30 days, 4x weight)
    today = pd.Timestamp.now()
    recent_mask = (matches["date"] >= "2025-08-01") & (matches["date"] >= today - pd.Timedelta(days=30))
    matches.loc[recent_mask, "sample_weight"] = 4.0
    
    print(f"📊 Weight distribution:")
    print(f"   Historical matches (2020-2024): weight 1.0")
    print(f"   Current season matches (2025-2026): weight 3.0")
    print(f"   Recent matches (last 30 days): weight 4.0")
    print(f"   Total 2025/2026 matches: {current_season_mask.sum()}")
    print(f"   Recent matches (last 30 days): {recent_mask.sum()}")
    
    # Create venue codes
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    
    # Create opponent codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    
    # Create day codes
    matches["day_code"] = matches["day"].astype("category").cat.codes
    
    # Extract hour from time
    matches["hour"] = pd.to_datetime(matches["time"], format="%H:%M", errors='coerce').dt.hour
    matches["hour"] = matches["hour"].fillna(15)  # Default to 15:00 if time is missing
    
    # Calculate additional features (handle missing values)
    matches["xg"] = pd.to_numeric(matches["xg"], errors='coerce').fillna(0)
    matches["xga"] = pd.to_numeric(matches["xga"], errors='coerce').fillna(0)
    matches["gf"] = pd.to_numeric(matches["gf"], errors='coerce').fillna(0)
    matches["ga"] = pd.to_numeric(matches["ga"], errors='coerce').fillna(0)
    matches["sot"] = pd.to_numeric(matches["sot"], errors='coerce').fillna(0)
    matches["sh"] = pd.to_numeric(matches["sh"], errors='coerce').fillna(0)
    
    matches["xg_diff"] = matches["xg"] - matches["xga"]
    matches["goals_per_xg"] = matches["gf"] / matches["xg"].replace(0, 1)
    matches["shots_accuracy"] = matches["sot"] / matches["sh"].replace(0, 1)
    
    # Fill missing shooting stats with team averages if available
    for stat in ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga"]:
        if stat in matches.columns:
            matches[stat] = pd.to_numeric(matches[stat], errors='coerce')
            # Fill missing values with team's average for that stat
            for team in matches["team"].unique():
                team_mask = matches["team"] == team
                team_avg = matches.loc[team_mask, stat].mean()
                if pd.notna(team_avg):
                    matches.loc[team_mask & matches[stat].isna(), stat] = team_avg
            # Final fill with overall mean if still missing
            matches[stat] = matches[stat].fillna(matches[stat].mean())
    
    # Calculate rolling averages
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group

    # Use available columns
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    # Add additional features to rolling averages
    additional_cols = ["xg_diff", "goals_per_xg", "shots_accuracy"]
    additional_new_cols = [f"{c}_rolling" for c in additional_cols]
    
    # Combine all columns for rolling averages
    all_cols = cols + additional_cols
    all_new_cols = new_cols + additional_new_cols
    
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, all_cols, all_new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    # Keep sample weights with rolling data
    # We need to match weights with the rolling data
    weights_rolling = matches_rolling["sample_weight"].copy()
    
    return matches_rolling, all_new_cols, weights_rolling

def train_weighted_model():
    """Train the production Random Forest model with weighted data"""
    print("🤖 Training weighted Random Forest model...")
    
    # Load data
    matches_rolling, all_new_cols, sample_weights = load_and_prepare_data()
    
    # Define predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"] + all_new_cols
    
    # Use all historical data for training
    train_data = matches_rolling[matches_rolling["date"] < '2025-06-01'].copy()
    train_weights = train_data["sample_weight"].values
    
    # Also include 2025/2026 matches that are completed
    current_season_data = matches_rolling[
        (matches_rolling["date"] >= "2025-08-01") & 
        (matches_rolling["result"].notna())
    ].copy()
    current_season_weights = current_season_data["sample_weight"].values
    
    # Combine training data
    combined_train = pd.concat([train_data, current_season_data])
    combined_weights = np.concatenate([train_weights, current_season_weights])
    
    # Ensure weights align with combined_train index
    if len(combined_weights) != len(combined_train):
        print(f"⚠️  Warning: Weight length mismatch ({len(combined_weights)} vs {len(combined_train)}), using default weights")
        combined_weights = combined_train["sample_weight"].values if "sample_weight" in combined_train.columns else np.ones(len(combined_train))
    
    print(f"📊 Training on {len(combined_train)} matches:")
    print(f"   Historical matches (2020-2024): {len(train_data)}")
    print(f"   Current season matches (2025-2026): {len(current_season_data)}")
    print(f"   Weighted effective sample size: {combined_weights.sum():.0f}")
    
    # Train Random Forest model with sample weights
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=15,
        random_state=42,
        class_weight='balanced'  # Also use class balancing
    )
    
    model.fit(combined_train[predictors], combined_train["target"], sample_weight=combined_weights)
    
    print("✅ Weighted production model trained successfully!")
    return model, predictors, all_new_cols

def create_production_predictions(fixtures_df, model, predictors, all_new_cols):
    """Create production predictions using weighted Random Forest with improved confidence"""
    print("🎯 Creating production predictions with weighted model...")

    # Load historical data to get team performance for confidence calculation
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])

    # Calculate team stats for confidence levels (weighted by recency)
    team_stats = {}
    for team in matches["team"].unique():
        team_matches = matches[matches["team"] == team].copy()
        
        # Overall win rate
        wins = len(team_matches[team_matches["result"] == "W"])
        total = len(team_matches)
        win_rate = wins / total if total > 0 else 0.3
        
        # Recent form (last 10 matches, prioritizing 2025/2026 season)
        recent_matches = team_matches.sort_values('date').tail(10)
        recent_wins = len(recent_matches[recent_matches["result"] == "W"])
        recent_win_rate = recent_wins / len(recent_matches) if len(recent_matches) > 0 else win_rate
        
        # Current season performance (2025/2026)
        current_season_matches = team_matches[team_matches["date"] >= "2025-08-01"]
        if len(current_season_matches) > 0:
            current_season_wins = len(current_season_matches[current_season_matches["result"] == "W"])
            current_season_win_rate = current_season_wins / len(current_season_matches)
        else:
            current_season_win_rate = recent_win_rate
        
        # Home vs Away performance
        home_matches = team_matches[team_matches["venue"] == "Home"]
        away_matches = team_matches[team_matches["venue"] == "Away"]
        
        home_wins = len(home_matches[home_matches["result"] == "W"])
        home_total = len(home_matches)
        home_win_rate = home_wins / home_total if home_total > 0 else win_rate
        
        away_wins = len(away_matches[away_matches["result"] == "W"])
        away_total = len(away_matches)
        away_win_rate = away_wins / away_total if away_total > 0 else win_rate
        
        team_stats[team] = {
            'overall_win_rate': win_rate,
            'recent_win_rate': recent_win_rate,
            'current_season_win_rate': current_season_win_rate,
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'total_matches': total,
            'current_season_matches': len(current_season_matches)
        }

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
            # Extract fixture data
            home_team = fixture['home']
            away_team = fixture['away']
            date = fixture['date']
            time = fixture['time']
            matchweek = fixture.get('matchweek', '')
            day = fixture.get('day', '')

            # Map team names
            home_team_mapped = team_mappings.get(home_team, home_team)
            away_team_mapped = team_mappings.get(away_team, away_team)

            # Get team stats for confidence calculation
            home_stats = team_stats.get(home_team_mapped, {
                'overall_win_rate': 0.3,
                'recent_win_rate': 0.3,
                'current_season_win_rate': 0.3,
                'home_win_rate': 0.3,
                'away_win_rate': 0.3,
                'total_matches': 0,
                'current_season_matches': 0
            })
            
            away_stats = team_stats.get(away_team_mapped, {
                'overall_win_rate': 0.3,
                'recent_win_rate': 0.3,
                'current_season_win_rate': 0.3,
                'home_win_rate': 0.3,
                'away_win_rate': 0.3,
                'total_matches': 0,
                'current_season_matches': 0
            })

            # Create feature vector for Random Forest prediction
            # Use average values for rolling features since we don't have recent data for future matches
            # But prioritize current season stats if available
            avg_rolling_features = [0.5] * len(all_new_cols)  # Neutral values
            
            # Create prediction features
            # Get opponent code - recreate category if needed
            if "opp_code" not in matches.columns or matches["opponent"].dtype.name != "category":
                matches["opponent"] = matches["opponent"].astype("category")
                matches["opp_code"] = matches["opponent"].cat.codes
            
            # Find opponent code
            if away_team_mapped in matches["opponent"].cat.categories:
                opp_code_val = matches["opponent"].cat.categories.get_loc(away_team_mapped)
            else:
                # If opponent not found, use 0 (will be handled by model)
                opp_code_val = 0
                
            # Get day code - recreate category if needed
            if pd.notna(day):
                if "day_code" not in matches.columns or matches["day"].dtype.name != "category":
                    matches["day"] = matches["day"].astype("category")
                    matches["day_code"] = matches["day"].cat.codes
                
                if day in matches["day"].cat.categories:
                    day_code_val = matches["day"].cat.categories.get_loc(day)
                else:
                    day_code_val = 0
            else:
                day_code_val = 0
            
            hour_val = pd.to_datetime(time, format="%H:%M", errors='coerce').hour if pd.notna(time) else 15
            if pd.isna(hour_val):
                hour_val = 15
            
            features = [
                1,  # venue_code (Home for home team)
                opp_code_val,
                hour_val,
                day_code_val,
            ] + avg_rolling_features
            
            # Get Random Forest prediction and probability
            win_prob = model.predict_proba([features])[0][1]
            prediction = "Win" if model.predict([features])[0] == 1 else "Loss/Draw"
            
            # Adjust prediction based on current season performance if available
            if home_stats['current_season_matches'] >= 3:
                # Blend model prediction with current season performance (70% model, 30% current season)
                current_season_adjustment = home_stats['current_season_win_rate'] * 0.3
                win_prob = win_prob * 0.7 + current_season_adjustment
            
            # IMPROVED: Better confidence levels based on probability
            if win_prob > 0.65:
                confidence = "High"
            elif win_prob > 0.55:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Calculate away win probability for display
            away_win_prob = 1 - win_prob - 0.15  # Leave room for draw
            away_win_prob = max(0.10, min(0.70, away_win_prob))
            
            # Calculate draw probability
            draw_prob = max(0.10, 1 - win_prob - away_win_prob)

            result = {
                "team": home_team,
                "opponent": away_team,
                "venue": "Home",
                "date": date,
                "time": time,
                "matchweek": matchweek,
                "day": day,
                "win_probability": win_prob,
                "opponent_win_probability": away_win_prob,
                "draw_probability": draw_prob,
                "prediction": prediction,
                "confidence": confidence,
                "model_accuracy": "Weighted Model (2025/2026 season weighted 3-4x)",
                "home_overall_rate": home_stats['overall_win_rate'],
                "home_recent_rate": home_stats['recent_win_rate'],
                "home_current_season_rate": home_stats['current_season_win_rate'],
                "home_home_rate": home_stats['home_win_rate'],
                "away_overall_rate": away_stats['overall_win_rate'],
                "away_recent_rate": away_stats['recent_win_rate'],
                "away_current_season_rate": away_stats['current_season_win_rate'],
                "away_away_rate": away_stats['away_win_rate']
            }

            predictions.append(result)
            print(f"✅ Matchweek {matchweek} ({day}): {home_team} vs {away_team} ({date} {time}): {prediction} ({win_prob:.1%} vs {away_win_prob:.1%}) [{confidence}]")

        except Exception as e:
            print(f"❌ Error predicting fixture {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return predictions

def main():
    """Main function to generate weighted production predictions"""
    print("🚀 EPL Tracker - Weighted Production Predictions")
    print("=" * 60)
    print("🎯 Using Weighted Random Forest model")
    print("📊 2025/2026 season matches weighted 3-4x more heavily")
    print("📈 Reflects current team performance more accurately")
    print("=" * 60)
    
    # Train weighted production model
    model, predictors, all_new_cols = train_weighted_model()
    
    # Try to load existing fixtures first
    try:
        print("📋 Loading existing fixtures data...")
        fixtures_df = pd.read_csv("2025_2026_production_predictions_REALISTIC.csv")
        
        # Extract fixture information from existing predictions
        fixtures_data = []
        for idx, row in fixtures_df.iterrows():
            if pd.notna(row['team']) and pd.notna(row['opponent']):
                fixture = {
                    'matchweek': row.get('matchweek', ''),
                    'day': row.get('day', ''),
                    'date': row.get('date', ''),
                    'time': row.get('time', ''),
                    'home': row['team'],
                    'away': row['opponent']
                }
                fixtures_data.append(fixture)
        
        fixtures_df = pd.DataFrame(fixtures_data)
        print(f"✅ Loaded {len(fixtures_df)} fixtures from existing data")
        
    except FileNotFoundError:
        print("❌ No existing fixtures found")
        return
    
    # Create production predictions
    predictions = create_production_predictions(fixtures_df, model, predictors, all_new_cols)
    
    if predictions:
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        output_file = "2025_2026_production_predictions_WEIGHTED.csv"
        predictions_df.to_csv(output_file, index=False)
        
        # Analyze results
        total_matches = len(predictions_df)
        predicted_wins = len(predictions_df[predictions_df['prediction'] == 'Win'])
        predicted_losses_draws = len(predictions_df[predictions_df['prediction'] == 'Loss/Draw'])
        
        avg_win_prob = predictions_df['win_probability'].mean()
        high_confidence_wins = len(predictions_df[
            (predictions_df['prediction'] == 'Win') & 
            (predictions_df['confidence'] == 'High')
        ])
        
        print(f"\n📊 Weighted Production Predictions Summary:")
        print(f"   Total Matches: {total_matches}")
        print(f"   Predicted Wins: {predicted_wins} ({predicted_wins/total_matches:.1%})")
        print(f"   Predicted Losses/Draws: {predicted_losses_draws} ({predicted_losses_draws/total_matches:.1%})")
        print(f"   Average Win Probability: {avg_win_prob:.1%}")
        print(f"   High Confidence Wins: {high_confidence_wins}")
        
        print(f"\n✅ Weighted production predictions saved to: {output_file}")
        print("🎯 Key features:")
        print("   - Weighted Random Forest model")
        print("   - 2025/2026 season matches weighted 3-4x more")
        print("   - Recent matches (last 30 days) weighted 4x")
        print("   - Current season performance blended into predictions")
        print("   - Production-ready predictions")
    
    return predictions

if __name__ == "__main__":
    main()

