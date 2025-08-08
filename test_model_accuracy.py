#!/usr/bin/env python3
"""
EPL Tracker - Model Accuracy Testing
Train on pre-2024-2025 data and test on 2024-2025 results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
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
    
    # Calculate rolling averages
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group

    # Use only available columns
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    # Calculate additional features that we can compute
    matches["xg_diff"] = matches["xg"] - matches["xga"]
    matches["goals_per_xg"] = matches["gf"] / matches["xg"].replace(0, 1)
    matches["shots_accuracy"] = matches["sot"] / matches["sh"].replace(0, 1)
    
    # Add these to the rolling averages
    additional_cols = ["xg_diff", "goals_per_xg", "shots_accuracy"]
    additional_new_cols = [f"{c}_rolling" for c in additional_cols]
    
    # Combine all columns for rolling averages
    all_cols = cols + additional_cols
    all_new_cols = new_cols + additional_new_cols
    
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, all_cols, all_new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    return matches_rolling, all_new_cols

def train_and_test_model():
    """Train model on pre-2024-2025 data and test on 2024-2025 results"""
    print("🤖 Training and testing model accuracy...")
    
    # Load data
    matches_rolling, all_new_cols = load_and_prepare_data()
    
    # Define predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"] + all_new_cols
    
    # Split data: train on pre-2024-2025, test on 2024-2025
    train_data = matches_rolling[matches_rolling["date"] < '2024-08-01']
    test_data = matches_rolling[matches_rolling["date"] >= '2024-08-01']
    
    print(f"📊 Training data: {len(train_data)} matches (pre-2024-2025)")
    print(f"📊 Test data: {len(test_data)} matches (2024-2025)")
    
    # Train model
    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=15,
        random_state=42
    )
    
    model.fit(train_data[predictors], train_data["target"])
    
    # Make predictions on test data
    print("🎯 Making predictions on 2024-2025 data...")
    predictions = model.predict(test_data[predictors])
    prediction_probs = model.predict_proba(test_data[predictors])[:, 1]
    
    # Calculate metrics
    actual = test_data["target"]
    
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions)
    recall = recall_score(actual, predictions)
    f1 = f1_score(actual, predictions)
    
    print(f"\n📈 Model Accuracy Results:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall: {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1-Score: {f1:.3f} ({f1*100:.1f}%)")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(actual, predictions, target_names=['Loss/Draw', 'Win']))
    
    # Analyze prediction distribution
    predicted_wins = sum(predictions)
    actual_wins = sum(actual)
    total_matches = len(predictions)
    
    print(f"\n📊 Prediction Distribution Analysis:")
    print(f"   Total Matches: {total_matches}")
    print(f"   Predicted Wins: {predicted_wins} ({predicted_wins/total_matches:.1%})")
    print(f"   Actual Wins: {actual_wins} ({actual_wins/total_matches:.1%})")
    print(f"   Prediction Error: {abs(predicted_wins - actual_wins)} matches")
    
    # Analyze by confidence levels (using probability thresholds)
    high_conf_threshold = 0.65
    medium_conf_threshold = 0.55
    
    high_conf_mask = prediction_probs > high_conf_threshold
    medium_conf_mask = (prediction_probs > medium_conf_threshold) & (prediction_probs <= high_conf_threshold)
    low_conf_mask = prediction_probs <= medium_conf_threshold
    
    high_conf_accuracy = accuracy_score(actual[high_conf_mask], predictions[high_conf_mask]) if sum(high_conf_mask) > 0 else 0
    medium_conf_accuracy = accuracy_score(actual[medium_conf_mask], predictions[medium_conf_mask]) if sum(medium_conf_mask) > 0 else 0
    low_conf_accuracy = accuracy_score(actual[low_conf_mask], predictions[low_conf_mask]) if sum(low_conf_mask) > 0 else 0
    
    print(f"\n🎯 Accuracy by Confidence Level:")
    print(f"   High Confidence (>65%): {sum(high_conf_mask)} predictions, {high_conf_accuracy:.1%} accuracy")
    print(f"   Medium Confidence (55-65%): {sum(medium_conf_mask)} predictions, {medium_conf_accuracy:.1%} accuracy")
    print(f"   Low Confidence (<55%): {sum(low_conf_mask)} predictions, {low_conf_accuracy:.1%} accuracy")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_wins': predicted_wins,
        'actual_wins': actual_wins,
        'total_matches': total_matches,
        'high_conf_accuracy': high_conf_accuracy,
        'medium_conf_accuracy': medium_conf_accuracy,
        'low_conf_accuracy': low_conf_accuracy,
        'high_conf_count': sum(high_conf_mask),
        'medium_conf_count': sum(medium_conf_mask),
        'low_conf_count': sum(low_conf_mask)
    }

def test_enhanced_approach():
    """Test the enhanced approach on 2024-2025 data"""
    print("\n🔄 Testing Enhanced Approach on 2024-2025 Data")
    print("=" * 60)
    
    # Load data
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    
    # Filter for 2024-2025 season
    test_matches = matches[matches["date"] >= '2024-08-01'].copy()
    
    print(f"📊 Testing on {len(test_matches)} 2024-2025 matches")
    
    # Calculate team stats from pre-2024-2025 data
    historical_matches = matches[matches["date"] < '2024-08-01']
    
    team_stats = {}
    for team in historical_matches["team"].unique():
        team_matches = historical_matches[historical_matches["team"] == team]
        
        # Overall win rate
        wins = len(team_matches[team_matches["result"] == "W"])
        total = len(team_matches)
        win_rate = wins / total if total > 0 else 0.3
        
        # Recent form (last 10 matches)
        recent_matches = team_matches.sort_values('date').tail(10)
        recent_wins = len(recent_matches[recent_matches["result"] == "W"])
        recent_win_rate = recent_wins / len(recent_matches) if len(recent_matches) > 0 else win_rate
        
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
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'total_matches': total
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
    
    enhanced_predictions = []
    correct_predictions = 0
    
    for idx, match in test_matches.iterrows():
        try:
            home_team = match['team']
            away_team = match['opponent']
            actual_result = match['result']
            
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
            
            # Enhanced prediction logic
            home_strength = (
                0.35 * home_stats['home_win_rate'] +
                0.25 * home_stats['recent_win_rate'] +
                0.25 * home_stats['overall_win_rate'] +
                0.15 * (1 - away_stats['away_win_rate'])
            )
            
            away_strength = (
                0.35 * away_stats['away_win_rate'] +
                0.25 * away_stats['recent_win_rate'] +
                0.25 * away_stats['overall_win_rate'] +
                0.15 * (1 - home_stats['home_win_rate'])
            )
            
            home_advantage = 0.10
            home_win_prob = home_strength + home_advantage
            away_win_prob = away_strength
            
            # Apply bounds
            home_win_prob = max(0.20, min(0.80, home_win_prob))
            away_win_prob = max(0.15, min(0.70, away_win_prob))
            
            # Normalize
            total_prob = home_win_prob + away_win_prob
            if total_prob > 0.85:
                home_win_prob = home_win_prob * 0.85 / total_prob
                away_win_prob = away_win_prob * 0.85 / total_prob
            
            # Make prediction
            if home_win_prob > 0.55:
                prediction = "Win"
                confidence = "High" if home_win_prob > 0.65 else "Medium"
            elif home_win_prob > 0.45:
                prediction = "Win"
                confidence = "Low"
            elif away_win_prob > 0.55:
                prediction = "Loss/Draw"
                confidence = "High" if away_win_prob > 0.65 else "Medium"
            else:
                prediction = "Loss/Draw"
                confidence = "Low"
            
            # Check if prediction is correct
            actual_win = actual_result == "W"
            predicted_win = prediction == "Win"
            is_correct = actual_win == predicted_win
            
            if is_correct:
                correct_predictions += 1
            
            enhanced_predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'actual_result': actual_result,
                'prediction': prediction,
                'confidence': confidence,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob,
                'is_correct': is_correct
            })
            
        except Exception as e:
            print(f"❌ Error processing match {idx}: {e}")
            continue
    
    # Calculate enhanced approach accuracy
    enhanced_accuracy = correct_predictions / len(enhanced_predictions) if enhanced_predictions else 0
    
    print(f"\n📈 Enhanced Approach Results:")
    print(f"   Total Matches: {len(enhanced_predictions)}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Accuracy: {enhanced_accuracy:.3f} ({enhanced_accuracy*100:.1f}%)")
    
    # Analyze by confidence
    enhanced_df = pd.DataFrame(enhanced_predictions)
    
    for conf_level in ['High', 'Medium', 'Low']:
        conf_matches = enhanced_df[enhanced_df['confidence'] == conf_level]
        if len(conf_matches) > 0:
            conf_accuracy = conf_matches['is_correct'].mean()
            print(f"   {conf_level} Confidence: {len(conf_matches)} predictions, {conf_accuracy:.1%} accuracy")
    
    return enhanced_accuracy

def main():
    """Main function to test model accuracy"""
    print("🏆 EPL Tracker - Model Accuracy Testing")
    print("=" * 60)
    
    # Test Random Forest model
    rf_results = train_and_test_model()
    
    # Test enhanced approach
    enhanced_accuracy = test_enhanced_approach()
    
    # Compare approaches
    print(f"\n🔄 Approach Comparison:")
    print("=" * 40)
    print(f"Random Forest Model: {rf_results['accuracy']:.1%} accuracy")
    print(f"Enhanced Approach: {enhanced_accuracy:.1%} accuracy")
    
    if rf_results['accuracy'] > enhanced_accuracy:
        print(f"✅ Random Forest performs better by {rf_results['accuracy'] - enhanced_accuracy:.1%}")
    else:
        print(f"✅ Enhanced approach performs better by {enhanced_accuracy - rf_results['accuracy']:.1%}")
    
    print(f"\n🎯 Key Insights:")
    print(f"- Model accuracy: {rf_results['accuracy']:.1%}")
    print(f"- Win prediction accuracy: {rf_results['precision']:.1%}")
    print(f"- High confidence accuracy: {rf_results['high_conf_accuracy']:.1%}")
    print(f"- Enhanced approach accuracy: {enhanced_accuracy:.1%}")

if __name__ == "__main__":
    main() 