#!/usr/bin/env python3
"""
Test Random Forest Model Accuracy
Evaluates the original RF model's accuracy on historical data
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

def test_rf_model_accuracy():
    """Test the Random Forest model's accuracy on historical data"""
    print("🔍 Testing Random Forest Model Accuracy")
    print("=" * 50)
    
    # Load and prepare the data (same as predictions.ipynb)
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    
    # Feature engineering (same as notebook)
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    matches["target"] = (matches["result"] == 'W').astype("int")
    
    # High-value features
    matches["xg_diff"] = matches["xg"] - matches["xga"]
    matches["xg_per_shot"] = matches["xg"] / matches["sh"].replace(0, 1)
    matches["goals_per_xg"] = matches["gf"] / matches["xg"].replace(0, 1)
    matches["shots_accuracy"] = matches["sot"] / matches["sh"].replace(0, 1)
    matches["possession_efficiency"] = matches["gf"] / (matches["poss"] + 1)
    matches["formation_code"] = matches["formation"].astype("category").cat.codes
    matches["season_stage"] = matches["date"].dt.month
    
    # Create rolling averages
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
    
    # Define predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"]
    updated_predictors = predictors + ["xg_per_shot", "possession_efficiency", "formation_code", "season_stage"] + new_cols
    
    print(f"📊 Total matches with rolling averages: {len(matches_rolling)}")
    print(f"🎯 Features used: {len(updated_predictors)}")
    
    # Test accuracy using time-based split
    matches_sorted = matches_rolling.sort_values('date')
    
    # Use first 80% for training, last 20% for testing
    split_point = int(len(matches_sorted) * 0.8)
    train_data = matches_sorted.iloc[:split_point]
    test_data = matches_sorted.iloc[split_point:]
    
    print(f"📚 Training on {len(train_data)} matches")
    print(f"🧪 Testing on {len(test_data)} matches")
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    rf_model.fit(train_data[updated_predictors], train_data["target"])
    
    # Make predictions on test set
    predictions = rf_model.predict(test_data[updated_predictors])
    actual = test_data["target"]
    
    # Calculate accuracy
    accuracy = accuracy_score(actual, predictions)
    
    print("\n" + "=" * 50)
    print("📊 RANDOM FOREST MODEL ACCURACY RESULTS")
    print("=" * 50)
    print(f"🎯 Test Accuracy: {accuracy:.1%}")
    print(f"✅ Correct Predictions: {sum(predictions == actual)}")
    print(f"📊 Total Predictions: {len(predictions)}")
    
    # Detailed classification report
    print("\n📋 Classification Report:")
    print(classification_report(actual, predictions, target_names=['Loss/Draw', 'Win']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': updated_predictors,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return accuracy

if __name__ == "__main__":
    accuracy = test_rf_model_accuracy()
    print(f"\n💡 Random Forest Model Accuracy: {accuracy:.1%}") 