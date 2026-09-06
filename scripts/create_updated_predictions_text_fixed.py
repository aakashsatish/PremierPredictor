#!/usr/bin/env python3
"""
Create an updated predictions.txt file that includes actual results from completed 2025/2026 matches
FIXED VERSION - Ensures all matches show prediction accuracy
"""

import pandas as pd
from datetime import datetime

def create_updated_predictions_text():
    """Create an updated text file with predictions and actual results"""
    
    # Load the updated matches data
    matches_df = pd.read_csv("matches.csv")
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    
    # Filter for 2025/2026 season matches that have been completed
    completed_matches = matches_df[
        (matches_df["date"] >= "2025-08-01") & 
        (matches_df["date"] <= datetime.now().strftime("%Y-%m-%d")) &
        (matches_df["result"].notna())
    ].copy()
    
    # Load predictions if available
    try:
        predictions_df = pd.read_csv("2025_2026_production_predictions_REALISTIC.csv")
        predictions_df["date"] = pd.to_datetime(predictions_df["date"])
        has_predictions = True
    except FileNotFoundError:
        has_predictions = False
        predictions_df = None
    
    # Create the text content
    text_content = []
    text_content.append("=" * 80)
    text_content.append("EPL TRACKER - 2025/2026 SEASON RESULTS & PREDICTIONS")
    text_content.append("=" * 80)
    text_content.append("")
    text_content.append("📅 Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    text_content.append("🎯 Model: Random Forest + Team Strength Comparison")
    text_content.append("📊 Accuracy: 64.5% (based on historical testing)")
    text_content.append("")
    text_content.append("Confidence Levels:")
    text_content.append("- High: >65% win probability")
    text_content.append("- Medium: 55-65% win probability") 
    text_content.append("- Low: <55% win probability")
    text_content.append("")
    text_content.append("=" * 80)
    text_content.append("")
    
    # Group completed matches by matchweek
    completed_matches["matchweek"] = completed_matches["round"].str.extract(r'(\d+)').astype(float)
    
    # Team name mapping for better matching
    team_mappings = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd", 
        "Newcastle United": "Newcastle Utd", 
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Nottingham Forest": "Nott'ham Forest"
    }
    
    correct_predictions = 0
    total_with_predictions = 0
    
    for matchweek in sorted(completed_matches['matchweek'].unique()):
        if pd.isna(matchweek):
            continue
            
        week_matches = completed_matches[completed_matches['matchweek'] == matchweek].sort_values('date')
        
        text_content.append(f"MATCHWEEK {int(matchweek)} - COMPLETED MATCHES")
        text_content.append("-" * 50)
        text_content.append("")
        
        for idx, match in week_matches.iterrows():
            team = match['team']
            opponent = match['opponent']
            date = match['date'].strftime("%Y-%m-%d")
            time = match['time']
            day = match['day']
            venue = match['venue']
            result = match['result']
            gf = match['gf']
            ga = match['ga']
            
            # Format the actual result
            if result == "W":
                result_text = f"{team} WIN"
            elif result == "L":
                result_text = f"{opponent} WIN"
            else:
                result_text = "DRAW"
            
            # Show home vs away
            if venue == "Home":
                match_text = f"{team} vs {opponent}"
            else:
                match_text = f"{opponent} vs {team}"
            
            text_content.append(f"{day} {date} {time}")
            text_content.append(f"{match_text}")
            text_content.append(f"RESULT: {result_text} ({gf}-{ga})")
            
            # Add prediction if available - try multiple matching strategies
            prediction_found = False
            if has_predictions:
                # Try different matching strategies
                matching_strategies = [
                    # Strategy 1: Exact team names and date
                    (team, opponent, date),
                    # Strategy 2: Mapped team names and date
                    (team_mappings.get(team, team), team_mappings.get(opponent, opponent), date),
                    # Strategy 3: Reverse team names and date
                    (opponent, team, date),
                    # Strategy 4: Mapped reverse team names and date
                    (team_mappings.get(opponent, opponent), team_mappings.get(team, team), date)
                ]
                
                for home_team, away_team, match_date in matching_strategies:
                    pred_match = predictions_df[
                        (predictions_df['team'] == home_team) & 
                        (predictions_df['opponent'] == away_team) &
                        (predictions_df['date'].dt.strftime("%Y-%m-%d") == match_date)
                    ]
                    
                    if len(pred_match) > 0:
                        pred = pred_match.iloc[0]
                        win_prob = pred['win_probability']
                        away_win_prob = pred['opponent_win_probability']
                        draw_prob = pred['draw_probability']
                        prediction = pred['prediction']
                        confidence = pred['confidence']
                        
                        # Format the prediction
                        if prediction == "Win":
                            pred_text = f"{home_team} WIN"
                        else:
                            pred_text = f"{away_team} WIN or DRAW"
                        
                        text_content.append(f"PREDICTION: {pred_text} [{confidence}]")
                        text_content.append(f"Probabilities: {home_team} {win_prob:.1%} | {away_team} {away_win_prob:.1%} | Draw {draw_prob:.1%}")
                        
                        # Check if prediction was correct
                        total_with_predictions += 1
                        
                        # Determine if prediction was correct based on actual result
                        if venue == "Home":
                            # Home team perspective
                            if (result == "W" and prediction == "Win") or (result in ["L", "D"] and prediction == "Loss/Draw"):
                                text_content.append("✅ PREDICTION CORRECT")
                                correct_predictions += 1
                            else:
                                text_content.append("❌ PREDICTION INCORRECT")
                        else:
                            # Away team perspective
                            if (result == "L" and prediction == "Win") or (result in ["W", "D"] and prediction == "Loss/Draw"):
                                text_content.append("✅ PREDICTION CORRECT")
                                correct_predictions += 1
                            else:
                                text_content.append("❌ PREDICTION INCORRECT")
                        
                        prediction_found = True
                        break
                
                if not prediction_found:
                    text_content.append("⚠️  NO PREDICTION AVAILABLE")
            
            text_content.append("")
    
    # Add upcoming matches if predictions are available
    if has_predictions:
        upcoming_matches = predictions_df[
            predictions_df["date"] > datetime.now().strftime("%Y-%m-%d")
        ]
        
        if len(upcoming_matches) > 0:
            text_content.append("=" * 80)
            text_content.append("UPCOMING MATCHES - PREDICTIONS")
            text_content.append("=" * 80)
            text_content.append("")
            
            # Group upcoming matches by matchweek
            for matchweek in sorted(upcoming_matches['matchweek'].unique()):
                if pd.isna(matchweek):
                    continue
                    
                week_matches = upcoming_matches[upcoming_matches['matchweek'] == matchweek].sort_values('date')
                
                text_content.append(f"MATCHWEEK {int(matchweek)} - UPCOMING")
                text_content.append("-" * 40)
                text_content.append("")
                
                for idx, match in week_matches.iterrows():
                    home_team = match['team']
                    away_team = match['opponent']
                    date = match['date'].strftime("%Y-%m-%d")
                    time = match['time']
                    day = match['day']
                    
                    win_prob = match['win_probability']
                    away_win_prob = match['opponent_win_probability']
                    draw_prob = match['draw_probability']
                    prediction = match['prediction']
                    confidence = match['confidence']
                    
                    # Format the prediction
                    if prediction == "Win":
                        result_text = f"{home_team} WIN"
                    else:
                        result_text = f"{away_team} WIN or DRAW"
                    
                    text_content.append(f"{day} {date} {time}")
                    text_content.append(f"{home_team} vs {away_team}")
                    text_content.append(f"PREDICTION: {result_text} [{confidence}]")
                    text_content.append(f"Probabilities: {home_team} {win_prob:.1%} | {away_team} {away_win_prob:.1%} | Draw {draw_prob:.1%}")
                    text_content.append("")
    
    # Add summary statistics
    text_content.append("=" * 80)
    text_content.append("SUMMARY STATISTICS")
    text_content.append("=" * 80)
    text_content.append("")
    
    total_completed = len(completed_matches)
    text_content.append(f"Completed Matches: {total_completed}")
    
    if has_predictions:
        total_predicted = len(predictions_df)
        text_content.append(f"Total Predicted Matches: {total_predicted}")
        
        if total_with_predictions > 0:
            accuracy = correct_predictions / total_with_predictions
            text_content.append(f"Prediction Accuracy: {correct_predictions}/{total_with_predictions} ({accuracy:.1%})")
        else:
            text_content.append("No predictions available for completed matches")
    
    # Team performance summary
    text_content.append("")
    text_content.append("TEAM PERFORMANCE SUMMARY:")
    text_content.append("-" * 30)
    
    team_stats = completed_matches.groupby('team').agg({
        'result': ['count', lambda x: (x == 'W').sum(), lambda x: (x == 'L').sum(), lambda x: (x == 'D').sum()]
    }).round(1)
    
    team_stats.columns = ['Matches', 'Wins', 'Losses', 'Draws']
    team_stats['Win%'] = (team_stats['Wins'] / team_stats['Matches'] * 100).round(1)
    team_stats = team_stats.sort_values('Win%', ascending=False)
    
    for team, stats in team_stats.iterrows():
        text_content.append(f"{team}: {stats['Wins']:.0f}W-{stats['Draws']:.0f}D-{stats['Losses']:.0f}L ({stats['Win%']:.1f}%)")
    
    text_content.append("")
    text_content.append("=" * 80)
    text_content.append("END OF RESULTS & PREDICTIONS")
    text_content.append("=" * 80)
    
    # Write to file
    with open("2025_2026_updated_predictions.txt", "w") as f:
        f.write("\n".join(text_content))
    
    print("✅ Created updated predictions text file: 2025_2026_updated_predictions.txt")
    print(f"📊 Completed matches: {total_completed}")
    if has_predictions and total_with_predictions > 0:
        print(f"🎯 Prediction accuracy: {correct_predictions}/{total_with_predictions} ({accuracy:.1%})")
    print(f"📈 Teams analyzed: {len(team_stats)}")

if __name__ == "__main__":
    create_updated_predictions_text()
