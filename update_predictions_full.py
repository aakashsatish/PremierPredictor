#!/usr/bin/env python3
"""
EPL Tracker - Full Update Pipeline
1. Rescrapes 2025/2026 season matches with full stats
2. Trains weighted model (2025/2026 matches weighted 3-4x more)
3. Generates new predictions
4. Creates updated predictions.txt file
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f"🔄 {description}")
    print("=" * 80)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_updated_predictions_text():
    """Create updated predictions.txt file with results and predictions"""
    import pandas as pd
    from datetime import datetime
    
    print("\n" + "=" * 80)
    print("📝 Creating updated predictions.txt file")
    print("=" * 80)
    
    # Load the updated matches data
    matches_df = pd.read_csv("matches.csv")
    matches_df["date"] = pd.to_datetime(matches_df["date"], errors='coerce')
    
    # Filter for 2025/2026 season matches that have been completed
    completed_matches = matches_df[
        (matches_df["date"] >= "2025-08-01") & 
        (matches_df["date"].notna()) &
        (matches_df["result"].notna())
    ].copy()
    
    # Load weighted predictions
    try:
        predictions_df = pd.read_csv("2025_2026_production_predictions_WEIGHTED.csv")
        predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors='coerce')
        has_predictions = True
        print(f"✅ Loaded {len(predictions_df)} predictions")
    except FileNotFoundError:
        print("⚠️  Weighted predictions file not found, trying REALISTIC predictions...")
        try:
            predictions_df = pd.read_csv("2025_2026_production_predictions_REALISTIC.csv")
            predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors='coerce')
            has_predictions = True
            print(f"✅ Loaded {len(predictions_df)} predictions from REALISTIC file")
        except FileNotFoundError:
            has_predictions = False
            predictions_df = None
            print("⚠️  No predictions file found, will only show results")
    
    # Create the text content
    text_content = []
    text_content.append("=" * 80)
    text_content.append("EPL TRACKER - 2025/2026 SEASON RESULTS & PREDICTIONS")
    text_content.append("=" * 80)
    text_content.append("")
    text_content.append("📅 Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    text_content.append("🎯 Model: Weighted Random Forest (2025/2026 season weighted 3-4x more)")
    text_content.append("📊 Model: Current season matches weighted 3-4x more heavily")
    text_content.append("")
    text_content.append("Confidence Levels:")
    text_content.append("- High: >65% win probability")
    text_content.append("- Medium: 55-65% win probability") 
    text_content.append("- Low: <55% win probability")
    text_content.append("")
    text_content.append("=" * 80)
    text_content.append("")
    
    # Group completed matches by matchweek
    if "round" in completed_matches.columns:
        completed_matches["matchweek"] = completed_matches["round"].str.extract(r'(\d+)').astype(float)
    elif "matchweek" in completed_matches.columns:
        completed_matches["matchweek"] = pd.to_numeric(completed_matches["matchweek"], errors='coerce')
    else:
        # Create matchweek from date
        start_date = pd.to_datetime("2025-08-15")
        completed_matches["matchweek"] = ((completed_matches["date"] - start_date).dt.days // 7 + 1).astype(int)
        completed_matches["matchweek"] = completed_matches["matchweek"].clip(lower=1)
    
    # Sort by matchweek and date
    completed_matches = completed_matches.sort_values(["matchweek", "date"])
    
    for matchweek in sorted(completed_matches['matchweek'].dropna().unique()):
        week_matches = completed_matches[completed_matches['matchweek'] == matchweek].sort_values('date')
        
        text_content.append(f"MATCHWEEK {int(matchweek)} - COMPLETED MATCHES")
        text_content.append("-" * 80)
        text_content.append("")
        
        for idx, match in week_matches.iterrows():
            team = match['team']
            opponent = match['opponent']
            date = match['date']
            if pd.notna(date):
                date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            else:
                date_str = "N/A"
            time = match.get('time', 'N/A')
            day = match.get('day', '')
            venue = match.get('venue', '')
            result = match['result']
            gf = match.get('gf', 'N/A')
            ga = match.get('ga', 'N/A')
            
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
            
            text_content.append(f"{day} {date_str} {time}")
            text_content.append(f"{match_text}")
            text_content.append(f"RESULT: {result_text} ({gf}-{ga})")
            
            # Add prediction if available
            if has_predictions:
                pred_match = predictions_df[
                    (predictions_df['team'] == team) & 
                    (predictions_df['opponent'] == opponent) &
                    (predictions_df['date'].notna()) &
                    (pd.to_datetime(predictions_df['date']).dt.strftime("%Y-%m-%d") == date_str)
                ]
                
                if len(pred_match) == 0:
                    # Try without date match
                    pred_match = predictions_df[
                        (predictions_df['team'] == team) & 
                        (predictions_df['opponent'] == opponent)
                    ]
                
                if len(pred_match) > 0:
                    pred = pred_match.iloc[0]
                    win_prob = pred.get('win_probability', 0)
                    away_win_prob = pred.get('opponent_win_probability', 0)
                    draw_prob = pred.get('draw_probability', 0)
                    prediction = pred.get('prediction', 'N/A')
                    confidence = pred.get('confidence', 'N/A')
                    
                    # Format the prediction
                    if prediction == "Win":
                        pred_text = f"{team} WIN"
                    else:
                        pred_text = f"{opponent} WIN or DRAW"
                    
                    text_content.append(f"PREDICTION: {pred_text} [{confidence}]")
                    text_content.append(f"Probabilities: {team} {win_prob:.1%} | {opponent} {away_win_prob:.1%} | Draw {draw_prob:.1%}")
                    
                    # Check if prediction was correct
                    correct = False
                    if result == "W" and prediction == "Win":
                        correct = True
                    elif result == "L" and prediction == "Loss/Draw":
                        correct = True
                    elif result == "D" and prediction == "Loss/Draw":
                        correct = True
                    
                    if correct:
                        text_content.append("✅ PREDICTION CORRECT")
                    else:
                        text_content.append("❌ PREDICTION INCORRECT")
            
            text_content.append("")
    
    # Add upcoming matches if predictions are available
    if has_predictions:
        upcoming_matches = predictions_df[
            (predictions_df["date"].notna()) &
            (pd.to_datetime(predictions_df["date"]) > datetime.now())
        ]
        
        if len(upcoming_matches) > 0:
            text_content.append("=" * 80)
            text_content.append("UPCOMING MATCHES - PREDICTIONS")
            text_content.append("=" * 80)
            text_content.append("")
            
            # Group upcoming matches by matchweek
            for matchweek in sorted(upcoming_matches['matchweek'].dropna().unique()):
                week_matches = upcoming_matches[upcoming_matches['matchweek'] == matchweek].sort_values('date')
                
                text_content.append(f"MATCHWEEK {int(matchweek)} - UPCOMING")
                text_content.append("-" * 80)
                text_content.append("")
                
                for idx, match in week_matches.iterrows():
                    home_team = match['team']
                    away_team = match['opponent']
                    date = match['date']
                    if pd.notna(date):
                        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    else:
                        date_str = "N/A"
                    time = match.get('time', 'N/A')
                    day = match.get('day', '')
                    
                    win_prob = match.get('win_probability', 0)
                    away_win_prob = match.get('opponent_win_probability', 0)
                    draw_prob = match.get('draw_probability', 0)
                    prediction = match.get('prediction', 'N/A')
                    confidence = match.get('confidence', 'N/A')
                    
                    # Format the prediction
                    if prediction == "Win":
                        result_text = f"{home_team} WIN"
                    else:
                        result_text = f"{away_team} WIN or DRAW"
                    
                    text_content.append(f"{day} {date_str} {time}")
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
        
        # Calculate accuracy for completed matches
        correct_predictions = 0
        total_with_predictions = 0
        
        for idx, match in completed_matches.iterrows():
            team = match['team']
            opponent = match['opponent']
            date = match['date']
            if pd.notna(date):
                date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            else:
                continue
            result = match['result']
            
            pred_match = predictions_df[
                (predictions_df['team'] == team) & 
                (predictions_df['opponent'] == opponent) &
                (predictions_df['date'].notna()) &
                (pd.to_datetime(predictions_df['date']).dt.strftime("%Y-%m-%d") == date_str)
            ]
            
            if len(pred_match) == 0:
                pred_match = predictions_df[
                    (predictions_df['team'] == team) & 
                    (predictions_df['opponent'] == opponent)
                ]
            
            if len(pred_match) > 0:
                total_with_predictions += 1
                pred = pred_match.iloc[0]
                prediction = pred.get('prediction', '')
                
                if (result == "W" and prediction == "Win") or \
                   (result == "L" and prediction == "Loss/Draw") or \
                   (result == "D" and prediction == "Loss/Draw"):
                    correct_predictions += 1
        
        if total_with_predictions > 0:
            accuracy = correct_predictions / total_with_predictions
            text_content.append(f"Prediction Accuracy: {correct_predictions}/{total_with_predictions} ({accuracy:.1%})")
    
    # Team performance summary
    text_content.append("")
    text_content.append("TEAM PERFORMANCE SUMMARY (2025/2026 Season):")
    text_content.append("-" * 80)
    
    team_stats = completed_matches.groupby('team').agg({
        'result': ['count', lambda x: (x == 'W').sum(), lambda x: (x == 'L').sum(), lambda x: (x == 'D').sum()]
    })
    
    if len(team_stats.columns) >= 4:
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
    output_file = "2025_2026_updated_predictions.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(text_content))
    
    print(f"✅ Created updated predictions text file: {output_file}")
    print(f"📊 Completed matches: {total_completed}")
    if has_predictions and total_with_predictions > 0:
        print(f"🎯 Prediction accuracy: {correct_predictions}/{total_with_predictions} ({accuracy:.1%})")
    if len(team_stats) > 0:
        print(f"📈 Teams analyzed: {len(team_stats)}")

def main():
    """Main function to run the full update pipeline"""
    print("🚀 EPL Tracker - Full Update Pipeline")
    print("=" * 80)
    print("📅 Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("")
    print("This will:")
    print("1. Rescrape 2025/2026 season matches with full statistics")
    print("2. Train weighted model (2025/2026 matches weighted 3-4x more)")
    print("3. Generate new predictions")
    print("4. Create updated predictions.txt file")
    print("=" * 80)
    
    # Step 1: Rescrape season data
    success = run_script("update_season_data.py", "Step 1: Rescraping 2025/2026 season data")
    if not success:
        print("\n⚠️  Warning: Data scraping may have failed, but continuing...")
    
    # Step 2: Train weighted model and generate predictions
    success = run_script("production_predictions_weighted.py", "Step 2: Training weighted model and generating predictions")
    if not success:
        print("\n❌ Failed to generate predictions. Exiting.")
        return
    
    # Step 3: Create updated predictions.txt
    print("\n" + "=" * 80)
    print("Step 3: Creating updated predictions.txt file")
    print("=" * 80)
    create_updated_predictions_text()
    
    print("\n" + "=" * 80)
    print("🎉 Full update pipeline completed!")
    print("=" * 80)
    print("📅 Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\n✅ Your predictions have been updated with:")
    print("   - Latest 2025/2026 season results")
    print("   - Weighted model (current season matches weighted 3-4x more)")
    print("   - Updated predictions.txt file")

if __name__ == "__main__":
    main()


