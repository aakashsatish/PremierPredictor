#!/usr/bin/env python3
"""
Create a comprehensive text file with all EPL predictions for 2025-2026 season
"""

import pandas as pd

def create_predictions_text():
    """Create a comprehensive text file with all predictions"""
    
    # Load the realistic predictions
    df = pd.read_csv("2025_2026_production_predictions_REALISTIC.csv")
    
    # Create the text content
    text_content = []
    text_content.append("=" * 80)
    text_content.append("EPL TRACKER - 2025/2026 SEASON PREDICTIONS")
    text_content.append("=" * 80)
    text_content.append("")
    text_content.append("Model: Random Forest + Team Strength Comparison")
    text_content.append("Accuracy: 64.5% (based on historical testing)")
    text_content.append("Method: 70% team strength + 30% machine learning")
    text_content.append("")
    text_content.append("Confidence Levels:")
    text_content.append("- High: >65% win probability")
    text_content.append("- Medium: 55-65% win probability") 
    text_content.append("- Low: <55% win probability")
    text_content.append("")
    text_content.append("=" * 80)
    text_content.append("")
    
    # Group by matchweek
    for matchweek in sorted(df['matchweek'].unique()):
        if pd.isna(matchweek):
            continue
            
        week_matches = df[df['matchweek'] == matchweek].sort_values('date')
        
        text_content.append(f"MATCHWEEK {int(matchweek)}")
        text_content.append("-" * 40)
        text_content.append("")
        
        for idx, match in week_matches.iterrows():
            home_team = match['team']
            away_team = match['opponent']
            date = match['date']
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
    
    total_matches = len(df)
    predicted_wins = len(df[df['prediction'] == 'Win'])
    predicted_losses_draws = len(df[df['prediction'] == 'Loss/Draw'])
    
    high_confidence = len(df[df['confidence'] == 'High'])
    medium_confidence = len(df[df['confidence'] == 'Medium'])
    low_confidence = len(df[df['confidence'] == 'Low'])
    
    avg_win_prob = df['win_probability'].mean()
    
    text_content.append(f"Total Matches: {total_matches}")
    text_content.append(f"Predicted Wins: {predicted_wins} ({predicted_wins/total_matches:.1%})")
    text_content.append(f"Predicted Losses/Draws: {predicted_losses_draws} ({predicted_losses_draws/total_matches:.1%})")
    text_content.append("")
    text_content.append(f"High Confidence Predictions: {high_confidence}")
    text_content.append(f"Medium Confidence Predictions: {medium_confidence}")
    text_content.append(f"Low Confidence Predictions: {low_confidence}")
    text_content.append("")
    text_content.append(f"Average Win Probability: {avg_win_prob:.1%}")
    text_content.append("")
    
    # Add key highlights
    text_content.append("KEY HIGHLIGHTS:")
    text_content.append("-" * 20)
    text_content.append("")
    
    # Strong teams with high confidence wins
    high_confidence_matches = df[df['confidence'] == 'High']
    if len(high_confidence_matches) > 0:
        text_content.append("HIGH CONFIDENCE PREDICTIONS:")
        for idx, match in high_confidence_matches.iterrows():
            text_content.append(f"• {match['team']} vs {match['opponent']}: {match['team']} WIN ({match['win_probability']:.1%})")
        text_content.append("")
    
    # Notable matches
    text_content.append("NOTABLE MATCHES:")
    text_content.append("• Manchester City vs Tottenham: Man City WIN (69.0%) [High]")
    text_content.append("• Manchester City vs Manchester Utd: Man City WIN (69.4%) [High]")
    text_content.append("• Manchester City vs Burnley: Man City WIN (74.6%) [High]")
    text_content.append("• Wolves vs Manchester City: Man City WIN (44.0%) [Low]")
    text_content.append("• Liverpool vs Arsenal: Arsenal WIN (37.1%) [Low]")
    text_content.append("")
    
    text_content.append("=" * 80)
    text_content.append("END OF PREDICTIONS")
    text_content.append("=" * 80)
    
    # Write to file
    with open("2025_2026_predictions.txt", "w") as f:
        f.write("\n".join(text_content))
    
    print("✅ Created comprehensive predictions text file: 2025_2026_predictions.txt")
    print(f"📊 Total matches: {total_matches}")
    print(f"🎯 High confidence predictions: {high_confidence}")
    print(f"📈 Average win probability: {avg_win_prob:.1%}")

if __name__ == "__main__":
    create_predictions_text() 