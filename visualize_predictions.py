#!/usr/bin/env python3
"""
EPL Tracker - Prediction Visualizations
Creates charts and graphs for the 2025-2026 predictions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_predictions():
    """Load the predictions data"""
    df = pd.read_csv("2025_2026_working_predictions.csv")
    return df

def create_team_performance_chart(df):
    """Create a bar chart of team performance"""
    print("📊 Creating team performance chart...")
    
    # Calculate team statistics
    team_stats = df.groupby('team').agg({
        'win_probability': 'mean',
        'prediction': lambda x: (x == 'Win').sum()
    }).rename(columns={'prediction': 'predicted_wins'})
    
    team_stats = team_stats.sort_values('win_probability', ascending=True)
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Win probability chart
    bars1 = ax1.barh(range(len(team_stats)), team_stats['win_probability'])
    ax1.set_yticks(range(len(team_stats)))
    ax1.set_yticklabels(team_stats.index)
    ax1.set_xlabel('Average Win Probability')
    ax1.set_title('Team Performance: Average Win Probability')
    ax1.grid(axis='x', alpha=0.3)
    
    # Color bars based on performance
    for i, bar in enumerate(bars1):
        if team_stats.iloc[i]['win_probability'] > 0.4:
            bar.set_color('green')
        elif team_stats.iloc[i]['win_probability'] > 0.25:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Predicted wins chart
    bars2 = ax2.barh(range(len(team_stats)), team_stats['predicted_wins'])
    ax2.set_yticks(range(len(team_stats)))
    ax2.set_yticklabels(team_stats.index)
    ax2.set_xlabel('Predicted Wins')
    ax2.set_title('Team Performance: Predicted Wins')
    ax2.grid(axis='x', alpha=0.3)
    
    # Color bars based on wins
    for i, bar in enumerate(bars2):
        if team_stats.iloc[i]['predicted_wins'] > 5:
            bar.set_color('green')
        elif team_stats.iloc[i]['predicted_wins'] > 0:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('team_performance_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: team_performance_chart.png")

def create_matchweek_analysis(df):
    """Create matchweek analysis chart"""
    print("📅 Creating matchweek analysis chart...")
    
    # Calculate matchweek statistics
    matchweek_stats = df.groupby('matchweek').agg({
        'prediction': lambda x: (x == 'Win').sum(),
        'win_probability': 'mean'
    }).rename(columns={'prediction': 'predicted_wins'})
    
    matchweek_stats['total_matches'] = 10  # Each matchweek has 10 matches
    matchweek_stats['win_rate'] = matchweek_stats['predicted_wins'] / matchweek_stats['total_matches']
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Predicted wins per matchweek
    ax1.bar(matchweek_stats.index, matchweek_stats['predicted_wins'], 
            color='skyblue', alpha=0.7)
    ax1.set_xlabel('Matchweek')
    ax1.set_ylabel('Predicted Wins')
    ax1.set_title('Predicted Wins by Matchweek')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(range(1, 39, 2))
    
    # Average win probability per matchweek
    ax2.plot(matchweek_stats.index, matchweek_stats['win_probability'], 
             marker='o', linewidth=2, markersize=4, color='red')
    ax2.set_xlabel('Matchweek')
    ax2.set_ylabel('Average Win Probability')
    ax2.set_title('Average Win Probability by Matchweek')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(range(1, 39, 2))
    
    plt.tight_layout()
    plt.savefig('matchweek_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: matchweek_analysis.png")

def create_win_probability_distribution(df):
    """Create win probability distribution chart"""
    print("📈 Creating win probability distribution chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of win probabilities
    ax1.hist(df['win_probability'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Win Probability')
    ax1.set_ylabel('Number of Matches')
    ax1.set_title('Distribution of Win Probabilities')
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot by prediction
    df.boxplot(column='win_probability', by='prediction', ax=ax2)
    ax2.set_xlabel('Prediction')
    ax2.set_ylabel('Win Probability')
    ax2.set_title('Win Probability by Prediction Type')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('win_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: win_probability_distribution.png")

def create_top_matches_chart(df):
    """Create chart of top matches"""
    print("🔥 Creating top matches chart...")
    
    # Get top 15 highest win probability matches
    top_matches = df.nlargest(15, 'win_probability')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create match labels
    match_labels = [f"{row['team']} vs {row['opponent']}\n(MW {row['matchweek']})" 
                   for _, row in top_matches.iterrows()]
    
    bars = ax.barh(range(len(top_matches)), top_matches['win_probability'])
    ax.set_yticks(range(len(top_matches)))
    ax.set_yticklabels(match_labels)
    ax.set_xlabel('Win Probability')
    ax.set_title('Top 15 Highest Win Probability Matches')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', ha='left', va='center')
    
    # Color bars
    for i, bar in enumerate(bars):
        if top_matches.iloc[i]['win_probability'] > 0.6:
            bar.set_color('green')
        elif top_matches.iloc[i]['win_probability'] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')
    
    plt.tight_layout()
    plt.savefig('top_matches_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: top_matches_chart.png")

def create_summary_statistics(df):
    """Create summary statistics table"""
    print("📋 Creating summary statistics...")
    
    # Calculate summary statistics
    total_matches = len(df)
    predicted_wins = len(df[df['prediction'] == 'Win'])
    predicted_losses = len(df[df['prediction'] == 'Loss/Draw'])
    avg_win_prob = df['win_probability'].mean()
    
    # Team statistics
    team_stats = df.groupby('team').agg({
        'win_probability': 'mean',
        'prediction': lambda x: (x == 'Win').sum()
    }).rename(columns={'prediction': 'predicted_wins'})
    
    # Create summary text
    summary_text = f"""
EPL Tracker - 2025-2026 Season Predictions Summary
==================================================

Overall Statistics:
- Total Matches: {total_matches:,}
- Predicted Wins: {predicted_wins:,} ({predicted_wins/total_matches:.1%})
- Predicted Losses/Draws: {predicted_losses:,} ({predicted_losses/total_matches:.1%})
- Average Win Probability: {avg_win_prob:.1%}

Top 5 Teams by Win Probability:
{team_stats.nlargest(5, 'win_probability')[['win_probability', 'predicted_wins']].to_string()}

Bottom 5 Teams by Win Probability:
{team_stats.nsmallest(5, 'win_probability')[['win_probability', 'predicted_wins']].to_string()}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save summary
    with open('prediction_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("✅ Saved: prediction_summary.txt")
    print(summary_text)

def main():
    """Main function to create all visualizations"""
    print("🎨 EPL Tracker - Creating Prediction Visualizations")
    print("=" * 50)
    
    # Load data
    df = load_predictions()
    print(f"📊 Loaded {len(df)} predictions")
    
    # Create visualizations
    create_team_performance_chart(df)
    create_matchweek_analysis(df)
    create_win_probability_distribution(df)
    create_top_matches_chart(df)
    create_summary_statistics(df)
    
    print("\n🎉 All visualizations created successfully!")
    print("📁 Generated files:")
    print("  - team_performance_chart.png")
    print("  - matchweek_analysis.png")
    print("  - win_probability_distribution.png")
    print("  - top_matches_chart.png")
    print("  - prediction_summary.txt")

if __name__ == "__main__":
    main() 