#!/usr/bin/env python3
"""
EPL Tracker - Improved 2025-2026 Predictions
Uses proper probability calculations and realistic prediction logic
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import warnings
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

def scrape_real_2025_2026_fixtures():
    """Scrape actual 2025-2026 Premier League fixtures from fbref.com"""
    print("🔍 Scraping real 2025-2026 Premier League fixtures...")

    # Correct URL for the fixtures
    fixtures_url = "https://fbref.com/en/comps/9/2025-2026/schedule/2025-2026-Premier-League-Scores-and-Fixtures"

    try:
        # Get the fixtures page
        data = scrape_with_scraperapi(fixtures_url)

        if data.status_code != 200:
            print(f"❌ HTTP Error: {data.status_code}")
            return None

        soup = BeautifulSoup(data.text, "html.parser")

        # Find the fixtures table
        fixtures_table = soup.find('table', {'id': 'sched_2025-2026_9'})

        if not fixtures_table:
            print("❌ Could not find fixtures table with ID 'sched_2025-2026_9'")
            # Try alternative approaches
            tables = soup.find_all('table')
            for table in tables:
                if 'sched' in str(table.get('id', '')) or 'fixture' in str(table.get('class', '')):
                    fixtures_table = table
                    print(f"✅ Found alternative table: {table.get('id', 'no-id')}")
                    break

        if not fixtures_table:
            print("❌ No fixtures table found")
            return None

        # Parse the fixtures table
        fixtures_df = pd.read_html(str(fixtures_table))[0]

        print(f"📊 Parsed table with {len(fixtures_df)} rows and columns: {list(fixtures_df.columns)}")

        # Clean up the data
        fixtures_df.columns = [col.lower() for col in fixtures_df.columns]
        print(f"📋 Columns after cleaning: {list(fixtures_df.columns)}")

        # Extract the key information
        processed_fixtures = []

        for idx, row in fixtures_df.iterrows():
            try:
                # Extract match information
                wk = row.get('wk', '')
                day = row.get('day', '')
                date = row.get('date', '')
                time = row.get('time', '')

                # Extract home and away teams
                home_team = row.get('home', '')
                away_team = row.get('away', '')

                # Clean up team names (remove links and extra text)
                if isinstance(home_team, str):
                    home_team = re.sub(r'\[.*?\]', '', home_team).strip()
                if isinstance(away_team, str):
                    away_team = re.sub(r'\[.*?\]', '', away_team).strip()

                # Skip if we don't have essential data
                if not all([date, time, home_team, away_team]):
                    continue

                # Parse date
                try:
                    # Extract date from the date column
                    if isinstance(date, str):
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date)
                        if date_match:
                            date = date_match.group(1)
                        else:
                            # Try to parse different date formats
                            date_obj = pd.to_datetime(date, errors='coerce')
                            if pd.notna(date_obj):
                                date = date_obj.strftime('%Y-%m-%d')
                            else:
                                continue
                except:
                    continue

                processed_fixtures.append({
                    'matchweek': wk,
                    'day': day,
                    'date': date,
                    'time': time,
                    'home': home_team,
                    'away': away_team
                })

            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                continue

        if processed_fixtures:
            result_df = pd.DataFrame(processed_fixtures)
            print(f"✅ Successfully processed {len(result_df)} fixtures")
            return result_df
        else:
            print("❌ No valid fixtures found")
            return None

    except Exception as e:
        print(f"❌ Error scraping fixtures: {e}")
        return None

def create_improved_predictions(fixtures_df):
    """Create improved predictions using proper probability calculations"""
    print("🤖 Creating improved predictions...")

    # Load historical data to get team performance
    matches = pd.read_csv("matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])

    # Calculate team win rates and recent form
    team_stats = {}
    for team in matches["team"].unique():
        team_matches = matches[matches["team"] == team]
        
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

            # Improved prediction logic
            # Weight: 40% home team's home performance, 30% recent form, 20% overall strength, 10% away team's away performance
            home_strength = (
                0.4 * home_stats['home_win_rate'] +
                0.3 * home_stats['recent_win_rate'] +
                0.2 * home_stats['overall_win_rate'] +
                0.1 * (1 - away_stats['away_win_rate'])  # Away team's weakness
            )
            
            away_strength = (
                0.4 * away_stats['away_win_rate'] +
                0.3 * away_stats['recent_win_rate'] +
                0.2 * away_stats['overall_win_rate'] +
                0.1 * (1 - home_stats['home_win_rate'])  # Home team's weakness
            )

            # Home advantage boost (typically 10-15% in football)
            home_advantage = 0.12
            
            # Calculate win probability
            home_win_prob = home_strength + home_advantage
            away_win_prob = away_strength
            
            # Normalize probabilities (they should sum to less than 1, with remainder being draw probability)
            total_prob = home_win_prob + away_win_prob
            if total_prob > 0.8:  # If too high, scale down
                home_win_prob = home_win_prob * 0.8 / total_prob
                away_win_prob = away_win_prob * 0.8 / total_prob
            
            # Ensure reasonable bounds
            home_win_prob = max(0.15, min(0.75, home_win_prob))
            away_win_prob = max(0.15, min(0.75, away_win_prob))

            # Determine prediction
            if home_win_prob > away_win_prob + 0.1:  # Clear home advantage
                prediction = "Win"
                confidence = "High"
            elif home_win_prob > away_win_prob:
                prediction = "Win"
                confidence = "Low"
            else:
                prediction = "Loss/Draw"
                confidence = "High" if away_win_prob > home_win_prob + 0.1 else "Low"

            result = {
                "team": home_team,
                "opponent": away_team,
                "venue": "Home",
                "date": date,
                "time": time,
                "matchweek": matchweek,
                "day": day,
                "win_probability": home_win_prob,
                "opponent_win_probability": away_win_prob,
                "draw_probability": 1 - home_win_prob - away_win_prob,
                "prediction": prediction,
                "confidence": confidence,
                "home_overall_rate": home_stats['overall_win_rate'],
                "home_recent_rate": home_stats['recent_win_rate'],
                "home_home_rate": home_stats['home_win_rate'],
                "away_overall_rate": away_stats['overall_win_rate'],
                "away_recent_rate": away_stats['recent_win_rate'],
                "away_away_rate": away_stats['away_win_rate']
            }

            predictions.append(result)
            print(f"✅ Matchweek {matchweek} ({day}): {home_team} vs {away_team} ({date} {time}): {prediction} ({home_win_prob:.1%} vs {away_win_prob:.1%}) [{confidence}]")

        except Exception as e:
            print(f"❌ Error predicting fixture {idx}: {e}")
            continue

    return predictions

def main():
    """Main function to scrape real fixtures and generate improved predictions"""
    print("🏆 EPL Tracker - Improved 2025-2026 Season Predictions")
    print("=" * 60)

    # Scrape real fixtures
    fixtures_df = scrape_real_2025_2026_fixtures()

    if fixtures_df is None or len(fixtures_df) == 0:
        print("❌ No real fixtures found. The fixtures might not be fully loaded yet.")
        return

    print(f"\n📅 Processing {len(fixtures_df)} real fixtures...")

    # Generate predictions
    predictions = create_improved_predictions(fixtures_df)

    # Save predictions to CSV
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv("2025_2026_improved_predictions.csv", index=False)
        print(f"\n💾 Saved {len(predictions)} predictions to '2025_2026_improved_predictions.csv'")

        # Display summary
        print("\n📊 Improved Prediction Summary:")
        print(f"Total fixtures: {len(predictions)}")
        wins = len(predictions_df[predictions_df['prediction'] == 'Win'])
        print(f"Predicted wins: {wins} ({wins/len(predictions):.1%})")
        print(f"Predicted losses/draws: {len(predictions) - wins} ({(len(predictions) - wins)/len(predictions):.1%})")

        # Show confidence breakdown
        high_conf_wins = len(predictions_df[(predictions_df['prediction'] == 'Win') & (predictions_df['confidence'] == 'High')])
        low_conf_wins = len(predictions_df[(predictions_df['prediction'] == 'Win') & (predictions_df['confidence'] == 'Low')])
        print(f"High confidence wins: {high_conf_wins}")
        print(f"Low confidence wins: {low_conf_wins}")

        # Show top 10 highest win probability matches
        print("\n🔥 Top 10 Highest Win Probability Matches:")
        top_matches = predictions_df.nlargest(10, 'win_probability')
        for _, match in top_matches.iterrows():
            print(f"  Matchweek {match['matchweek']}: {match['team']} vs {match['opponent']}: {match['win_probability']:.1%}")

        # Show team performance summary
        print("\n🏆 Team Performance Summary:")
        team_summary = predictions_df.groupby('team').agg({
            'win_probability': 'mean',
            'prediction': lambda x: (x == 'Win').sum()
        }).rename(columns={'prediction': 'predicted_wins'})
        team_summary = team_summary.sort_values('win_probability', ascending=False)

        for team, row in team_summary.head(10).iterrows():
            print(f"  {team}: {row['win_probability']:.1%} avg win probability, {row['predicted_wins']} predicted wins")

    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    main() 