#!/usr/bin/env python3
"""
EPL Tracker - Robust Manual Update
Handles rate limiting and gets all 2025/2026 season matches
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
import random

def robust_request(url, max_retries=5):
    """Robust request with retry logic and random delays"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(max_retries):
        try:
            # Random delay to avoid rate limiting
            delay = random.uniform(2, 5)
            time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=30)
            print(f"📡 {url.split('/')[-1]} - Status: {response.status_code} (Attempt: {attempt + 1})")
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(1, 3)
                print(f"⏳ Rate limited, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(3, 6))
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(3, 6))
    
    return None

def get_all_team_urls():
    """Get all Premier League team URLs"""
    print("🔍 Getting all Premier League team URLs...")
    
    standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    
    try:
        data = robust_request(standings_url)
        
        if data is None:
            print("❌ Failed to get standings page")
            return []
        
        soup = BeautifulSoup(data.text, "html.parser")
        standings_table = soup.select('table.stats_table')[0]
        
        links = [l.get("href") for l in standings_table.find_all('a')]
        links = [l for l in links if '/squads/' in l]
        team_urls = [f"https://fbref.com{l}" for l in links]
        
        print(f"✅ Found {len(team_urls)} Premier League teams")
        return team_urls
        
    except Exception as e:
        print(f"❌ Error getting team URLs: {e}")
        return []

def scrape_team_data(team_url):
    """Scrape data for a single team"""
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    print(f"\n📊 Processing {team_name}...")
    
    try:
        # Get team page
        data = robust_request(team_url)
        
        if data is None:
            print(f"❌ Failed to get team page for {team_name}")
            return None
        
        # Parse matches table
        try:
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        except ValueError:
            print(f"❌ Could not find matches table for {team_name}")
            return None
        
        # Filter for Premier League matches
        matches = matches[matches["Comp"] == "Premier League"]
        
        # Add team info
        matches["Team"] = team_name
        matches["Season"] = 2025
        
        # Filter for 2025/2026 season matches (from August 2025 onwards)
        matches["Date"] = pd.to_datetime(matches["Date"])
        recent_matches = matches[matches["Date"] >= "2025-08-01"]
        recent_matches = recent_matches[recent_matches["Result"].notna()]
        
        print(f"✅ Found {len(recent_matches)} recent completed matches for {team_name}")
        return recent_matches
        
    except Exception as e:
        print(f"❌ Error scraping {team_name}: {e}")
        return None

def update_matches_csv():
    """Update matches.csv with new 2025/2026 season data"""
    print("🚀 Starting robust 2025/2026 season match update...")
    print("=" * 60)
    print("⚠️  Using direct requests with rate limiting protection")
    print("📅 Current date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Get team URLs
    team_urls = get_all_team_urls()
    
    if not team_urls:
        print("❌ No team URLs found. Exiting.")
        return False
    
    all_matches = []
    successful_teams = 0
    
    # Process each team
    for i, team_url in enumerate(team_urls):
        print(f"\n📊 Processing team {i+1}/{len(team_urls)}")
        team_data = scrape_team_data(team_url)
        
        if team_data is not None and len(team_data) > 0:
            all_matches.append(team_data)
            successful_teams += 1
        
        # Progress update
        if (i + 1) % 5 == 0:
            print(f"\n📈 Progress: {i+1}/{len(team_urls)} teams processed, {successful_teams} successful")
    
    if not all_matches:
        print("❌ No match data scraped. Exiting.")
        return False
    
    # Combine all team data
    new_matches_df = pd.concat(all_matches, ignore_index=True)
    
    # Clean column names
    new_matches_df.columns = [c.lower() for c in new_matches_df.columns]
    
    print(f"\n📊 Successfully scraped {len(new_matches_df)} total matches from {successful_teams} teams")
    print(f"📅 Date range: {new_matches_df['date'].min()} to {new_matches_df['date'].max()}")
    
    # Load existing matches.csv
    existing_file = "matches.csv"
    backup_file = f"matches_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if os.path.exists(existing_file):
        print(f"📁 Loading existing matches.csv...")
        existing_df = pd.read_csv(existing_file)
        
        # Create backup
        existing_df.to_csv(backup_file, index=False)
        print(f"💾 Created backup: {backup_file}")
        
        # Remove any existing 2025/2026 season data to avoid duplicates
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        existing_df = existing_df[existing_df["date"] < "2025-08-01"]
        
        # Combine with new data
        combined_df = pd.concat([existing_df, new_matches_df], ignore_index=True)
        
    else:
        print("📁 No existing matches.csv found, creating new file...")
        combined_df = new_matches_df
    
    # Sort by date
    combined_df = combined_df.sort_values("date")
    
    # Save updated file
    combined_df.to_csv(existing_file, index=False)
    
    print(f"\n✅ Successfully updated {existing_file}")
    print(f"📊 Total matches: {len(combined_df)}")
    print(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Show summary of new data
    new_teams = new_matches_df['team'].unique()
    print(f"\n🏆 Teams with new data: {len(new_teams)}")
    for team in sorted(new_teams):
        team_matches = len(new_matches_df[new_matches_df['team'] == team])
        print(f"   {team}: {team_matches} matches")
    
    return True

def main():
    """Main function"""
    print("🚀 EPL Tracker - Robust Manual Update")
    print("=" * 60)
    print("⚠️  ScraperAPI quota exceeded - using robust alternative")
    print("📅 Current date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Target: Update matches.csv with latest 2025/2026 season data")
    print("=" * 60)
    
    success = update_matches_csv()
    
    if success:
        print("\n🎉 Robust update completed successfully!")
        print("📊 Your matches.csv file has been updated with the latest data.")
        print("🤖 You can now run your prediction models with the updated data.")
        print("\n💡 Tip: Your ScraperAPI quota will reset on November 6th")
    else:
        print("\n❌ Robust update failed. Please check the error messages above.")
        print("💡 You may need to wait until November 6th for ScraperAPI quota reset")

if __name__ == "__main__":
    main()
