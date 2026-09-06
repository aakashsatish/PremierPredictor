#!/usr/bin/env python3
"""
EPL Tracker - Manual Data Update for 2025/2026 Season
Alternative approach without ScraperAPI - uses direct requests with proper headers
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
import re

def scrape_with_headers(url, max_retries=3):
    """Scrape using direct requests with proper headers and retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            print(f"📡 Scraping: {url} (Status: {response.status_code}, Attempt: {attempt + 1})")
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

def get_team_urls():
    """Get all Premier League team URLs"""
    print("🔍 Getting Premier League team URLs...")
    
    standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    
    try:
        data = scrape_with_headers(standings_url)
        
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

def scrape_team_matches(team_url):
    """Scrape matches and shooting data for a specific team"""
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    print(f"📊 Scraping {team_name}...")
    
    try:
        # Get team page
        data = scrape_with_headers(team_url)
        
        if data is None:
            print(f"❌ Failed to get team page for {team_name}")
            return None
        
        # Parse matches table
        try:
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        except ValueError:
            print(f"❌ Could not find matches table for {team_name}")
            return None
        
        # Get shooting data
        soup = BeautifulSoup(data.text, "html.parser")
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        
        if not links:
            print(f"❌ Could not find shooting data link for {team_name}")
            return None
        
        # Scrape shooting data
        shooting_data = scrape_with_headers(f"https://fbref.com{links[0]}")
        
        if shooting_data is None:
            print(f"❌ Failed to get shooting data for {team_name}")
            return None
        
        try:
            shooting = pd.read_html(shooting_data.text, match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()
        except ValueError:
            print(f"❌ Could not parse shooting data for {team_name}")
            return None
        
        # Merge matches with shooting data
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            print(f"❌ Could not merge data for {team_name}")
            return None
        
        # Filter for Premier League matches only
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        # Add season and team information
        team_data["Season"] = 2025
        team_data["Team"] = team_name
        
        # Filter for 2025/2026 season matches (from August 2025 onwards)
        team_data["Date"] = pd.to_datetime(team_data["Date"])
        team_data = team_data[team_data["Date"] >= "2025-08-01"]
        
        # Only include completed matches (those with results)
        team_data = team_data[team_data["Result"].notna()]
        
        print(f"✅ Scraped {len(team_data)} matches for {team_name}")
        return team_data
        
    except Exception as e:
        print(f"❌ Error scraping {team_name}: {e}")
        return None

def create_manual_update():
    """Create a manual update approach for when ScraperAPI is not available"""
    print("🚀 Starting manual 2025/2026 season match update...")
    print("=" * 60)
    print("⚠️  Note: This method uses direct requests and may be slower")
    print("📅 Current date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Get team URLs
    team_urls = get_team_urls()
    
    if not team_urls:
        print("❌ No team URLs found. Exiting.")
        return False
    
    all_matches = []
    
    # Scrape each team with generous rate limiting
    for i, team_url in enumerate(team_urls):
        print(f"\n📊 Processing team {i+1}/{len(team_urls)}")
        team_data = scrape_team_matches(team_url)
        
        if team_data is not None and len(team_data) > 0:
            all_matches.append(team_data)
        
        # Generous rate limiting to avoid being blocked
        print("⏳ Waiting 5 seconds between teams...")
        time.sleep(5)
    
    if not all_matches:
        print("❌ No match data scraped. Exiting.")
        return False
    
    # Combine all team data
    new_matches_df = pd.concat(all_matches, ignore_index=True)
    
    # Clean column names
    new_matches_df.columns = [c.lower() for c in new_matches_df.columns]
    
    print(f"\n📊 Scraped {len(new_matches_df)} total matches")
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
    print("🚀 EPL Tracker - Manual 2025/2026 Season Update")
    print("=" * 60)
    print("⚠️  ScraperAPI quota exceeded - using alternative method")
    print("📅 Current date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Target: Update matches.csv with latest 2025/2026 season data")
    print("=" * 60)
    
    success = create_manual_update()
    
    if success:
        print("\n🎉 Manual update completed successfully!")
        print("📊 Your matches.csv file has been updated with the latest data.")
        print("🤖 You can now run your prediction models with the updated data.")
        print("\n💡 Tip: Your ScraperAPI quota will reset on November 6th")
    else:
        print("\n❌ Manual update failed. Please check the error messages above.")
        print("💡 You may need to wait until November 6th for ScraperAPI quota reset")

if __name__ == "__main__":
    main()
