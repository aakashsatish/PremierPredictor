# Premier League Tracker 🏆

I created a machine learning system that predicts Premier League match outcomes using historical data. The idea came to me when I was watching the games and wondering if I could use my software engineering skills to predict results. So I built this - it scrapes match data from fbref.com and uses a Random Forest model to predict who's likely to win or draw/tie.

## Tech Stack I Used

### Core Languages & Tools
- **Python 3.7+** - My go-to language for this project
- **Jupyter Notebook** - Was good for experimenting with data and models
- **Git** - Keeping track of all my changes

### Machine Learning & Data Science
- **scikit-learn** - Used Random Forest classifier (chose this because it's great for tabular data)
- **pandas** - Essential for cleaning and manipulating all that match data
- **numpy** - For all the numerical operations and calculations

### Web Scraping
- **ScraperAPI** - This helped for webscraping especially with anti-bot systems
- **Beautiful Soup 4** - For parsing the HTML and extracting the data I need
- **requests** - Making HTTP calls to get the data

### Data Storage
- **CSV files** - Simple but effective for storing all the match data
- **fbref.com** - Amazing source for football statistics

### Key Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
beautifulsoup4>=4.10.0
requests>=2.28.0
```

## What I Built

- **Data Scraping**: Automated system that pulls PL match data from fbref.com
- **Machine Learning Model**: Random Forest classifier that actually works pretty well!
- **Advanced Analytics**: I included stuff like Expected Goals (xG), possession stats, and rolling averages
- **Season Predictions**: Ready to predict the 2025-2026 season
- **Easy to Use**: Simple functions to predict individual matches or entire fixture lists

## Project Structure

```
EPL_Tracker/
├── scraping.ipynb      # Where I scrape data from fbref.com
├── predictions.ipynb   # My ML model training and predictions
├── matches.csv         # All the historical data I collected (3800+ matches!)
├── requirements.txt    # Python packages you'll need
└── README.md          # This file
```

## Getting Started

1. **Clone this repo**
   ```bash
   git clone <repository-url>
   cd EPL_Tracker
   ```

2. **Install the packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up ScraperAPI** (you'll need this for data scraping)
   - Sign up at [ScraperAPI](https://www.scraperapi.com/) (they have a free tier which I used)
   - Update the `api_key` variable in `scraping.ipynb`

## How to Use It

### For 2025-2026 Season Predictions

The models are already trained and ready to go! Here's how:

1. **Open the prediction notebook**
   ```bash
   jupyter notebook predictions.ipynb
   ```

2. **Run the cells** to load everything up

3. **Start predicting!**
   ```python
   # Predict a single match
   result = predict_match("Arsenal", "Chelsea", "Home", "2025-08-15", "15:00")
   print(f"Win probability: {result['win_probability']:.2%}")
   
   # Predict multiple fixtures at once
   fixtures_df = pd.DataFrame({
       'team': ['Arsenal', 'Liverpool'],
       'opponent': ['Chelsea', 'Manchester City'],
       'venue': ['Home', 'Away'],
       'date': ['2025-08-15', '2025-08-16'],
       'time': ['15:00', '17:30']
   })
   predictions = predict_fixtures(fixtures_df)
   ```

### Updating Data for New Seasons

When new season data comes out:

1. **Update the scraping notebook** with the new season years
2. **Run scraping.ipynb** to get fresh data
3. **Re-run predictions.ipynb** to retrain the models

## What My Model Looks At

I trained the model to consider:

- **Basic stuff**: Home/away, opponent, match time, day of the week
- **Advanced metrics**: Expected goals difference, shot accuracy, possession efficiency
- **Recent form**: 3-match rolling averages for all performance stats
- **Seasonal factors**: Team formations, what month of the season it is

## Why Random Forest?

I chose Random Forest because:
- It handles tabular data really well (which is exactly what we have)
- It's pretty robust and doesn't overfit easily
- It can handle missing values and different types of features
- Plus, it gives you feature importance, which is cool for understanding what actually matters

## Where I Got the Data

- **fbref.com**: Amazing site with detailed match statistics
- **ScraperAPI**: Makes web scraping actually reliable
- **Historical Data**: I collected data from 2021-2025 seasons (over 3800 matches!)

## My Workflow

1. **Data Collection** → Scrape match data using `scraping.ipynb`
2. **Data Processing** → Clean it up and calculate features
3. **Model Training** → Train the Random Forest model
4. **Predictions** → Use the model on new fixtures
5. **Results** → Get win probabilities and predictions

## Key Functions I Built

- `predict_match()`: Predict a single match outcome
- `predict_fixtures()`: Predict multiple fixtures at once
- `get_latest_team_stats()`: Get recent form for any team
- `rolling_averages()`: Calculate how teams are trending

## What You Need

- Python 3.7+
- All the packages listed in requirements.txt
- A ScraperAPI account (free tier works fine)

## Some Notes

- The models are trained on data from 2020-2025
- I made sure it handles different ways team names might be written
- The rolling averages make sure recent form gets weighted properly

## What I Want to Add Next

- Real-time fixture updates
- Player performance analysis (player form)
- Injury/suspension tracking (cards)
- Compare with betting odds
- Maybe build a web interface (so others can use it too!)

---

**Ready to predict the 2025-2026 season! 🚀**
