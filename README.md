# Premier League Tracker 🏆
NOTE: Waiting for the 2025-2026 season to release predictions

A machine learning project for predicting Premier League match outcomes using historical data and advanced statistical modeling.

## Overview

This project scrapes PL match data from fbref.com and uses machine learning models (Random Forest & XGBoost) to predict match outcomes. The system analyzes team performance metrics, rolling averages, and other statistics to make predictions.

## Tech Stack

### Programming Languages
- **Python 3.7+** - Dev language 

### Machine Learning & Data
- **scikit-learn** - Random Forest classifier and model evaluation
- **XGBoost** - Advanced gradient boosting algorithm
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing and array operations

### Web Scraping & APIs
- **ScraperAPI** - Reliable web scraping service
- **Beautiful Soup 4** - HTML parsing and data extraction
- **requests** - HTTP library for API calls

### Development Environment
- **Jupyter Notebook** - Interactive development and analysis
- **Git** - Version control system

### Data Sources
- **fbref.com** - Football statistics and match data
- **CSV files** - Data storage and processing

### Key Libraries
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
beautifulsoup4>=4.10.0
requests>=2.28.0
```

## Features

- **Data Scraping**: Automated scraping of EPL match data from fbref.com
- **Machine Learning**: Dual-model approach using Random Forest and XGBoost
- **Advanced Analytics**: Expected Goals (xG), possession efficiency, and rolling averages
- **Season Predictions**: Ready for 2025-2026 season predictions
- **Easy Usage**: Simple functions to predict individual matches or entire fixture lists

## Project Structure

```
EPL_Tracker/
├── scraping.ipynb      # Data scraping from fbref.com
├── predictions.ipynb   # Model training and predictions
├── matches.csv         # Historical match data (3800+ matches)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EPL_Tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up ScraperAPI** (for data scraping)
   - Get an API key from [ScraperAPI](https://www.scraperapi.com/)
   - Update the `api_key` variable in `scraping.ipynb`

## Usage

### For 2025-2026 Season Predictions

The models are already trained and ready! Simply:

1. **Run the prediction notebook**
   ```bash
   jupyter notebook predictions.ipynb
   ```

2. **Execute the cells** to load models and prediction functions

3. **Make predictions**
   ```python
   # Single match prediction
   result = predict_match("Arsenal", "Chelsea", "Home", "2025-08-15", "15:00")
   print(f"Win probability: {result['win_probability']:.2%}")
   
   # Batch predictions for fixtures
   fixtures_df = pd.DataFrame({
       'team': ['Arsenal', 'Liverpool'],
       'opponent': ['Chelsea', 'Manchester City'],
       'venue': ['Home', 'Away'],
       'date': ['2025-08-15', '2025-08-16'],
       'time': ['15:00', '17:30']
   })
   predictions = predict_fixtures(fixtures_df)
   ```

### For Data Updates

When new season data is available:

1. **Update scraping notebook** with new season years
2. **Run scraping.ipynb** to collect new data
3. **Re-run predictions.ipynb** to retrain models with updated data

## Model Features

The prediction models use these key features:

- **Basic**: Venue (home/away), opponent, time, day of week
- **Advanced**: Expected goals difference, shots accuracy, possession efficiency
- **Rolling Stats**: 3-match rolling averages for all performance metrics
- **Seasonal**: Formation codes, season stage (month)

## Model Performance

The system uses ensemble predictions from:
- **Random Forest**: Robust baseline model
- **XGBoost**: Advanced gradient boosting (when available)
- **Ensemble**: Combines both models for improved accuracy

## Data Sources

- **fbref.com**: Match results, fixtures, and detailed statistics
- **ScraperAPI**: Reliable web scraping service
- **Historical Data**: 2021-2025 seasons (3800+ matches)

## Workflow

1. **Data Collection** → `scraping.ipynb` scrapes match data
2. **Data Processing** → Convert to features, calculate rolling averages
3. **Model Training** → Train Random Forest & XGBoost models
4. **Predictions** → Use trained models on new fixtures
5. **Results** → Win probabilities and match predictions

## Key Functions

- `predict_match()`: Predict a single match outcome
- `predict_fixtures()`: Batch predict multiple fixtures
- `get_latest_team_stats()`: Get rolling averages for any team
- `rolling_averages()`: Calculate team performance trends

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- xgboost (this is optional)
- beautifulsoup4, requests
- jupyter notebook

## Notes

- Models are trained on data from 2020 - 2025
- The system handles team name variations automatically
- Rolling averages ensure recent form is weighted appropriately

## Future Enhancements

- Real-time fixture updates
- Player performance analysis
- Injury/suspension tracking
- Betting odds comparison
- Front-end implmentation
---

**Ready for the 2025-2026 season! 🚀** 
