# 🚀 EPL Tracker - Deployment Guide

This guide covers how to deploy and run the EPL Tracker production system.

## 📋 Prerequisites

- Python 3.7+
- pip package manager
- Git
- ScraperAPI account (free tier available)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd EPL_Tracker
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Update the `SCRAPER_API_KEY` in `config.py`:
```python
SCRAPER_API_KEY = "your_api_key_here"
```

## 🎯 Production Deployment

### Quick Start
```bash
# Run production predictions
python production_predictions.py

# Test model accuracy
python test_model_accuracy.py

# Validate configuration
python config.py
```

### Expected Output
- `2025_2026_production_predictions.csv` - Complete season predictions
- Console output with prediction summary
- Accuracy metrics and confidence levels

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 2GB
- **Storage**: 100MB
- **CPU**: 1 core
- **Network**: Internet connection for data scraping

### Recommended Requirements
- **RAM**: 4GB+
- **Storage**: 500MB+
- **CPU**: 2+ cores
- **Network**: Stable internet connection

## 🔧 Configuration

### Model Parameters
Edit `config.py` to adjust:
- Random Forest parameters
- Confidence thresholds
- Feature engineering settings
- Team name mappings

### Data Sources
- Historical data: `matches.csv`
- Fixtures source: `2025_2026_improved_predictions.csv`
- Output: `2025_2026_production_predictions.csv`

## 📈 Monitoring

### Log Files
- Application logs: `epl_tracker.log`
- Error tracking: Check console output
- Performance metrics: Accuracy analysis in `ACCURACY_ANALYSIS.md`

### Key Metrics
- Model accuracy: 64.5%
- Prediction distribution: 84.7% wins, 15.3% losses/draws
- Average win probability: 50.9%

## 🔄 Maintenance

### Regular Updates
1. **Data Updates**: Run scraping for new season data
2. **Model Retraining**: Retrain on updated historical data
3. **Accuracy Validation**: Test on latest season results
4. **Configuration Review**: Update team mappings and settings

### Backup Strategy
- Version control: All code in Git
- Data backup: CSV files in repository
- Configuration: Settings in `config.py`

## 🚨 Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```bash
pip install -r requirements.txt
```

#### 2. API Key Issues
- Verify ScraperAPI key in `config.py`
- Check API quota and limits
- Test with simple request

#### 3. Data File Missing
```bash
# Check if matches.csv exists
ls -la matches.csv

# Re-download if needed
python scraping.ipynb
```

#### 4. Memory Issues
- Increase system RAM
- Reduce batch size in predictions
- Use smaller rolling window

### Error Codes
- `FileNotFoundError`: Missing data files
- `KeyError`: Missing columns in data
- `ValueError`: Invalid configuration
- `ConnectionError`: Network/API issues

## 🔒 Security

### API Key Management
- Store API keys in environment variables
- Never commit keys to version control
- Use separate keys for development/production

### Data Privacy
- Historical match data is public
- No personal information collected
- Predictions are statistical only

## 📚 Documentation

### Key Files
- `README.md` - Project overview
- `ACCURACY_ANALYSIS.md` - Model performance
- `ENHANCED_IMPLEMENTATION_SUMMARY.md` - Technical details
- `config.py` - Configuration settings

### Support
- Check documentation files
- Review error logs
- Validate configuration with `python config.py`

## 🎯 Performance Optimization

### Speed Improvements
- Use smaller rolling windows
- Reduce feature set
- Cache model predictions
- Parallel processing for large datasets

### Accuracy Improvements
- Add more historical data
- Feature engineering improvements
- Ensemble methods
- Hyperparameter tuning

## 🔄 CI/CD Integration

### Automated Testing
```bash
# Run all tests
python test_model_accuracy.py
python config.py
python production_predictions.py
```

### Deployment Pipeline
1. Code validation
2. Configuration check
3. Model training
4. Accuracy testing
5. Production deployment

---

**Ready for production deployment! 🚀** 