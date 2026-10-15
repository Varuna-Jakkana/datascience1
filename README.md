# BikeShare Analytics

An interactive Streamlit dashboard for exploring bike-sharing demand and comparing hourly rental prediction models.

## Features

- Overview of rental demand, seasonal patterns, and working-day trends
- Demand analysis with season and day-type filters
- Weather summaries and numeric feature correlations
- Dataset explorer with filtered CSV downloads
- Hourly demand predictions from linear regression and random forest models
- Model evaluation using chronological training, validation, and test periods

## Requirements

- Python 3.11 or newer
- Internet access to download the UCI Bike Sharing Dataset on first run

If the dataset cannot be downloaded, place a valid `hour.csv` file beside `app.py` to use the local fallback.

## Setup

From this project directory, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`, when the app starts.

## Prediction inputs

Temperature, feeling temperature, humidity, and wind speed use the normalized values from the source dataset. Calendar choices must be consistent: the month must match the season, and day type must agree with the selected weekday and holiday status.

Predictions are estimates based on the available UCI data and should not be treated as guaranteed future demand.