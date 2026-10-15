# Bike-Sharing Demand Analysis & Prediction

An interactive Streamlit dashboard for analyzing bike-sharing demand, exploring weather and seasonal patterns, and predicting hourly bike rentals using machine learning.

## Project Overview

Bike-sharing services need to understand how rental demand changes throughout the day and under different environmental conditions. This project uses the UCI Bike Sharing Dataset to analyze historical rental patterns, identify factors associated with demand, and develop machine learning models to predict hourly bike rentals.

The project combines exploratory data analysis (EDA), data visualization, machine learning, model evaluation, and an interactive web dashboard.

## Objectives

- Analyze historical bike-sharing rental demand.
- Inspect and prepare the dataset for analysis.
- Perform exploratory data analysis using visualizations.
- Study the effects of time, season, working days, and weather on demand.
- Train and compare Linear Regression and Random Forest models.
- Evaluate prediction performance using MAE, RMSE, and R².
- Develop an interactive Streamlit dashboard for data exploration and demand prediction.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Bike Sharing Dataset
- **File used:** `hour.csv`
- **Number of records:** 17,379
- **Original number of columns:** 17

Dataset link: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The dataset contains hourly bike rental counts and information about time, seasons, weather, temperature, humidity, and wind speed.

### Important Features

| Feature | Description |
|---|---|
| `season` | Season of the year |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour of the day |
| `holiday` | Holiday indicator |
| `weekday` | Day of the week |
| `workingday` | Working-day indicator |
| `weathersit` | Weather condition category |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `cnt` | Total bike rentals per hour (target variable) |

The `casual` and `registered` columns are excluded from the model inputs because they contribute directly to the target variable `cnt`. The `instant` record identifier is also excluded from the prediction features.

## Technologies Used

- **Programming language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** Scikit-learn
- **Interactive dashboard:** Streamlit
- **Development environments:** Google Colab, Visual Studio Code
- **Version control:** Git and GitHub

## Project Workflow

1. Define the bike-sharing demand prediction problem.
2. Obtain the dataset from the UCI Machine Learning Repository.
3. Inspect the data and check for missing values and duplicate records.
4. Perform exploratory data analysis and visualization.
5. Identify patterns in hourly, seasonal, working-day, and weather-related demand.
6. Select relevant features and prepare the data.
7. Split the data chronologically for model training and evaluation.
8. Train Linear Regression and Random Forest Regression models.
9. Evaluate and compare model performance.
10. Develop an interactive Streamlit dashboard.
11. Provide an interface for users to enter conditions and estimate hourly bike demand.

## Exploratory Data Analysis

The project explores:

- Hourly and daily rental demand.
- Seasonal variations in bike rentals.
- Differences between working days and non-working days.
- Relationships between weather conditions and rental demand.
- Correlations between temperature, humidity, wind speed, and rentals.

### Key Findings

The following findings were obtained during the exploratory analysis:

- **Peak demand:** The highest average hourly demand occurs around 5:00 PM (17:00).
- **Seasonal demand:** Fall has the highest average hourly rental demand among the seasons in the analyzed dataset.
- **Working days:** Average demand is approximately 193.21 rentals per hour, compared with 181.41 on non-working days.
- **Clear weather:** Clear or mostly clear conditions have an average demand of approximately 204.87 rentals per hour.
- **Heavy rain or snow:** These conditions have an average demand of approximately 74.33 rentals per hour.
- **Temperature:** Temperature has a positive correlation with rental demand of approximately 0.40.
- **Humidity:** Humidity has a negative correlation with rental demand of approximately -0.32.

These observations describe patterns in the dataset and do not necessarily establish causal relationships.

## Machine Learning Models

Two regression models are trained and compared.

### 1. Linear Regression

Linear Regression is used as a baseline model to estimate the relationship between input features and hourly bike rental demand.

### 2. Random Forest Regression

Random Forest Regression combines multiple decision trees to model relationships between time, weather, and bike rental demand.

### Model Evaluation

The recorded evaluation results are shown below.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 98.691 | 133.819 | 0.632 |
| Random Forest | 60.449 | 89.114 | 0.837 |

**Best-performing model: Random Forest Regression**

Based on these results, Random Forest achieved lower MAE and RMSE values and a higher R² score than Linear Regression.

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted rental counts.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Indicates how much of the variation in the target variable is explained by the model on the evaluation data.

*Note: These metrics are the recorded results from the project experiment. Reproducing the same data split, feature selection, preprocessing, and model configuration is necessary to verify them.*

## Streamlit Dashboard Features

The interactive application provides the following sections.

### 1. Overview
Displays an introduction to the project and a summary of bike-sharing demand patterns.

### 2. Demand Analysis
Explores hourly rental demand, seasonal patterns, and working-day versus non-working-day trends.

### 3. Weather Analysis
Examines rental demand under different weather conditions and explores relationships with environmental variables.

### 4. Data Explorer
Allows users to inspect the dataset and explore filtered records, with CSV downloads where available.

### 5. Demand Prediction
Allows users to enter time, season, weekday, holiday, weather, temperature, humidity, and wind-speed conditions to estimate hourly bike rental demand.

Users can select a prediction model and compare estimates from Linear Regression and Random Forest.

### 6. Model Evaluation
Displays model performance metrics to help users compare prediction accuracy.

## Project Structure

```text
BikeShareProject/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Bike_Sharing_Demand_Analysis_Colab (1).ipynb
└── Bike_Sharing_Demand_Analysis_Report.docx
```

- `app.py` — Streamlit dashboard, data analysis, and prediction functionality.
- `requirements.txt` — Python packages required to run the application.
- `README.md` — Project documentation.
- `.gitignore` — Specifies files and folders Git should ignore.
- `Bike_Sharing_Demand_Analysis_Colab (1).ipynb` — Notebook containing data analysis, visualizations, model training, and evaluation.
- `Bike_Sharing_Demand_Analysis_Report.docx` — Written project report.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git.
- Visual Studio Code or another Python development environment.
- Internet access to download the dataset if it is not already available locally.

### Step 1: Clone the repository

```bash
git clone https://github.com/Varuna-Jakkana/datascience1.git
cd datascience1
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

On Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

On Windows using PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

http://localhost:8501

The application may require internet access to download the UCI dataset on its first run. If the download fails, check the application's local dataset fallback instructions.

## Deployment

The Streamlit dashboard can be deployed using Streamlit Community Cloud.

1. Push the project files to GitHub.
2. Visit https://share.streamlit.io/.
3. Sign in with your GitHub account.
4. Create a new app.
5. Select the `datascience1` repository.
6. Select the `main` branch.
7. Set `app.py` as the main file.
8. Click **Deploy**.

After successful deployment, Streamlit provides a public URL for accessing the dashboard.

## Limitations

- Predictions are based on historical data and may not represent future demand under every condition.
- Unexpected events and external factors can affect actual bike rental demand.
- The dataset may not represent every city or bike-sharing service.
- Model performance may vary with different datasets, time periods, or model configurations.
- Predictions are estimates and should not be treated as guaranteed future rental counts.

## Future Enhancements

- Experiment with additional machine learning algorithms.
- Perform hyperparameter tuning and cross-validation.
- Add prediction uncertainty estimates.
- Improve dashboard accessibility and mobile responsiveness.
- Explore time-series forecasting for future demand planning.
- Evaluate the model on newer data and different locations.

## Conclusion

This project demonstrates an end-to-end data science workflow for analyzing and predicting hourly bike-sharing demand. It combines data preprocessing, exploratory analysis, visualization, machine learning, and interactive dashboard development.

The analysis identifies relationships between rental demand, time, season, and weather conditions. The recorded model comparison indicates that Random Forest performed better than Linear Regression according to the selected evaluation metrics.

The Streamlit dashboard makes the analysis interactive and allows users to explore rental patterns and generate demand estimates from selected conditions.

## References

1. UCI Machine Learning Repository — Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
2. Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence.

## Author

**Name:** J. Varuna  
**Course:** B.Tech — Computer Science and Engineering (AI & ML)

**Project:** Bike-Sharing Demand Analysis & Prediction
