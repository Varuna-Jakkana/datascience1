# Bike-Sharing Demand Analysis and Prediction

## Project Overview
This project analyzes bike-sharing demand using data science and machine learning techniques. It explores how demand changes according to time, season, working days, and weather conditions. Linear Regression and Random Forest models are developed to predict hourly bike rentals.

## Objectives
- Analyze historical bike-sharing demand.
- Clean and inspect the dataset.
- Perform Exploratory Data Analysis (EDA) using visualizations.
- Identify the factors associated with bike rental demand.
- Train and compare machine learning models.
- Evaluate model performance using MAE, RMSE, and R².

## Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset:** Bike Sharing Dataset
- **File used:** `hour.csv`
- **Original dataset size:** 17,379 rows and 17 columns.

Dataset link: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Technologies Used
- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Machine Learning

## Project Workflow
1. Define the problem.
2. Collect the dataset.
3. Inspect data quality and prepare the data.
4. Perform exploratory data analysis.
5. Select features and preprocess the data.
6. Split the data chronologically into training and testing sets.
7. Train Linear Regression and Random Forest models.
8. Evaluate and compare model performance.
9. Interpret the results and identify limitations.

## Key Findings
- Peak average demand occurs at **5:00 PM (17:00)**.
- **Fall** has the highest average seasonal demand.
- Working days have an average demand of **193.21 rentals per hour**, compared with **181.41** on non-working days.
- Clear or mostly clear weather has the highest average demand, at **204.87 rentals per hour**.
- Heavy rain or snow has the lowest average demand, at **74.33 rentals per hour**.

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 98.691 | 133.819 | 0.632 |
| Random Forest | 60.449 | 89.114 | 0.837 |

**Best-performing model:** Random Forest.

Random Forest achieved lower prediction errors and a higher R² score than Linear Regression on the test dataset.

## Repository Contents
- `Bike_Sharing_Demand_Analysis_Colab.ipynb` — Notebook containing the data science workflow, visualizations, model training, and evaluation.
- `Bike_Sharing_Demand_Analysis_Report.docx` — Project report.

## How to Run
1. Download the notebook from this repository.
2. Open https://colab.research.google.com/.
3. Upload the `.ipynb` notebook.
4. Run the cells sequentially.
5. The notebook downloads the dataset and performs the analysis and model evaluation.

## Conclusion
This project demonstrates an end-to-end data science workflow for analyzing and predicting hourly bike-sharing demand. Among the two models tested, Random Forest performed better based on MAE, RMSE, and R².

## Limitations
- Historical patterns may not represent future demand in every location.
- Unexpected events and other external factors may influence rentals.
- Model performance may change on different datasets or time periods.

## Author
**Name:** J.Varuna 
**Course:** B.Tech — CSE (AI & ML)
