import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="BikeShare Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(100, 120, 150, 0.10);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(120, 140, 160, 0.20);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SEASON_NAMES = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter",
}

WEATHER_NAMES = {
    1: "Clear / mostly clear",
    2: "Mist / cloudy",
    3: "Light rain or snow",
    4: "Heavy rain or snow",
}

WEEKDAY_NAMES = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

SEASON_MONTHS = {
    "Spring": (1, 2, 3),
    "Summer": (4, 5, 6),
    "Fall": (7, 8, 9),
    "Winter": (10, 11, 12),
}

CATEGORY_FEATURES = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed"]
FEATURES = CATEGORY_FEATURES + NUMERIC_FEATURES
TARGET = "cnt"


# ---------------------------------------------------------
# DATA COLLECTION
# ---------------------------------------------------------
@st.cache_data(show_spinner="Downloading the official UCI dataset...")
def load_data():
    url = "https://archive.ics.uci.edu/static/public/" "275/bike+sharing+dataset.zip"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as error:
        local_hourly_file = Path(__file__).with_name("hour.csv")
        if not local_hourly_file.exists():
            raise RuntimeError(
                "UCI is unavailable and no local hour.csv fallback was found."
            ) from error
        data = pd.read_csv(local_hourly_file)
    else:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            hourly_file = next(
                name for name in archive.namelist() if name.endswith("hour.csv")
            )
            with archive.open(hourly_file) as file:
                data = pd.read_csv(file)

    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
    data = data.dropna(subset=["dteday", TARGET])
    data = data.drop_duplicates()
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    data["season_name"] = data["season"].map(SEASON_NAMES)
    data["weather_name"] = data["weathersit"].map(WEATHER_NAMES)
    data["day_type"] = data["workingday"].map(
        {
            0: "Non-working day",
            1: "Working day",
        }
    )
    data["weekday_name"] = data["weekday"].map(WEEKDAY_NAMES)
    data["hour_label"] = data["hr"].map(lambda hour: f"{hour:02d}:00")
    data = data.drop(columns=["yr"], errors="ignore")

    return data


# ---------------------------------------------------------
# MODEL TRAINING AND EVALUATION
# ---------------------------------------------------------
def make_model(model_type):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORY_FEATURES),
        ]
    )

    if model_type == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        estimator = LinearRegression()

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(data):
    # Do not use casual or registered: their sum equals cnt.
    # Exclude instant because it is only a record identifier.
    X = data[FEATURES].copy()
    y = data[TARGET].astype(float).copy()

    # Keep validation separate from the final chronological test set.
    train_end = int(len(data) * 0.70)
    test_start = int(len(data) * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_validation = X.iloc[train_end:test_start]
    y_validation = y.iloc[train_end:test_start]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    fitted_models = {}
    results = []

    for model_name in ["Linear Regression", "Random Forest"]:
        model = make_model(model_name)
        model.fit(X_train, y_train)
        validation_predictions = np.maximum(model.predict(X_validation), 0)
        validation_rmse = np.sqrt(
            mean_squared_error(y_validation, validation_predictions)
        )

        # Refit on all pre-test data after selecting the model family.
        model.fit(X.iloc[:test_start], y.iloc[:test_start])
        fitted_models[model_name] = model
        test_predictions = np.maximum(model.predict(X_test), 0)

        results.append(
            {
                "Model": model_name,
                "Validation RMSE": validation_rmse,
                "Test MAE": mean_absolute_error(y_test, test_predictions),
                "Test RMSE": np.sqrt(
                    mean_squared_error(y_test, test_predictions)
                ),
                "Test R²": r2_score(y_test, test_predictions),
            }
        )

    metrics = (
        pd.DataFrame(results)
        .sort_values("Validation RMSE")
        .reset_index(drop=True)
    )

    return fitted_models, metrics, test_start


# ---------------------------------------------------------
# LOAD DATA AND HANDLE ERRORS
# ---------------------------------------------------------
st.title("🚲 BikeShare Analytics")
st.markdown("### Bike-Sharing Demand Analysis & Prediction")
st.caption(
    "An interactive Data Science dashboard powered by the "
    "UCI Bike Sharing Dataset, Python and machine learning."
)

try:
    df = load_data()
except Exception as error:
    st.error(
        "The dataset could not be loaded. Check your internet connection "
        "or place a valid hour.csv beside app.py, then refresh."
    )
    st.caption(f"Technical details: {error}")
    st.stop()

with st.spinner("Preparing the analysis..."):
    models, metrics, test_start = train_models(df)

best_model_name = metrics.loc[0, "Model"]

st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Demand Analysis",
        "Weather Analysis",
        "Data Explorer",
        "Demand Prediction",
        "Model Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset: UCI Bike Sharing")


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.header("Project Overview")
    st.write(
        "Explore historical bike rental patterns, investigate the "
        "influence of weather and time, and compare machine learning "
        "models that estimate hourly bike demand."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Hourly records", f"{len(df):,}")
    col2.metric("Average rentals/hour", f"{df['cnt'].mean():.1f}")
    col3.metric("Peak observed rentals", f"{int(df['cnt'].max()):,}")
    col4.metric("Selected model test R²", f"{metrics.loc[0, 'Test R²']:.3f}")

    st.subheader("Hourly Demand Pattern")

    hourly = (
        df.groupby("hr", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    peak_row = hourly.loc[hourly["Average rentals"].idxmax()]
    st.info(
        f"Highest average demand occurs at "
        f"{int(peak_row['hr']):02d}:00, with "
        f"{peak_row['Average rentals']:.1f} rentals per hour."
    )

    fig = px.line(
        hourly,
        x="hr",
        y="Average rentals",
        markers=True,
        title="Average Bike Rentals by Hour",
        labels={"hr": "Hour of day", "Average rentals": "Average rentals"},
    )
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    season_summary = (
        df.groupby("season_name", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )
    season_summary["season_name"] = pd.Categorical(
        season_summary["season_name"],
        categories=["Spring", "Summer", "Fall", "Winter"],
        ordered=True,
    )
    season_summary = season_summary.sort_values("season_name")

    with left:
        fig = px.bar(
            season_summary,
            x="season_name",
            y="Average rentals",
            title="Average Demand by Season",
            labels={
                "season_name": "Season",
                "Average rentals": "Average rentals/hour",
            },
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        day_summary = (
            df.groupby("day_type", as_index=False)["cnt"]
            .mean()
            .rename(columns={"cnt": "Average rentals"})
        )
        fig = px.bar(
            day_summary,
            x="day_type",
            y="Average rentals",
            title="Working vs Non-working Days",
            labels={
                "day_type": "Day type",
                "Average rentals": "Average rentals/hour",
            },
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Model Performance")
    st.dataframe(
        metrics.style.format(
            {
                "Validation RMSE": "{:.3f}",
                "Test MAE": "{:.3f}",
                "Test RMSE": "{:.3f}",
                "Test R²": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------
# DEMAND ANALYSIS
# ---------------------------------------------------------
elif page == "Demand Analysis":
    st.header("Exploratory Demand Analysis")

    selected_seasons = st.multiselect(
        "Select seasons",
        ["Spring", "Summer", "Fall", "Winter"],
        default=["Spring", "Summer", "Fall", "Winter"],
    )

    selected_day_types = st.multiselect(
        "Select day types",
        ["Working day", "Non-working day"],
        default=["Working day", "Non-working day"],
    )

    filtered = df[
        df["season_name"].isin(selected_seasons)
        & df["day_type"].isin(selected_day_types)
    ]

    if filtered.empty:
        st.warning("Choose at least one season and one day type.")
        st.stop()

    hourly = (
        filtered.groupby("hr", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    fig = px.line(
        hourly,
        x="hr",
        y="Average rentals",
        markers=True,
        title="Average Demand by Hour",
        labels={"hr": "Hour of day"},
    )
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

    season_summary = (
        filtered.groupby("season_name", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    fig = px.bar(
        season_summary,
        x="season_name",
        y="Average rentals",
        color="season_name",
        title="Seasonal Demand",
        labels={
            "season_name": "Season",
            "Average rentals": "Average rentals/hour",
        },
    )
    st.plotly_chart(fig, width="stretch")

    day_hour = (
        filtered.groupby(["hr", "day_type"], as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    fig = px.line(
        day_hour,
        x="hr",
        y="Average rentals",
        color="day_type",
        markers=True,
        title="Hourly Demand by Day Type",
        labels={"hr": "Hour of day"},
    )
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Filtered data summary")
    st.write(f"Records included: **{len(filtered):,}**")
    st.dataframe(
        filtered[["dteday", "hr", "season_name", "day_type", "cnt"]].head(20),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------
# WEATHER ANALYSIS
# ---------------------------------------------------------
elif page == "Weather Analysis":
    st.header("Weather and Demand Analysis")

    weather_summary = (
        df.groupby("weather_name", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    fig = px.bar(
        weather_summary,
        x="weather_name",
        y="Average rentals",
        color="weather_name",
        title="Average Rentals by Weather Situation",
        labels={
            "weather_name": "Weather",
            "Average rentals": "Average rentals/hour",
        },
    )
    st.plotly_chart(fig, width="stretch")

    # Correlation heatmap uses numeric features only.
    corr_columns = NUMERIC_FEATURES + [TARGET]
    corr = df[corr_columns].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric Feature Correlation Heatmap",
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "Correlation describes association, not causation. Casual and "
        "registered rentals are excluded because their sum equals total "
        "rentals, which would create target leakage."
    )

    st.subheader("Numeric feature correlations with total demand")
    weather_corr = (
        corr["cnt"]
        .drop("cnt")
        .sort_values()
        .rename("Correlation with total rentals")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    st.dataframe(weather_corr, width="stretch", hide_index=True)


# ---------------------------------------------------------
# DATA EXPLORER
# ---------------------------------------------------------
elif page == "Data Explorer":
    st.header("Explore the Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Filter records")

    season_filter = st.multiselect(
        "Season",
        ["Spring", "Summer", "Fall", "Winter"],
        default=["Spring", "Summer", "Fall", "Winter"],
    )

    weather_filter = st.multiselect(
        "Weather",
        list(WEATHER_NAMES.values()),
        default=list(WEATHER_NAMES.values()),
    )

    day_filter = st.multiselect(
        "Day type",
        ["Working day", "Non-working day"],
        default=["Working day", "Non-working day"],
    )

    filtered = df[
        df["season_name"].isin(season_filter)
        & df["weather_name"].isin(weather_filter)
        & df["day_type"].isin(day_filter)
    ]

    st.write(f"Showing **{len(filtered):,}** matching records.")

    visible_columns = [
        "dteday",
        "hr",
        "season_name",
        "weather_name",
        "day_type",
        "temp",
        "hum",
        "windspeed",
        "cnt",
    ]

    st.dataframe(
        filtered[visible_columns],
        width="stretch",
        hide_index=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="filtered_bike_sharing_data.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# DEMAND PREDICTION
# ---------------------------------------------------------
elif page == "Demand Prediction":
    st.header("Predict Hourly Bike Demand")
    st.write(
        "Enter a new set of conditions and click **Predict Bike Demand**. "
        "The trained machine-learning models will generate an estimate. "
        "Temperature, feeling temperature, humidity and wind speed use the "
        "normalized scale from the original UCI dataset."
    )

    with st.form("prediction_form"):
        st.subheader("1. Time and calendar conditions")
        col1, col2, col3 = st.columns(3)

        with col1:
            season_name = st.selectbox(
                "Season", ["Spring", "Summer", "Fall", "Winter"], index=2
            )
            month = st.selectbox("Month", list(range(1, 13)), index=6)
            hour = st.slider("Hour of day", 0, 23, 17)

        with col2:
            weekday = st.selectbox(
                "Day of week", list(WEEKDAY_NAMES.values()), index=2
            )
            working_day_label = st.selectbox(
                "Day type", ["Working day", "Non-working day"]
            )
            holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])

        with col3:
            weather_name = st.selectbox(
                "Weather", list(WEATHER_NAMES.values()), index=0
            )
            temperature = st.slider(
                "Normalized temperature", 0.02, 1.0, 0.50, 0.01,
                help="The original dataset uses a normalized temperature value."
            )
            feels_like = st.slider(
                "Normalized feeling temperature", 0.0, 1.0, 0.48, 0.01
            )
            humidity = st.slider(
                "Normalized humidity", 0.0, 1.0, 0.63, 0.01
            )
            windspeed = st.slider(
                "Normalized wind speed", 0.0, 0.85, 0.19, 0.01
            )

        selected_model = st.selectbox(
            "Prediction model",
            ["Best-performing model", "Random Forest", "Linear Regression"],
            help="Best-performing model is selected by validation RMSE."
        )

        submitted = st.form_submit_button(
            "Predict Bike Demand", type="primary", width="stretch"
        )

    if submitted:
        season_code = {
            name: code for code, name in SEASON_NAMES.items()
        }[season_name]
        weather_code = {
            name: code for code, name in WEATHER_NAMES.items()
        }[weather_name]
        weekday_code = {
            name: code for code, name in WEEKDAY_NAMES.items()
        }[weekday]

        if month not in SEASON_MONTHS[season_name]:
            st.warning("Choose a month that matches the selected season.")
            st.stop()

        is_weekday = weekday_code not in (0, 6)
        expected_working_day = is_weekday and holiday == "No"
        if (working_day_label == "Working day") != expected_working_day:
            st.warning(
                "Day type must match the weekday and holiday selections."
            )
            st.stop()

        input_row = pd.DataFrame([{
            "season": season_code,
            "mnth": month,
            "hr": hour,
            "holiday": 1 if holiday == "Yes" else 0,
            "weekday": weekday_code,
            "workingday": 1 if working_day_label == "Working day" else 0,
            "weathersit": weather_code,
            "temp": temperature,
            "atemp": feels_like,
            "hum": humidity,
            "windspeed": windspeed,
        }], columns=FEATURES)

        # A new estimate is generated from the current form values each time
        # the user presses the prediction button.
        chosen_model_name = (
            best_model_name
            if selected_model == "Best-performing model"
            else selected_model
        )
        chosen_model = models[chosen_model_name]
        prediction = max(float(chosen_model.predict(input_row)[0]), 0.0)

        st.divider()
        st.subheader("2. Prediction result")
        result_col, model_col = st.columns(2)
        result_col.metric("Estimated rentals for this hour", f"{prediction:,.0f} bikes")
        model_col.metric("Model used", chosen_model_name)

        st.success(
            f"Estimated demand is **{prediction:,.0f} bike rentals** for "
            f"{hour:02d}:00, based on the conditions you entered."
        )

        # Show both model estimates so the user can compare the results.
        comparison_predictions = []
        for name, model in models.items():
            value = max(float(model.predict(input_row)[0]), 0.0)
            comparison_predictions.append({
                "Model": name,
                "Estimated rentals": round(value),
            })
        st.subheader("Compare predictions from both models")
        st.dataframe(
            pd.DataFrame(comparison_predictions),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Conditions used")
        summary = pd.DataFrame([{
            "Month": month,
            "Season": season_name,
            "Hour": f"{hour:02d}:00",
            "Weekday": weekday,
            "Day type": working_day_label,
            "Holiday": holiday,
            "Weather": weather_name,
            "Temperature (normalized)": temperature,
            "Feeling temperature (normalized)": feels_like,
            "Humidity (normalized)": humidity,
            "Wind speed (normalized)": windspeed,
        }])
        st.dataframe(summary, width="stretch", hide_index=True)

        st.caption(
            "This prediction is generated by a model trained on the UCI "
            "bike-sharing dataset. It is an estimate, not a "
            "guarantee of actual future demand. The selected month, season, "
            "weekday and day type should describe a consistent scenario."
        )


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.header("Machine Learning Model Evaluation")

    st.write(
        "Both models use chronological train, validation, and test periods. "
        "Validation RMSE selects the preferred model; test metrics report "
        "performance on the final unseen period."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Training + validation records", f"{test_start:,}")
    col2.metric("Test records", f"{len(df) - test_start:,}")
    col3.metric("Best model", best_model_name)

    st.subheader("Model comparison")
    st.dataframe(
        metrics.style.format(
            {
                "Validation RMSE": "{:.3f}",
                "Test MAE": "{:.3f}",
                "Test RMSE": "{:.3f}",
                "Test R²": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    fig = px.bar(
        metrics,
        x="Model",
        y="Test RMSE",
        color="Model",
        title="Model Comparison by RMSE (Lower is Better)",
        labels={"Test RMSE": "Test RMSE"},
        text_auto=".2f",
    )
    st.plotly_chart(fig, width="stretch")

    selected_model = st.selectbox(
        "Choose a model for the actual-vs-predicted chart",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    y_actual = df[TARGET].iloc[test_start:].to_numpy()
    X_test = df[FEATURES].iloc[test_start:]
    y_predicted = np.maximum(models[selected_model].predict(X_test), 0)

    comparison = pd.DataFrame(
        {
            "Actual rentals": y_actual,
            "Predicted rentals": y_predicted,
        }
    )

    fig = px.scatter(
        comparison,
        x="Actual rentals",
        y="Predicted rentals",
        title=f"Actual vs Predicted — {selected_model}",
        opacity=0.5,
    )

    axis_min = float(
        min(
            comparison["Actual rentals"].min(),
            comparison["Predicted rentals"].min(),
        )
    )
    axis_max = float(
        max(
            comparison["Actual rentals"].max(),
            comparison["Predicted rentals"].max(),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="Ideal prediction",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(
        xaxis_title="Actual rentals",
        yaxis_title="Predicted rentals",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Sample predictions")
    st.dataframe(
        comparison.head(20).round(2),
        width="stretch",
        hide_index=True,
    )

    st.markdown("""
        **Metric interpretation**
        - **MAE:** Average absolute prediction error; lower is better.
        - **RMSE:** Penalizes larger errors more strongly; lower is better.
        - **R²:** Measures explained variation compared with a mean baseline;
          higher is generally better.
        """)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "Dataset: Fanaee-T (2013), Bike Sharing Dataset, "
    "UCI Machine Learning Repository. "
    "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
)
