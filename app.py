
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Cloud-Based Sales Forecasting System")
st.caption("Machine Learning | Data Science | Cloud Computing")

# ---------------- LOAD DATASET ----------------
DATA_PATH = Path(__file__).parent / "data" / "sales_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"])
    df = df[df["Sales"] >= 0]
    df = df.sort_values("Date")
    df = df.groupby("Date", as_index=False)["Sales"].sum()
    return df


if not DATA_PATH.exists():
    st.error("Dataset not found. Please create data/sales_data.csv.")
    st.stop()


df = load_data()

st.sidebar.header("Sales Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your sales CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = {"Date", "Sales"}
        if not required_columns.issubset(df.columns):
            st.error("CSV must contain Date and Sales columns.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Date", "Sales"])
        df = df[df["Sales"] >= 0]
        df = (
            df.groupby("Date", as_index=False)["Sales"]
            .sum()
            .sort_values("Date")
        )

        st.sidebar.success("Sales dataset uploaded!")

    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

if len(df) < 10:
    st.error("Please provide at least 10 valid daily sales records.")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=60,
    value=14
)

# ---------------- SUMMARY METRICS ----------------
total_sales = df["Sales"].sum()
average_sales = df["Sales"].mean()
latest_sales = df["Sales"].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Total Historical Sales", f"{total_sales:,.2f}")
col2.metric("Average Daily Sales", f"{average_sales:,.2f}")
col3.metric("Latest Recorded Sales", f"{latest_sales:,.2f}")

# ---------------- PREPARE FEATURES ----------------
df["DayNumber"] = (df["Date"] - df["Date"].min()).dt.days
df["DayOfWeek"] = df["Date"].dt.dayofweek

X = df[["DayNumber", "DayOfWeek"]]
y = df["Sales"]

# Use a chronological holdout for evaluation.
split_index = max(1, int(len(df) * 0.8))
if len(df) - split_index < 1:
    split_index = len(df) - 1

evaluation_model = LinearRegression()
evaluation_model.fit(X.iloc[:split_index], y.iloc[:split_index])

test_predictions = evaluation_model.predict(X.iloc[split_index:])
actual_values = y.iloc[split_index:]

mae = mean_absolute_error(actual_values, test_predictions)
rmse = np.sqrt(mean_squared_error(actual_values, test_predictions))

# Train the final model on all available historical data.
model = LinearRegression()
model.fit(X, y)

# ---------------- GENERATE FORECAST ----------------
last_date = df["Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)

future_df = pd.DataFrame({"Date": future_dates})
future_df["DayNumber"] = (
    future_df["Date"] - df["Date"].min()
).dt.days
future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek

future_X = future_df[["DayNumber", "DayOfWeek"]]
future_df["ForecastSales"] = np.maximum(
    model.predict(future_X), 0
)

# ---------------- SALES CHART ----------------
st.subheader("Historical Sales and Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Sales"],
    mode="lines",
    name="Historical Sales"
))

fig.add_trace(go.Scatter(
    x=future_df["Date"],
    y=future_df["ForecastSales"],
    mode="lines+markers",
    name="Forecast Sales",
    line=dict(dash="dash")
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified",
    legend_title="Series"
)

st.plotly_chart(fig, use_container_width=True)



# ---------------- MODEL EVALUATION ----------------
st.subheader("Model Evaluation")

# Baseline: predict the average sales from training data
baseline_predictions = np.full(
    len(actual_values),
    y.iloc[:split_index].mean()
)

baseline_mae = mean_absolute_error(
    actual_values, baseline_predictions
)
baseline_rmse = np.sqrt(
    mean_squared_error(actual_values, baseline_predictions)
)

metric1, metric2 = st.columns(2)
metric1.metric("Linear Regression MAE", f"{mae:,.2f}")
metric2.metric("Linear Regression RMSE", f"{rmse:,.2f}")

st.caption(
    "Both methods are evaluated on the same chronological test period."
)

# Comparison table
comparison = pd.DataFrame({
    "Model": ["Historical Average Baseline", "Linear Regression"],
    "MAE": [baseline_mae, mae],
    "RMSE": [baseline_rmse, rmse]
})

st.subheader("Forecasting Model Comparison")
st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)

if mae < baseline_mae:
    st.success(
        "Linear Regression has lower MAE than the baseline "
        "on this test dataset."
    )
elif mae > baseline_mae:
    st.info(
        "The baseline has lower MAE on this test dataset. "
        "The forecasting model may need improvement."
    )
else:
    st.info("Both methods have equal MAE on this test dataset.")

# Actual versus predicted sales
test_results = df.iloc[split_index:].copy()
test_results["PredictedSales"] = test_predictions

fig_eval = go.Figure()

fig_eval.add_trace(go.Scatter(
    x=test_results["Date"],
    y=test_results["Sales"],
    mode="lines+markers",
    name="Actual Sales"
))

fig_eval.add_trace(go.Scatter(
    x=test_results["Date"],
    y=test_results["PredictedSales"],
    mode="lines+markers",
    name="Linear Regression"
))

fig_eval.add_trace(go.Scatter(
    x=test_results["Date"],
    y=baseline_predictions,
    mode="lines",
    name="Historical Average Baseline"
))

fig_eval.update_layout(
    title="Actual vs Predicted Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified"
)

st.plotly_chart(fig_eval, use_container_width=True)


# ---------------- FORECAST TABLE ----------------
st.subheader("Forecast Results")
display_forecast = future_df[["Date", "ForecastSales"]].copy()
display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
display_forecast["ForecastSales"] = display_forecast["ForecastSales"].round(2)

st.dataframe(display_forecast, use_container_width=True, hide_index=True)

csv_data = display_forecast.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast as CSV",
    data=csv_data,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.info(
    "This project uses synthetic sample data. Replace it with genuine "
    "historical sales records for meaningful business forecasts."
)
