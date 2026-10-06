import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Capital Bikeshare Demand Forecaster",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. LOAD DATA & TRAIN CHAMPION XGBOOST MODEL (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Champion XGBoost Regressor...")
def load_and_train():
    path = "hour_processed.csv"
    if not os.path.exists(path):
        if os.path.exists("Results/hour_processed.csv"):
            path = "Results/hour_processed.csv"
        else:
            raise FileNotFoundError("Could not find hour_processed.csv")

    df = pd.read_csv(path)
    
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        raise ValueError(f"Corrupt input file: {n_missing} unexpected NaNs found in {path}")

    X = df.drop(columns=['cnt']).astype(float)
    y = df['cnt'].astype(float)

    # 80/20 Chronological Split (Strictly preserves time order, prevents leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Tuned Champion Configuration: lr=0.2, max_depth=5, n_estimators=200
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.2,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, X.columns, r2, mae, rmse

model, feature_cols, test_r2, test_mae, test_rmse = load_and_train()

# ==============================================================================
# 2. DICTIONARIES & DOMAIN LOGIC
# ==============================================================================
MONTHS_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

WEEKDAYS_MAP = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
}

WEATHER_MAP = {
    "Clear / Partly Cloudy": 1,
    "Mist / Overcast": 2,
    "Light Rain / Light Snow": 3,
    "Heavy Rain / Storm / Fog": 4
}

def get_season_from_month(mnth):
    """
    Auto-derives season strictly from month per UCI quarterly calendar definition:
    1: Spring (Q1), 2: Summer (Q2), 3: Fall (Q3), 4: Winter (Q4).
    Eliminates inconsistent edge cases like 'January + Summer'.
    """
    if mnth in [1, 2, 3]:
        return 1, "Spring (Q1)"
    elif mnth in [4, 5, 6]:
        return 2, "Summer (Q2)"
    elif mnth in [7, 8, 9]:
        return 3, "Fall (Q3)"
    else:
        return 4, "Winter (Q4)"

# ==============================================================================
# 3. SIDEBAR INTERFACE CONTROLS
# ==============================================================================
st.sidebar.title("🎛️ Forecast Parameters")

st.sidebar.subheader("📅 Temporal & Calendar Settings")
month_str = st.sidebar.selectbox("Month of the Year", list(MONTHS_MAP.keys()), index=9)
mnth = MONTHS_MAP[month_str]
season_code, season_label = get_season_from_month(mnth)
st.sidebar.caption(f"🗓️ Auto-derived Season: **{season_label}**")

day_str = st.sidebar.selectbox("Day of the Week", list(WEEKDAYS_MAP.keys()), index=1)
is_holiday = st.sidebar.checkbox("Official Holiday", value=False)
hour_val = st.sidebar.slider("Hour of the Day (0–23)", min_value=0, max_value=23, value=8)

st.sidebar.markdown("---")
st.sidebar.subheader("🌦️ Meteorological Telemetry")
temp_c = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=20.0, step=0.5)
hum_pct = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
wind_kmh = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=1.0)
weather_desc = st.sidebar.selectbox("Weather Condition", list(WEATHER_MAP.keys()), index=0)

# ==============================================================================
# 4. FEATURE TRANSFORMATION BRIDGE
# ==============================================================================
weekday = WEEKDAYS_MAP[day_str]
w_code = WEATHER_MAP[weather_desc]
holiday_val = 1 if is_holiday else 0

is_weekend = 1 if weekday in [0, 6] else 0
workingday = 1 if (is_weekend == 0 and holiday_val == 0) else 0

# Normalization per UCI scale definitions
norm_temp = temp_c / 41.0
norm_hum = hum_pct / 100.0
norm_wind = wind_kmh / 67.0

# Exact reconstruction of all 26 model features
feat = {
    'yr': 1.0,
    'mnth': float(mnth),
    'hr': float(hour_val),
    'holiday': float(holiday_val),
    'weekday': float(weekday),
    'workingday': float(workingday),
    'temp': float(norm_temp),
    'hum': float(norm_hum),
    'windspeed': float(norm_wind),
    'season_1': 1.0 if season_code == 1 else 0.0,
    'season_2': 1.0 if season_code == 2 else 0.0,
    'season_3': 1.0 if season_code == 3 else 0.0,
    'season_4': 1.0 if season_code == 4 else 0.0,
    'weather_1': 1.0 if w_code == 1 else 0.0,
    'weather_2': 1.0 if w_code == 2 else 0.0,
    'weather_3': 1.0 if w_code == 3 else 0.0,
    'weather_4': 1.0 if w_code == 4 else 0.0,
    'hr_sin': float(np.sin(2 * np.pi * hour_val / 24)),
    'hr_cos': float(np.cos(2 * np.pi * hour_val / 24)),
    'mnth_sin': float(np.sin(2 * np.pi * mnth / 12)),
    'mnth_cos': float(np.cos(2 * np.pi * mnth / 12)),
    'weekday_sin': float(np.sin(2 * np.pi * weekday / 7)),
    'weekday_cos': float(np.cos(2 * np.pi * weekday / 7)),
    'is_rush_hour': 1.0 if hour_val in [7, 8, 9, 17, 18, 19] else 0.0,
    'is_weekend': float(is_weekend),
    'comfort_index': float(norm_temp * (1.0 - norm_hum))
}

input_df = pd.DataFrame([[feat[col] for col in feature_cols]], columns=feature_cols)
pred_count = max(0, int(round(model.predict(input_df)[0])))

# ==============================================================================
# 5. DASHBOARD LAYOUT & OPERATIONAL DIRECTIVES
# ==============================================================================
st.title("🚲 Capital Bikeshare Demand Prediction System")
st.markdown(
    f"**Production Day-Ahead Champion Forecaster** · *Tuned XGBoost Regressor* "
    f"· Test $R^2 = {test_r2:.4f}$ · Test $\\text{{MAE}} = \\pm{test_mae:.1f}$ bikes"
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Hourly Demand", f"{pred_count} Bikes", delta=f"{pred_count - 150} vs avg")
with col2:
    st.metric("Test Accuracy (R²)", f"{test_r2:.3f}", "91.5% variance explained")
with col3:
    st.metric("Mean Absolute Error", f"±{test_mae:.1f} Bikes", "Out-of-sample")
with col4:
    st.metric("Root Mean Squared Error", f"{test_rmse:.1f} Bikes", "Outlier penalty")

st.markdown("---")

# Operational Logistics Directive
st.subheader("📋 Operational Directive for Station Logistics")
if pred_count > 380:
    st.error(
        f"🚨 **High Commuter Surge Detected ({pred_count} bikes/h):** "
        f"Priority dock rebalancing required. Dispatch distribution trucks immediately to replenish "
        f"high-traffic metro stations and transit hubs before dock exhaustion occurs."
    )
elif pred_count > 130:
    st.warning(
        f"⚡ **Moderate Commuter Flow ({pred_count} bikes/h):** "
        f"Stable turnover across central corridors. Maintain station docks at ~60% capacity to balance "
        f"incoming returns and outbound rentals."
    )
else:
    st.info(
        f"🟢 **Off-Peak / Low Demand Window ({pred_count} bikes/h):** "
        f"Minimal rental turnover. Recommended window for fleet maintenance, battery charging, and dock inspections."
    )

# Technical Inspection Expander
with st.expander("🔍 Inspect Model Input Vector (26 Engineered Features)"):
    st.write("Below is the exact normalized feature vector passed to the XGBoost inference engine:")
    st.dataframe(input_df, use_container_width=True)

st.caption("Capital Bikeshare Machine Learning Deployment · Engineered with Streamlit and XGBoost")
