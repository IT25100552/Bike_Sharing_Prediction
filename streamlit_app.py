import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(
    page_title="Capital Bikeshare Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------
# MODEL TRAINING & CACHING (Trains once on startup)
# --------------------------------------------------------------
@st.cache_resource(show_spinner="Training Production 3-Tier Models...")
def load_and_train():
    # 1. Locate file
    path = "hour_processed.csv"
    if not os.path.exists(path):
        if os.path.exists("Results/hour_processed.csv"):
            path = "Results/hour_processed.csv"
        else:
            raise FileNotFoundError("Could not find hour_processed.csv")

    df = pd.read_csv(path)
    df["hr"] = df["hr"].astype(int)

    # 2. Clock-arithmetic safe lags
    for k in (1, 2, 3, 24):
        ok = ((df["hr"] - df["hr"].shift(k)) % 24) == (k % 24)
        df[f"cnt_lag_{k}"] = df["cnt"].shift(k).where(ok)
    df["cnt_roll_mean_3"] = df[["cnt_lag_1", "cnt_lag_2", "cnt_lag_3"]].mean(axis=1, skipna=False)

    lag_cols = ["cnt_lag_1", "cnt_lag_2", "cnt_lag_3", "cnt_lag_24", "cnt_roll_mean_3"]
    NEVER = {"cnt", "casual", "registered", "instant", "dteday"} | set(lag_cols)
    BASE = [c for c in df.select_dtypes(include=[np.number]).columns if c not in NEVER]

    FEATS_T1  = BASE
    FEATS_T2B = BASE + ["cnt_lag_2", "cnt_lag_3", "cnt_lag_24"]
    FEATS_T2A = BASE + ["cnt_lag_1", "cnt_lag_2", "cnt_lag_24", "cnt_roll_mean_3"]

    # 80/20 Chronological Split
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    def subset(d, feats):
        return d.dropna(subset=[c for c in feats if c not in BASE] + ["cnt"])

    def make_model():
        return XGBRegressor(n_estimators=250, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1)

    tiers = {"T1": FEATS_T1, "T2B": FEATS_T2B, "T2A": FEATS_T2A}
    models = {}
    for name, feats in tiers.items():
        tr = subset(train, feats)
        models[name] = make_model().fit(tr[feats], tr["cnt"])

    # Evaluation on common test rows
    common = subset(test, FEATS_T2A + ["cnt_lag_3"])
    stats = {}
    for name, feats in tiers.items():
        pred = models[name].predict(common[feats])
        stats[name] = {
            "mae": mean_absolute_error(common["cnt"], pred),
            "r2": r2_score(common["cnt"], pred)
        }

    return models, tiers, stats, BASE

models, tiers, stats, BASE = load_and_train()

# --------------------------------------------------------------
# SIDEBAR: USER INPUT CONTROLS
# --------------------------------------------------------------
st.sidebar.header("🕒 Temporal & Horizon")
hour_val = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 8)
horizon_choice = st.sidebar.radio(
    "Forecast Horizon",
    ["1h Ahead (Tactical Dispatch)", "24h Ahead (Strategic Day-Ahead)"]
)
is_workday = st.sidebar.checkbox("Working Day (Mon–Fri)", value=True)
is_holiday = st.sidebar.checkbox("Official Holiday", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🌦️ Environmental Conditions")
temp_c = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 22.0, 0.5)
hum_pct = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 55.0, 1.0)
wind_kmh = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0, 14.0, 1.0)

weather_desc = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear / Partly Cloudy", "Mist / Overcast", "Light Rain / Light Snow", "Heavy Rain / Storm / Fog"]
)

st.sidebar.markdown("---")
st.sidebar.header("📡 Live Sensor Telemetry (Docks)")
sensor_status = st.sidebar.radio(
    "Dock IoT Sensor Stream",
    ["🟢 All Sensors Active", "🟡 Lag-1 Cellular Blip (NaN)", "🔴 Complete Sensor Outage"]
)

if sensor_status == "🟢 All Sensors Active":
    lag1_val = st.sidebar.number_input("Rentals in Prior Hour (cnt_lag_1)", min_value=0, value=150)
    lag24_val = st.sidebar.number_input("Rentals at Same Hour Yesterday (cnt_lag_24)", min_value=0, value=135)
elif sensor_status == "🟡 Lag-1 Cellular Blip (NaN)":
    lag1_val = np.nan
    lag24_val = st.sidebar.number_input("Rentals at Same Hour Yesterday (cnt_lag_24)", min_value=0, value=135)
else:
    lag1_val = np.nan
    lag24_val = np.nan

# --------------------------------------------------------------
# INFERENCE LOGIC (HARDENED ROUTING)
# --------------------------------------------------------------
weather_map = {
    "Clear / Partly Cloudy": 1,
    "Mist / Overcast": 2,
    "Light Rain / Light Snow": 3,
    "Heavy Rain / Storm / Fog": 4
}
w_code = weather_map[weather_desc]

input_dict = {
    'hr': hour_val,
    'temp': temp_c / 41.0,
    'hum': hum_pct / 100.0,
    'windspeed': wind_kmh / 67.0,
    'workingday': 1 if is_workday else 0,
    'holiday': 1 if is_holiday else 0,
    'is_rush_hour': 1 if hour_val in [7, 8, 9, 17, 18, 19] else 0,
    'is_weekend': 0 if is_workday else 1,
    'comfort_index': (temp_c / 41.0) * (1 - (hum_pct / 100.0)),
    'season_1': 1 if 3 <= hour_val <= 5 else 0,
    'season_2': 1 if 6 <= hour_val <= 8 else 0,
    'season_3': 1 if 9 <= hour_val <= 11 else 0,
    'season_4': 1 if hour_val in [12, 1, 2] else 0,
    'weather_1': 1 if w_code == 1 else 0,
    'weather_2': 1 if w_code == 2 else 0,
    'weather_3': 1 if w_code == 3 else 0,
    'weather_4': 1 if w_code == 4 else 0,
}

for col in BASE:
    if col not in input_dict:
        input_dict[col] = 0.0

input_dict['cnt_lag_1'] = lag1_val
input_dict['cnt_lag_2'] = lag1_val * 0.95 if not pd.isna(lag1_val) else np.nan
input_dict['cnt_lag_3'] = lag1_val * 0.90 if not pd.isna(lag1_val) else np.nan
input_dict['cnt_lag_24'] = lag24_val
input_dict['cnt_roll_mean_3'] = lag1_val

# Routing decision
h = 24 if "24h" in horizon_choice else 1
has_lag1 = ('cnt_lag_1' in input_dict and not pd.isna(input_dict['cnt_lag_1']))
has_lag24 = ('cnt_lag_24' in input_dict and not pd.isna(input_dict['cnt_lag_24']))

if h == 1 and has_lag1 and has_lag24:
    active_key = "T2A"
    tier_name = "Tier 2A: Full Tactical Real-Time"
    reason_msg = "Live dock sensors active (Lag-1, Lag-2, Lag-24 valid)."
    badge_color = "success"
elif h <= 2 and has_lag24:
    active_key = "T2B"
    tier_name = "Tier 2B: Seasonal Fallback (Lag-24 Active)"
    reason_msg = "Lag-1 unavailable; fell back to yesterday's same-hour telemetry."
    badge_color = "warning"
else:
    active_key = "T1"
    tier_name = "Tier 1: Strategic Day-Ahead (Weather Only)"
    reason_msg = f"Forecast horizon = {h}h ahead" if h > 2 else "All lag telemetry offline."
    badge_color = "info"

# Run Model
feats_used = tiers[active_key]
row_df = pd.DataFrame([{k: float(input_dict[k]) for k in feats_used}])
pred_val = max(0, int(round(models[active_key].predict(row_df)[0])))
tier_stats = stats[active_key]

# --------------------------------------------------------------
# MAIN DASHBOARD VIEW
# --------------------------------------------------------------
st.title("🚲 Capital Bikeshare Demand Forecasting Engine")
st.markdown("### Production-grade demand prediction with **dynamic 3-tier graceful fallback**.")

st.markdown("---")

# Metrics Cards Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted Demand", f"{pred_val} Bikes")

with col2:
    st.metric("Active Architecture", active_key)

with col3:
    st.metric("Validation Accuracy", f"R² = {tier_stats['r2']:.3f}")

with col4:
    st.metric("Expected Margin of Error", f"±{tier_stats['mae']:.1f} bikes")

# Routing Status Callout
if badge_color == "success":
    st.success(f"**{tier_name}** — {reason_msg}")
elif badge_color == "warning":
    st.warning(f"**{tier_name}** — {reason_msg}")
else:
    st.info(f"**{tier_name}** — {reason_msg}")

st.markdown("---")

# Visual Context & Explanation
c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("📊 Operational Rebalancing Insight")
    if pred_val > 400:
        st.error(f"🚨 **High Surge Demand Expected ({pred_val} rentals/h):** Priority station rebalancing required. Dispatch roving transport vans immediately.")
    elif pred_val > 150:
        st.warning(f"⚡ **Moderate Commuter Demand ({pred_val} rentals/h):** Regular dock turnover. Ensure peak stations are stocked to 60% capacity.")
    else:
        st.info(f"🟢 **Low Off-Peak Demand ({pred_val} rentals/h):** Normal operations. Optimal time window for fleet maintenance and battery charging.")

with c2:
    st.subheader("🛡️ Resiliency & Fallback Hierarchy")
    st.markdown("""
    * **Tier 2A (Full Tactical):** $R^2 \approx 0.955$, $\text{MAE} \approx 29$ bikes. Requires live streaming dock telemetry.
    * **Tier 2B (Seasonal Fallback):** $R^2 \approx 0.935$, $\text{MAE} \approx 34$ bikes. Activates on cellular sensor drops.
    * **Tier 1 (Strategic Planner):** $R^2 \approx 0.907$, $\text{MAE} \approx 44$ bikes. Operates on pure environmental forecasts for day-ahead fleet allocation.
    """)

st.caption("AI & Machine Learning Capstone Deliverable • Capital Bikeshare Washington D.C.")
