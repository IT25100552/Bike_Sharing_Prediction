# 🚲 Capital Bikeshare Demand Forecaster (Streamlit Deployment)

An interactive, production-grade web application predicting hourly bike-sharing demand in Washington D.C. Powered by an optimized **XGBoost Regressor** trained on the Capital Bikeshare dataset.

---

## 🌟 Model Highlights & Performance

* **Champion Model**: eXtreme Gradient Boosting (`XGBRegressor`)
* **Optimal Hyperparameters**: `n_estimators=200`, `learning_rate=0.20`, `max_depth=5`
* **Test Accuracy ($R^2$)**: **0.9153** (explains 91.5% of out-of-sample demand variance)
* **Mean Absolute Error (MAE)**: **±41.37 bikes**
* **Root Mean Squared Error (RMSE)**: **64.16 bikes**
* **Zero Input Inconsistencies**: Automatically synchronizes month and season to prevent impossible combinations (e.g. *January + Summer*).

---

## 📁 Repository Structure

```text
├── app.py                  # Main Streamlit web application & inference engine
├── hour_processed.csv      # Preprocessed feature-engineered dataset (2.8 MB)
├── requirements.txt        # Minimal production Python dependencies
├── .gitignore              # Ignores bytecode and cache files
└── README.md               # Project documentation & deployment guide
```

---

## 🚀 Quickstart: Run Locally

### 1. Clone or Download this repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Community Cloud (100% Free)

1. Push this folder to a new **Public** repository on [GitHub](https://github.com).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"** and fill in:
   * **Repository**: `<YOUR-USERNAME>/<YOUR-REPO-NAME>`
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. Click **"Deploy!"**
5. Your app will be live 24/7 with a public link (e.g., `https://your-app.streamlit.app`).

---

## 🧠 Feature Engineering Overview (26 Features)

* **Temporal Dynamics**: Year (`yr`), Month (`mnth`), Hour (`hr`), Weekday (`weekday`), Official Holiday (`holiday`), Working Day (`workingday`).
* **Trigonometric Cyclical Transformations**: Sine and Cosine encodings (`hr_sin`, `hr_cos`, `mnth_sin`, `mnth_cos`, `weekday_sin`, `weekday_cos`).
* **Environmental & Meteorological**: Normalized Temperature (`temp`), Humidity (`hum`), Wind Speed (`windspeed`), Weather Categorical Dummies (`weather_1` to `weather_4`).
* **Engineered Indicators**: Rush-Hour Flag (`is_rush_hour`), Weekend Flag (`is_weekend`), Thermal Comfort Index (`comfort_index`).
