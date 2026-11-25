"""
Weather Prediction using LSTM
-----------------------------
Predicts tomorrow's mean temperature from the previous N days of weather.

Setup:   pip install tensorflow pandas numpy scikit-learn matplotlib requests
Run:     python weather_lstm.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------- Config ---------------------------------
LATITUDE, LONGITUDE = 12.97, 77.59      # change to your city
START_DATE, END_DATE = "2015-01-01", "2025-12-31"
WINDOW = 14                              # days of history used per prediction
EPOCHS, BATCH_SIZE = 60, 32
SEED = 42
FEATURES = ["temp", "humidity", "pressure", "wind", "rain"]  # temp = target

np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


# ----------------------------- 1. Data --------------------------------
CACHE_FILE = "weather_data.csv"


def download_open_meteo(lat=None, lon=None, start=None, end=None) -> pd.DataFrame:
    """Download hourly historical weather from Open-Meteo and resample to daily."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE if lat is None else lat,
        "longitude": LONGITUDE if lon is None else lon,
        "start_date": start or START_DATE,
        "end_date": end or END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,"
                  "wind_speed_10m,precipitation",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temp": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "pressure": h["surface_pressure"],
        "wind": h["wind_speed_10m"],
        "rain": h["precipitation"],
    }).set_index("time")
    daily = df.resample("D").agg(
        {"temp": "mean", "humidity": "mean", "pressure": "mean",
         "wind": "mean", "rain": "sum"}
    )
    return daily.interpolate().dropna()[FEATURES]


def make_synthetic(days: int = 3000) -> pd.DataFrame:
    """Fake seasonal weather. ONLY for testing that the code runs offline."""
    rng = np.random.default_rng(SEED)
    t = np.arange(days)
    season = np.sin(2 * np.pi * t / 365.25)
    temp = 25 + 6 * season + rng.normal(0, 1.5, days)
    humidity = 65 - 15 * season + rng.normal(0, 5, days)
    pressure = 1010 + 3 * season + rng.normal(0, 1.5, days)
    wind = 10 + 3 * np.cos(2 * np.pi * t / 365.25) + rng.normal(0, 2, days)
    rain = np.clip(rng.gamma(0.4, 5, days) * (1 - season) / 2, 0, None)
    idx = pd.date_range("2015-01-01", periods=days, freq="D")
    return pd.DataFrame({"temp": temp, "humidity": humidity, "pressure": pressure,
                         "wind": wind, "rain": rain}, index=idx)


def load_data(args) -> pd.DataFrame:
    """Priority: --synthetic > --csv file > cached download > fresh download."""
    if args.synthetic:
        print("WARNING: using SYNTHETIC data (testing only, not real weather).")
        return make_synthetic()
    if args.csv:
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        return df[FEATURES].interpolate().dropna()
    if os.path.exists(CACHE_FILE):
        print(f"Using cached data from {CACHE_FILE} (delete it to re-download)")
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    df = download_open_meteo()
    df.to_csv(CACHE_FILE)
    return df


# ----------------------------- 2. Windows -----------------------------
def make_windows(values: np.ndarray, window: int):
    """values: (days, features), column 0 is the target. Returns X, y."""
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window, 0])
    return np.array(X), np.array(y)


# ----------------------------- 3. Model -------------------------------
def build_model(cell: str, n_features: int) -> keras.Model:
    rnn = layers.LSTM if cell == "lstm" else layers.GRU
    model = keras.Sequential([
        layers.Input(shape=(WINDOW, n_features)),
        rnn(64, return_sequences=True),
        layers.Dropout(0.2),
        rnn(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return mae, rmse, r2


# ----------------------------- 4. Main --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Weather prediction with LSTM")
    ap.add_argument("--synthetic", action="store_true", help="offline test data")
    ap.add_argument("--csv", help="your own CSV: date index + temp,humidity,pressure,wind,rain")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    data = load_data(args)
    print(f"Loaded {len(data)} days: {data.index[0].date()} -> {data.index[-1].date()}")
    print(data.describe().round(2))

    # Chronological split (never shuffle time series)
    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df, val_df, test_df = data[:train_end], data[train_end:val_end], data[val_end:]

    # Fit scalers on TRAIN only to avoid data leakage
    scaler = MinMaxScaler().fit(train_df.values)
    target_scaler = MinMaxScaler().fit(train_df[["temp"]].values)

    def prep(df):
        # prepend the tail of the previous split so the first windows are complete
        return make_windows(scaler.transform(df.values), WINDOW)

    X_train, y_train = prep(train_df)
    X_val, y_val = prep(pd.concat([train_df.tail(WINDOW), val_df]))
    X_test, y_test = prep(pd.concat([val_df.tail(WINDOW), test_df]))
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    results, preds = {}, {}
    for cell in ("lstm", "gru"):
        print(f"\nTraining {cell.upper()} ...")
        model = build_model(cell, len(FEATURES))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=2,
        )
        p = target_scaler.inverse_transform(model.predict(X_test)).ravel()
        preds[cell] = p
        if cell == "lstm":
            lstm_history = history
            model.save("weather_lstm.keras")

    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Baseline: "tomorrow's temperature = today's temperature"
    last_day = target_scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)).ravel()

    print("\n=== Test results (deg C) ===")
    results["Persistence baseline"] = evaluate("Persistence baseline", y_true, last_day)
    results["LSTM"] = evaluate("LSTM", y_true, preds["lstm"])
    results["GRU"] = evaluate("GRU", y_true, preds["gru"])

    # ------------------------- Plots ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(lstm_history.history["loss"], label="train")
    ax[0].plot(lstm_history.history["val_loss"], label="validation")
    ax[0].set_title("LSTM training curve (MSE)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()

    show = 120
    ax[1].plot(y_true[:show], label="actual")
    ax[1].plot(preds["lstm"][:show], label="LSTM")
    ax[1].plot(preds["gru"][:show], label="GRU", alpha=0.7)
    ax[1].set_title(f"Next-day temperature (first {show} test days)")
    ax[1].set_ylabel("deg C")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved: results.png, weather_lstm.keras")

    # Predict tomorrow from the most recent window
    latest = scaler.transform(data.values[-WINDOW:])[np.newaxis, ...]
    model = keras.models.load_model("weather_lstm.keras")
    tomorrow = target_scaler.inverse_transform(model.predict(latest))[0, 0]
    print(f"Forecast for next day after {data.index[-1].date()}: {tomorrow:.1f} deg C")


if __name__ == "__main__":
    main()
