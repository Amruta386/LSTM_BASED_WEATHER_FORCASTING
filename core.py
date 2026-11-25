"""Shared logic for the live weather website: APIs, training, prediction."""
import json
import os
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from weather_lstm import FEATURES, WINDOW, build_model, download_open_meteo, make_windows

MODELS_DIR, DATA_DIR = "models", "data"
HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation"
CURRENT = ("temperature_2m,apparent_temperature,relative_humidity_2m,surface_pressure,"
           "wind_speed_10m,precipitation,weather_code")
COLS = {"temperature_2m": "temp", "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure", "wind_speed_10m": "wind", "precipitation": "rain"}


def city_key(lat, lon):
    return f"{lat:.2f}_{lon:.2f}"


# ------------------------------------------------------------------ APIs
def geocode(name, count=5):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": count, "language": "en", "format": "json"},
                     timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_live(lat, lon):
    """Current conditions + recent/forecast days aggregated to daily rows."""
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": lat, "longitude": lon, "current": CURRENT,
                             "hourly": HOURLY, "past_days": WINDOW + 1, "forecast_days": 3,
                             "timezone": "auto"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    h = js["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]),
                       **{v: h[k] for k, v in COLS.items()}}).set_index("time")
    daily = df.resample("D").agg({"temp": "mean", "humidity": "mean", "pressure": "mean",
                                  "wind": "mean", "rain": "sum"}).interpolate().dropna()
    return js["current"], daily[FEATURES]


# -------------------------------------------------------------- Training
def fit_scalers(data):
    train = data[: int(len(data) * 0.70)]
    return MinMaxScaler().fit(train.values), MinMaxScaler().fit(train[["temp"]].values)


def train_city(lat, lon, epochs=30, progress=None):
    """Download ~10 years of history for a city, train an LSTM, save everything."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    key = city_key(lat, lon)
    end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    data = download_open_meteo(lat, lon, "2015-01-01", end)
    data.to_csv(f"{DATA_DIR}/{key}.csv")

    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    scaler, tscaler = fit_scalers(data)
    prep = lambda df: make_windows(scaler.transform(df.values), WINDOW)
    tr, va, te = data[:train_end], data[train_end:val_end], data[val_end:]
    X_tr, y_tr = prep(tr)
    X_va, y_va = prep(pd.concat([tr.tail(WINDOW), va]))
    X_te, y_te = prep(pd.concat([va.tail(WINDOW), te]))

    cbs = [keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
    if progress:
        cbs.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: progress((e + 1) / epochs, e + 1, logs)))
    model = build_model("lstm", len(FEATURES))
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=32, callbacks=cbs, verbose=0)

    inv = lambda a: tscaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
    y_true, y_pred, y_base = inv(y_te), inv(model.predict(X_te, verbose=0)), inv(X_te[:, -1, 0])
    metrics = {"lstm_mae": float(mean_absolute_error(y_true, y_pred)),
               "lstm_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
               "lstm_r2": float(r2_score(y_true, y_pred)),
               "baseline_mae": float(mean_absolute_error(y_true, y_base)),
               "days": int(n), "test_days": int(len(y_true)),
               "period": f"{data.index[0].date()} to {data.index[-1].date()}"}
    model.save(f"{MODELS_DIR}/{key}.keras")
    json.dump(metrics, open(f"{MODELS_DIR}/{key}.json", "w"))
    return metrics


def load_city(lat, lon):
    key = city_key(lat, lon)
    paths = [f"{MODELS_DIR}/{key}.keras", f"{MODELS_DIR}/{key}.json", f"{DATA_DIR}/{key}.csv"]
    if not all(os.path.exists(p) for p in paths):
        return None
    data = pd.read_csv(paths[2], index_col=0, parse_dates=True)[FEATURES]
    scaler, tscaler = fit_scalers(data)
    return {"model": keras.models.load_model(paths[0]), "scaler": scaler,
            "tscaler": tscaler, "metrics": json.load(open(paths[1]))}


# ------------------------------------------------------------ Prediction
def predict_next(bundle, window_df):
    x = bundle["scaler"].transform(window_df.values)[np.newaxis, ...]
    y = bundle["model"].predict(x, verbose=0)
    return float(bundle["tscaler"].inverse_transform(y)[0, 0])
