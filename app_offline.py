"""
Live demo for the exhibition.

Run AFTER training (python weather_lstm.py):
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

WINDOW = 14
FEATURES = ["temp", "humidity", "pressure", "wind", "rain"]
LABELS = {"temp": "Temperature (°C)", "humidity": "Humidity (%)",
          "pressure": "Pressure (hPa)", "wind": "Wind (km/h)", "rain": "Rain (mm)"}

st.set_page_config(page_title="Weather Prediction using LSTM", page_icon="🌦️", layout="wide")


@st.cache_resource
def load_everything():
    if not (os.path.exists("weather_data.csv") and os.path.exists("weather_lstm.keras")):
        return None
    data = pd.read_csv("weather_data.csv", index_col=0, parse_dates=True)[FEATURES]
    model = keras.models.load_model("weather_lstm.keras")
    n = len(data)
    train = data[: int(n * 0.70)]
    scaler = MinMaxScaler().fit(train.values)          # same as training script
    target_scaler = MinMaxScaler().fit(train[["temp"]].values)
    return data, model, scaler, target_scaler, int(n * 0.85)


def predict(model, scaler, target_scaler, window_df):
    x = scaler.transform(window_df.values)[np.newaxis, ...]
    return float(target_scaler.inverse_transform(model.predict(x, verbose=0))[0, 0])


st.title("🌦️ Weather Prediction using LSTM")
st.caption("Deep learning project: a Long Short-Term Memory network that predicts "
           "tomorrow's temperature from the past 14 days of weather.")

loaded = load_everything()
if loaded is None:
    st.error("Model or data not found. Run `python weather_lstm.py` first, "
             "then start this app from the same folder.")
    st.stop()
data, model, scaler, target_scaler, test_start = loaded

tab_demo, tab_forecast, tab_how, tab_perf = st.tabs(
    ["🎯 Test the model", "🔮 Forecast tomorrow", "🧠 How it works", "📊 Performance"])

# ---------------------------------------------------------------- Test tab
with tab_demo:
    st.subheader("Pick a day the model has never seen")
    test_dates = data.index[test_start + WINDOW:]
    day = st.select_slider("Day to predict", options=list(test_dates),
                           value=test_dates[len(test_dates) // 2],
                           format_func=lambda d: d.strftime("%d %b %Y"))
    pos = data.index.get_loc(day)
    window_df = data.iloc[pos - WINDOW:pos]
    actual = float(data.iloc[pos]["temp"])
    pred = predict(model, scaler, target_scaler, window_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM prediction", f"{pred:.1f} °C")
    c2.metric("Actual temperature", f"{actual:.1f} °C")
    c3.metric("Error", f"{abs(pred - actual):.2f} °C")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(window_df.index, window_df["temp"], marker="o", label="Past 14 days (model input)")
    ax.scatter([day], [actual], color="green", s=90, label="Actual", zorder=3)
    ax.scatter([day], [pred], color="red", marker="x", s=110, label="LSTM prediction", zorder=3)
    ax.set_ylabel("°C")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    with st.expander("See the 14-day input the model was given"):
        st.dataframe(window_df.rename(columns=LABELS).round(2))

# ------------------------------------------------------------ Forecast tab
with tab_forecast:
    st.subheader("Live forecast from the most recent data")
    latest = data.iloc[-WINDOW:]
    tomorrow = predict(model, scaler, target_scaler, latest)
    st.metric(f"Predicted mean temperature for {(data.index[-1] + pd.Timedelta(days=1)).strftime('%d %b %Y')}",
              f"{tomorrow:.1f} °C",
              delta=f"{tomorrow - float(latest['temp'].iloc[-1]):+.1f} °C vs last day")

    st.markdown("**What-if:** change the last day's conditions and watch the forecast move.")
    a, b, c = st.columns(3)
    t_adj = a.slider("Temperature on last day (°C)", 0.0, 45.0, float(latest["temp"].iloc[-1]), 0.5)
    h_adj = b.slider("Humidity (%)", 0.0, 100.0, float(latest["humidity"].iloc[-1]), 1.0)
    p_adj = c.slider("Pressure (hPa)", float(data["pressure"].min()), float(data["pressure"].max()),
                     float(latest["pressure"].iloc[-1]), 0.5)
    what_if = latest.copy()
    what_if.iloc[-1, what_if.columns.get_loc("temp")] = t_adj
    what_if.iloc[-1, what_if.columns.get_loc("humidity")] = h_adj
    what_if.iloc[-1, what_if.columns.get_loc("pressure")] = p_adj
    st.metric("What-if forecast", f"{predict(model, scaler, target_scaler, what_if):.1f} °C")

# ---------------------------------------------------------------- How tab
with tab_how:
    st.subheader("How the system works")
    st.markdown("""
1. **Data**: daily temperature, humidity, pressure, wind and rainfall history.
2. **Windows**: every training example is 14 consecutive days (input) and the following day's temperature (answer).
3. **Scaling**: values are squeezed to 0–1 so the network learns stably. The scaler only sees training data.
4. **LSTM layers**: an LSTM keeps a memory of the sequence, so it learns patterns like seasons and pressure trends.
5. **Prediction**: a final dense layer outputs tomorrow's temperature.
6. **Testing**: the last 15% of the timeline was never seen during training. That is what you tested in the first tab.
""")
    st.info("Why LSTM instead of a plain neural network? Weather is a time series, and "
            "what happened over the last days matters. LSTM cells decide what to remember and what to forget.")

# ---------------------------------------------------------------- Perf tab
with tab_perf:
    st.subheader("Model performance on unseen test data")
    if os.path.exists("results.png"):
        st.image("results.png", use_container_width=True)
    else:
        st.write("Run `python weather_lstm.py` to generate `results.png`.")
    st.caption("Compared against a baseline that simply predicts tomorrow = today, "
               "and against a GRU model. See the console output of the training script for the metrics table.")
