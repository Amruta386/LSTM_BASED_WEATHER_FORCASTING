"""
Real-time weather website with LSTM forecasting.

    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core import WINDOW, fetch_live, geocode, load_city, predict_next, train_city

st.set_page_config(page_title="Live Weather + LSTM", page_icon="🌦️", layout="wide")

CODES = {0: ("Clear sky", "☀️"), 1: ("Mostly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
         3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Fog", "🌫️"),
         51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
         61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
         71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"),
         80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"), 82: ("Violent showers", "⛈️"),
         95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm, hail", "⛈️"), 99: ("Severe thunderstorm", "⛈️")}


@st.cache_data(ttl=86400, show_spinner=False)
def cached_geocode(q):
    return geocode(q)


@st.cache_data(ttl=600, show_spinner="Fetching live weather...")
def cached_live(lat, lon):
    return fetch_live(lat, lon)


# ------------------------------------------------------------- Sidebar
st.sidebar.header("🔎 Choose a city")
query = st.sidebar.text_input("City name", "Delhi")
if st.sidebar.button("🔄 Refresh live data"):
    cached_live.clear()

try:
    hits = cached_geocode(query) if query.strip() else []
except Exception as e:
    st.error(f"Could not reach the geocoding service. Check your internet connection.\n\n{e}")
    st.stop()
if not hits:
    st.warning("No city found. Try another spelling.")
    st.stop()

label = lambda h: f"{h['name']}, {h.get('admin1', '')}, {h.get('country', '')}".replace(", ,", ",")
city = hits[st.sidebar.selectbox("Matches", range(len(hits)), format_func=lambda i: label(hits[i]))]
lat, lon = city["latitude"], city["longitude"]

# ---------------------------------------------------------- Live weather
try:
    current, daily = cached_live(lat, lon)
except Exception as e:
    st.error(f"Could not fetch live weather. Check your internet connection.\n\n{e}")
    st.stop()

desc, icon = CODES.get(current.get("weather_code"), ("Unknown", "🌡️"))
st.title(f"{icon} {city['name']}: live weather + LSTM forecast")
st.caption(f"Local time of last reading: {current['time'].replace('T', ' ')}  |  "
           f"Source: Open-Meteo  |  Condition: {desc}")

c = st.columns(5)
c[0].metric("Temperature", f"{current['temperature_2m']:.1f} °C",
            f"feels like {current['apparent_temperature']:.1f} °C", delta_color="off")
c[1].metric("Humidity", f"{current['relative_humidity_2m']:.0f} %")
c[2].metric("Pressure", f"{current['surface_pressure']:.0f} hPa")
c[3].metric("Wind", f"{current['wind_speed_10m']:.1f} km/h")
c[4].metric("Rain (now)", f"{current['precipitation']:.1f} mm")

st.divider()

# ------------------------------------------------------- LSTM forecast
st.subheader("🧠 LSTM forecast")
key = f"bundle_{lat:.2f}_{lon:.2f}"
if key not in st.session_state:
    b = load_city(lat, lon)
    if b:
        st.session_state[key] = b
bundle = st.session_state.get(key)

if bundle is None:
    st.info(f"The LSTM has not been trained for {city['name']} yet. Training downloads about "
            "10 years of history and takes 1 to 3 minutes. It only happens once per city.")
    if st.button(f"🚀 Train LSTM for {city['name']}", type="primary"):
        bar = st.progress(0.0, text="Downloading history...")
        m = train_city(lat, lon, progress=lambda f, e, logs: bar.progress(
            f, text=f"Training epoch {e} (loss {logs['loss']:.4f})"))
        bar.progress(1.0, text="Done!")
        st.session_state[key] = load_city(lat, lon)
        st.rerun()
    st.stop()

today = pd.Timestamp(current["time"]).normalize()
tomorrow = today + pd.Timedelta(days=1)
win_tomorrow = daily.loc[today - pd.Timedelta(days=WINDOW - 1): today]
win_today = daily.loc[today - pd.Timedelta(days=WINDOW): today - pd.Timedelta(days=1)]
if len(win_tomorrow) < WINDOW or len(win_today) < WINDOW or tomorrow not in daily.index:
    st.error("Not enough recent data returned for this location.")
    st.stop()

lstm_tomorrow = predict_next(bundle, win_tomorrow)
lstm_today = predict_next(bundle, win_today)
off_tomorrow, off_today = float(daily.loc[tomorrow, "temp"]), float(daily.loc[today, "temp"])
m = bundle["metrics"]

a, b_, c_ = st.columns(3)
a.metric("LSTM: tomorrow's mean temp", f"{lstm_tomorrow:.1f} °C",
         f"{lstm_tomorrow - float(daily.loc[today, 'temp']):+.1f} °C vs today", delta_color="off")
b_.metric("Official forecast model: tomorrow", f"{off_tomorrow:.1f} °C")
c_.metric("Typical LSTM error on unseen days", f"± {m['lstm_mae']:.2f} °C",
          f"baseline ± {m['baseline_mae']:.2f} °C", delta_color="off")

fig, ax = plt.subplots(figsize=(10, 3.6))
hist = daily.loc[today - pd.Timedelta(days=WINDOW): today]
ax.plot(hist.index, hist["temp"], marker="o", label="Daily mean temperature")
ax.scatter([tomorrow], [lstm_tomorrow], color="red", marker="x", s=120, label="LSTM prediction", zorder=3)
ax.scatter([tomorrow], [off_tomorrow], color="green", s=70, label="Official forecast model", zorder=3)
ax.scatter([today], [lstm_today], color="red", marker="x", s=60, alpha=0.5, zorder=3)
ax.set_ylabel("°C")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.autofmt_xdate()
st.pyplot(fig)
st.caption("Today's point uses the forecast for the remaining hours of the day, so it is an estimate until midnight. "
           "The faint red × is the LSTM's guess for today made from the previous 14 days only.")

with st.expander("📈 Model details for this city"):
    st.write(f"Trained on {m['days']} days ({m['period']}); evaluated on the last "
             f"{m['test_days']} days it never saw.")
    st.table(pd.DataFrame({"MAE (°C)": [m["lstm_mae"], m["baseline_mae"]],
                           "RMSE (°C)": [m["lstm_rmse"], None],
                           "R²": [m["lstm_r2"], None]},
                          index=["LSTM", "Baseline (tomorrow = today)"]).round(3))
with st.expander("🧠 How it works"):
    st.markdown("""
1. Live weather comes from the Open-Meteo API (current conditions plus the last 14 days).
2. For each city, an LSTM is trained once on about 10 years of daily history.
3. The last 14 days of temperature, humidity, pressure, wind and rain are fed to the LSTM.
4. It outputs tomorrow's mean temperature. We compare it with the official numerical forecast and with a naive baseline.
""")
    st.info("Real forecasts use physics-based atmospheric simulations. This LSTM is a data-driven "
            "learner that finds patterns in history, so treat it as a learning project, not a safety tool.")
