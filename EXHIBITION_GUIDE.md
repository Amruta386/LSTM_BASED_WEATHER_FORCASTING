# Exhibition Guide: Weather Prediction using LSTM

## Setup checklist (do this the day before)
1. `pip install -r requirements.txt`
2. `python pretrain.py Delhi Mumbai Chennai <your city>` (pre-trains so the demo is instant)
3. `streamlit run app.py` and try each city
4. Also run `python weather_lstm.py` once and keep `results.png` as a backup slide
5. The website needs internet for LIVE data, so check the venue Wi-Fi or use a phone hotspot

## 30-second pitch
"Weather is a sequence: today depends on the last few days. I built a deep learning model,
an LSTM, that reads the past 14 days of temperature, humidity, pressure, wind and rain, and
predicts tomorrow's temperature. I trained it on years of real data and tested it on days it
had never seen. You can try it yourself on this screen."

## 2-minute demo flow
1. Type a visitor's home city in the sidebar; live conditions appear instantly.
2. Show the LSTM's prediction for tomorrow next to the official forecast and the chart.
3. Open "Model details" and show the error on unseen days vs the baseline.
4. Open "How it works" and walk through the steps; show `results.png` for LSTM vs GRU.

## Likely questions and answers
**What is an LSTM?**
A type of recurrent neural network with gates that decide what to remember, forget and output,
so it can learn patterns across time steps, unlike a normal network that treats each input separately.

**Why not a normal neural network or linear regression?**
They ignore the order of the data. An LSTM sees the whole 14-day sequence and its trend.

**What data did you use and how much?**
Daily weather (temperature, humidity, pressure, wind, rain) from the Open-Meteo archive, about 11 years.

**How did you avoid cheating / overfitting?**
The data is split by time (train / validation / test), never shuffled. The scaler is fit on training data only.
Dropout and early stopping stop the model from memorising.

**How accurate is it?**
Quote your own MAE / RMSE / R² from the console output, and compare with the "tomorrow = today" baseline.
Be honest: temperature is quite predictable, so the baseline is already strong.

**Can it predict rain or storms?**
Not in this version; it predicts temperature. Rain could be added as a classification task.

**How would you improve it?**
More stations, hourly data, multi-day forecasting, attention or transformer models.

**Why is the window 14 days?**
A design choice. You can test 7 / 14 / 30 and compare the errors (change `WINDOW` in the script).

**What is the difference between LSTM and GRU?**
GRU has fewer gates, so it's simpler and faster; LSTM has a separate memory cell.
In this project the two perform similarly (see the results table).

**Is this how real weather forecasting works?**
No. Real forecasts use physics-based simulations of the atmosphere. This is a data-driven
approach that learns patterns from history only.
