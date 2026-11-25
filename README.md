# Live Weather Website with LSTM Forecasting

A real-time weather website. Search any city in the world, see live conditions, and get an
LSTM deep-learning forecast of tomorrow's temperature, compared against the official
forecast model and a naive baseline.

## Quick start
```
pip install -r requirements.txt
python pretrain.py Delhi Mumbai Chennai     # optional: train cities ahead of time
streamlit run app.py                        # opens the website in your browser
```
Needs internet (live data from the free Open-Meteo API, no key needed).
The first time you pick a city that has no model, click "Train LSTM" (1-3 minutes, once per city).

## Files
| File | Purpose |
|---|---|
| `app.py` | The website (live weather, LSTM forecast, charts) |
| `core.py` | API calls, training, prediction |
| `weather_lstm.py` | Standalone research script: train + evaluate LSTM vs GRU vs baseline for one place |
| `pretrain.py` | Train models for several cities before the exhibition |
| `app_offline.py` | Simple demo of the standalone script's model (run `weather_lstm.py` first) |
| `EXHIBITION_GUIDE.md` | Pitch, demo flow, likely questions and answers |

## Method
1. History: ~10 years of hourly data resampled to daily (temperature, humidity, pressure, wind, rain)
2. Chronological split 70/15/15 (no shuffling); scaler fitted on the training part only
3. 14-day sliding windows -> next-day mean temperature, input shape (samples, 14, 5)
4. LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1); Adam, MSE, early stopping
5. Live: the last 14 days from the forecast API are fed to the model to predict tomorrow
6. Evaluation on unseen test days: MAE, RMSE, R2 vs a "tomorrow = today" baseline

## Deploying online (optional)
Push to GitHub and deploy on Streamlit Community Cloud. TensorFlow is large, and on-demand
training is slow on free tiers, so pre-train cities and commit the `models/` and `data/` folders.
