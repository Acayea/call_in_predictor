# ❄️ Snow Crew Call-In Predictor

A machine learning project that predicts whether snow plow crews are likely to be called in based on current and historical weather data.  
Built with **Python**, **TensorFlow**, and real-world weather data from APIs like OpenWeatherMap or NOAA.

---

## 🚀 Overview

This project uses a neural network to analyze weather conditions — including temperature, snowfall, wind speed, humidity, and time of day — to estimate the probability that snow plow crews will be called in to work.

The goal is to help municipalities, contractors, and drivers anticipate upcoming shifts before the storm hits.

---

## 🧠 How It Works

1. **Collect Historical Data**
   - Pull past weather data using an API (e.g., OpenWeatherMap or NOAA).
   - Combine it with actual “call-in” records (e.g., dates/times when plowing occurred).

2. **Train a Neural Network**
   - Input features include weather metrics like:
     - Temperature  
     - Snowfall (inches)  
     - Wind speed  
     - Humidity  
     - Time of day / season  
   - The model predicts a binary outcome: `1 = called in`, `0 = not called in`.

3. **Make Predictions**
   - The trained model outputs the likelihood of a call-in for a given forecast.
   - Useful for scheduling, planning, or alert notifications.

---

## 🧩 Example Code

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
data = pd.read_csv("snow_call_data.csv")
X = data[["temperature", "snowfall", "wind_speed", "humidity", "hour"]]
y = data["called_in"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
