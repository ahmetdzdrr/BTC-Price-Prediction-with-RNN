import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
from matplotlib.ticker import FormatStrFormatter

plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
params = {
    "vs_currency": "usd",
    "from": 1451606400,
    "to": 1679740800,
}

response = requests.get(url, params=params)
data = response.json()

timestamps = [entry[0] for entry in data["prices"]]
dates = pd.to_datetime(timestamps, unit="ms")
prices = [entry[1] for entry in data["prices"]]

df = pd.DataFrame({"Date": dates, "Close": prices})
print(df.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
predict_days = 60

x_train, y_train = [], []

for x in range(predict_days, len(scaled_df)):
    x_train.append(scaled_df[x-predict_days:x, 0])
    y_train.append(scaled_df[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=32))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=32)

test_start = dt.datetime(2016, 1, 1)
test_end = dt.datetime(2023,1,1)

test_df = df[df['Date'] >= test_start]

model_inputs = test_df["Close"].values.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(predict_days, len(model_inputs)):
    x_test.append(model_inputs[x-predict_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], 1])

predict_prices = model.predict(x_test)
predict_prices = scaler.inverse_transform(predict_prices)

actual_prices = test_df["Close"].values

plt.plot(actual_prices, label="Actual Price")
plt.plot(predict_prices, label="Predicted Price")
plt.title("Bitcoin-USD price prediction analysis")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=30, freq='D')

future_prices = []
x_test = scaled_df[-predict_days:].reshape(1, predict_days, 1)

for _ in range(30):
    prediction = model.predict(x_test)
    future_prices.append(scaler.inverse_transform(prediction)[0, 0])
    x_test = np.concatenate((x_test[:, 1:, :], prediction.reshape(1, 1, 1)), axis=1)

future_df = pd.DataFrame({'Date': future_dates, 'Close': future_prices})

plt.plot(df['Date'], df['Close'], label="Historical Price")
plt.plot(future_df['Date'], future_df['Close'], label="Predicted Price (Next 30 Days)")
plt.title("Bitcoin-USD Price Prediction for the Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
