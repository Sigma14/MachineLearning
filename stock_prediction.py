import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error

# Fetch historical stock data
symbol = "AAPL"
data = yf.download(symbol, start="2010-01-01", end="2022-01-01")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Create a dataset with a specific time step
time_step = 60
X = []
y = []

for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i - time_step : i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Create the LSTM model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=100, batch_size=32)

# Prepare the test dataset
test_data = yf.download(symbol, start="2022-01-01", end="2022-12-31")
scaled_test_data = scaler.transform(test_data["Close"].values.reshape(-1, 1))

X_test = []
for i in range(time_step, len(scaled_test_data)):
    X_test.append(scaled_test_data[i - time_step : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Calculate the mean squared error
mse = mean_squared_error(test_data["Close"][time_step:], predicted_stock_price)
print("Mean Squared Error:", mse)
