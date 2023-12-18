import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import tensorflow as tf
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import pyodbc
import tkinter.simpledialog
import time


symbols = ['AAPL', 'GOOG', 'MSFT']
start_date = '2020-01-01'
end_date = date.today()
num_days = 365

def get_sql_credentials():
    username = input("Enter your SQL Server username:")
    password = tkinter.simpledialog.askstring("Login","Enter your SQL Server password:", show='*')
    return username, password

# Update your SQL Server connection details
server = 'LAPTOP-B5J6KRT7'
database = 'PortfolioProject'

# Function to download or load stock data
def download_stock_data(symbols, start_date, end_date):
    # Get SQL Server credentials from the user
    username, password = get_sql_credentials()
    
    all_data = []

    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False).reset_index()
        print(f"Downloaded data for {symbol} from yfinance:")
        print(data)

        # Save data to SQL Server
        engine = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")
        data.to_sql(name=symbol, con=engine, if_exists='replace', index=False)
        
        all_data.append(data)

    return all_data

def predict_stock(all_stock_data, num_days):
    all_future_predictions = []

    for data in all_stock_data:
        print("Loaded data:")
        print(data)

        x = data[["Open", "High", "Low", "Volume"]]
        y = data["Close"]

        # Normalize input features and target variable
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y.values.reshape(-1, 1))

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
        model.add(Activation('relu'))
        model.add(LSTM(64, return_sequences=False))
        model.add(Activation('relu'))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dense(750))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(300))
        model.add(Activation('relu'))
        model.add(Dense(250))
        model.add(Activation('relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1)), ytrain, batch_size=1, epochs=1)

        predictions = model.predict(xtest.reshape((xtest.shape[0], xtest.shape[1], 1)))

        # Predicting future stock prices
        last_data_point = data.tail(1)[["Open", "High", "Low", "Volume"]].values.reshape(1, -1)
        last_data_point = scaler_x.transform(last_data_point)

        future_predictions = []

        for _ in range(num_days):
            future_prediction = model.predict(last_data_point.reshape(1, last_data_point.shape[1], 1))
            future_predictions.append(future_prediction[0][0])

            # Update the last data point for the next iteration
            last_data_point = np.append(last_data_point[:, 1:], future_prediction[0][0].reshape(1, -1), axis=1)

        # Inverse transform the predictions to the original scale
        future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        print("Predictions:")
        print(predictions)
        print("Future Predictions:")
        print(future_predictions)

        all_future_predictions.append(future_predictions)

    return np.squeeze(all_future_predictions)

# Modify the plot_candlestick_chart function to handle a 1D array directly
def plot_candlestick_chart(all_data, future_predictions, symbols, num_days):
    figure = go.Figure()

    for i, stock_data in enumerate(all_data):
        symbol = symbols[i]
        individual_future_predictions = future_predictions[i]

        # Plot historical data
        figure.add_trace(go.Candlestick(x=stock_data["Date"],
                                        open=stock_data["Open"].astype(float),
                                        high=stock_data["High"].astype(float),
                                        low=stock_data["Low"].astype(float),
                                        close=stock_data["Close"].astype(float),
                                        name=f'Historical Data - {symbol}',
                                        increasing_line_color='green', decreasing_line_color='red'))

        # Plot future predictions
        last_date = stock_data["Date"].iloc[-1]
        future_dates = [last_date + timedelta(days=j) for j in range(1, num_days + 1)]

        future_close_values = individual_future_predictions
        future_open_values = future_close_values[:-1]
        future_high_values = [max(float(last_close), float(future_close)) for last_close, future_close in
                            zip(future_close_values[:-1], future_close_values)]
        future_low_values = [min(float(last_close), float(future_close)) for last_close, future_close in
                            zip(future_close_values[:-1], future_close_values)]

        figure.add_trace(go.Candlestick(x=future_dates,
                                        open=future_open_values,
                                        high=future_high_values,
                                        low=future_low_values,
                                        close=future_close_values,
                                        name=f'Future Predictions - {symbol}',
                                        increasing_line_color='blue', decreasing_line_color='orange'))

    figure.update_layout(title=f"Stock Price Analysis with Predictions", xaxis_rangeslider_visible=False)
    figure.show()

def live_stock_feed(symbols, all_data, scaler_x, scaler_y, model, num_days):
    while True:
        for symbol, data in zip(symbols, all_data):
            print(f"{symbol}")
            print(data)

            x_live = data[["Open", "High", "Low", "Volume"]].values
            x_live = scaler_x.transform(x_live.reshape(1, -1))

            x_live = x_live.reshape((1, x_live.shape[0], x_live.shape[1]))

            prediction = model.predict(x_live)

            predicted_close = scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]

            print(f"Predicted Close for the next day: {predicted_close}")

            time.sleep(30)
            
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
model = Sequential()

all_data = download_stock_data(symbols, start_date, end_date)
future_predictions = predict_stock(all_data, num_days)
plot_candlestick_chart(all_data, future_predictions, symbols, num_days)

live_stock_feed(symbols, all_data, scaler_x, scaler_y, model, num_days)