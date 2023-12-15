import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import tensorflow as tf
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tkinter as tk
from tkinter import ttk
import os
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import pyodbc
import tkinter.simpledialog

def get_sql_credentials():
    username = tkinter.simpledialog.askstring("Login", "Enter your SQL Server username:")
    password = tkinter.simpledialog.askstring("Login", "Enter your SQL Server password:", show='*')
    return username, password

# Update your SQL Server connection details
server = 'LAPTOP-B5J6KRT7'
database = 'PortfolioProject'

# Function to download or load stock data
def download_stock_data(symbol, start_date, end_date):
    data_filename = f"{symbol}_stock_data.csv"

    if os.path.exists(data_filename):
        # Load data from file
        data = pd.read_csv(data_filename, parse_dates=['Date'], index_col='Date', date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        print("Loaded data from file:")
        print(data)
    else:
        # Get SQL Server credentials from the user
        username, password = get_sql_credentials()

        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False).reset_index()
        print("Downloaded data from yfinance:")
        print(data)

        # Save data to SQL Server
        engine = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")
        data.to_sql(name=symbol, con=engine, if_exists='replace', index=False)

        # Save data to file for future use
        data.to_csv(data_filename, index=False, date_format="%Y-%m-%d", columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        print("Saved data to file.")

    return data

def predict_stock(data, num_days, epochs_label):
    print("Loaded data: ")
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
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(500))
    model.add(Dense(250))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Set up a callback to capture the number of epochs
    class EpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epochs_label.config(text=f"Epochs: {epoch + 1}")

    callback = EpochCallback()

    model.fit(xtrain, ytrain, batch_size=1, epochs=num_days, callbacks=[callback])

    # Predicting on the test set
    predictions = model.predict(xtest)

    # Predicting future stock prices
    last_data_point = data.tail(1)[["Open", "High", "Low", "Volume"]].values.reshape(1, -1)
    last_data_point = scaler_x.transform(last_data_point)
    future_predictions = []

    for _ in range(num_days):
        future_prediction = model.predict(last_data_point)
        future_predictions.append(future_prediction[0][0])
        last_data_point = last_data_point[:, 1:]
        last_data_point = np.append(last_data_point, future_prediction[0][0].reshape(1, -1), axis=1)

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))[0]

    return predictions, future_predictions

def plot_candlestick_chart(data, future_data):
    figure = go.Figure()

    # Plot historical data
    figure.add_trace(go.Candlestick(x=data["Date"],
                                    open=data["Open"],
                                    high=data["High"],
                                    low=data["Low"],
                                    close=data["Close"],
                                    name='Historical Data'))

    # Plot future predictions
    last_close_price = data["Close"].iloc[-1]
    future_dates = [pd.to_datetime(data.index[-1]) + timedelta(days=i) for i in range(1, len(future_data) + 1)]
    future_open_values = [last_close_price] + [last_close_price + sum(future_data[:i]) for i in range(1, len(future_data))]
    future_close_values = [last_close_price + sum(future_data[:i]) for i in range(1, len(future_data) + 1)]
    future_high_values = [max(last_close_price, last_close_price + val) for val in future_data]
    future_low_values = [min(last_close_price, last_close_price + val) for val in future_data]

    figure.add_trace(go.Candlestick(x=future_dates,
                                    open=future_open_values,
                                    high=future_high_values,
                                    low=future_low_values,
                                    close=future_close_values,
                                    name='Future Predictions'))

    figure.update_layout(title="Stock Price Analysis with Predictions", xaxis_rangeslider_visible=False)
    figure.show()

def on_predict_button_click():
    symbol = symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    num_days = int(num_days_entry.get())

    # Convert the start and end dates to the required format
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Clear previous epoch information
    epochs_label.config(text="")

    data = download_stock_data(symbol, start_date, end_date)

    # Displaying the first future prediction value as an example
    result_label.config(text=f"Predicted Close Price (Future): ...")

    predictions, future_predictions = predict_stock(data, num_days, epochs_label)
    plot_candlestick_chart(data, future_predictions)

    # Displaying the first future prediction value as an example
    result_label.config(text=f"Predicted Close Price (Future): {future_predictions[0]:.2f}")

# GUI setup
root = tk.Tk()
root.title("Stock Prediction App")

# Input fields
symbol_label = ttk.Label(root, text="Stock Symbol:")
symbol_label.pack()
symbol_entry = ttk.Entry(root)
symbol_entry.pack()

start_date_label = ttk.Label(root, text="Start Date (YYYY-MM-DD):")
start_date_label.pack()
start_date_entry = ttk.Entry(root)
start_date_entry.pack()

end_date_label = ttk.Label(root, text="End Date (YYYY-MM-DD):")
end_date_label.pack()
end_date_entry = ttk.Entry(root)
end_date_entry.pack()

num_days_label = ttk.Label(root, text="Number of Days to Predict:")
num_days_label.pack()
num_days_entry = ttk.Entry(root)
num_days_entry.pack()

epochs_label = ttk.Label(root, text="Epochs: ")
epochs_label.pack()

predict_button = ttk.Button(root, text="Predict", command=on_predict_button_click)
predict_button.pack()

# Result label
result_label = ttk.Label(root, text="")
result_label.pack()

root.mainloop()