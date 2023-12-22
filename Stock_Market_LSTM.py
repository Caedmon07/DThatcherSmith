import pandas as pd
import numpy as np
import yfinance as yf
import datetime
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import matplotlib.pyplot as plt

# Global Constants
SYMBOLS = ['GOOG', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = str(date.today())
NUM_DAYS = 5
SERVER = 'LAPTOP-B5J6KRT7'
DATABASE = 'PortfolioProject'

def send_email(subject, body):
    """Send an email."""
    sender_email = "auto.predict.stock@gmail.com"
    receiver_email = "danielsmith77@btinternet.com"
    app_password = "ygev gfji goca rxlu"
    

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as SERVER:
        SERVER.login(sender_email, app_password)
        SERVER.sendmail(sender_email, receiver_email, message.as_string())

def get_sql_credentials():
    """Get SQL Server credentials."""
    username = input("Enter your SQL Server username:")
    password = tkinter.simpledialog.askstring("Login", "Enter your SQL Server password:", show='*')
    
    return username, password

def plot_graph(data, title, xlabel, ylabel):
    """Plot a graph"""
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="predictions")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

    print("Data to be plotted: ", data)

def download_stock_data(SYMBOLS, START_DATE, END_DATE):
    """Download stock data."""
    username, password = get_sql_credentials()
    all_data = []

    for symbol in SYMBOLS:
        data = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False).reset_index()
        print(f"Downloaded data for {symbol} from yfinance:")
        print(data)

        engine = create_engine(f"mssql+pyodbc://{username}:{password}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server")
        data.to_sql(name=symbol, con=engine, if_exists='replace', index=False)

        all_data.append(data)

    return all_data

all_data = download_stock_data(SYMBOLS, START_DATE, END_DATE)

def preprocess_data(data, scaler_x, scaler_y):
    """Preprocess stock data."""
    x = data[["Open", "High", "Low", "Volume"]]
    y = data["Close"]

    x_scaled = scaler_x.fit_transform(x)
    x_scaled_reshaped = x_scaled.reshape((x_scaled.shape[0], x_scaled.shape[1], 1))
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    xtrain, xtest, ytrain, ytest = train_test_split(x_scaled_reshaped, y_scaled, test_size=0.2, random_state=42)

    return xtrain, xtest, ytrain, ytest

def build_lstm_model(input_shape):
    """Build LSTM model."""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(input_shape[0], 1), kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(125, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(150, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(200, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(150, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(250, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_lstm_model(model, xtrain, ytrain, xtest, ytest):
    """Train LSTM model."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(xtrain, ytrain, batch_size=1, epochs=50, validation_data=(xtest, ytest), callbacks=[early_stopping])

    return model

def predictions_lstm(model, scaler_x, scaler_y, data, num_days):
    """Generate predictions using the LSTM model."""
    xtrain, xtest, ytrain, ytest = preprocess_data(data, scaler_x, scaler_y)

    xtest_reshaped = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

    predictions = model.predict(xtest_reshaped)

    last_data_point = data.tail(1)[["Open", "High", "Low", "Volume"]].values.reshape(1, -1)
    last_data_point = scaler_x.transform(last_data_point)

    future_predictions = []

    for _ in range(num_days):
        future_prediction = model.predict(last_data_point.reshape(1, last_data_point.shape[1], 1))
        future_predictions.append(future_prediction[0][0])

        last_data_point = np.append(last_data_point[:, 1:], future_prediction[0][0].reshape(1, -1), axis=1)

    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return future_predictions

def plot_graph_LSTM(future_predictions):
    plot_graph(future_predictions, "LSTM Future Predictions", "Days", "Stock Price")

def predict_stock_random_forest(data, num_days):
    """Predict stock using a Random Forest model."""
    x = data[["Open", "High", "Low", "Volume"]]
    y = data["Close"]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(xtrain, ytrain.ravel())

    test_predictions = rf_model.predict(xtest)
    test_predictions = scaler_y.inverse_transform(test_predictions.reshape(-1, 1))

    last_data_point = data.tail(1)[["Open", "High", "Low", "Volume"]].values.reshape(1, -1)
    last_data_point = scaler_x.transform(last_data_point)

    future_predictions_rf = []

    for _ in range(num_days):
        future_prediction = rf_model.predict(last_data_point.reshape(1, -1))
        future_predictions_rf.append(future_prediction[0])

        last_data_point = np.append(last_data_point[:, 1:], future_prediction.reshape(1, -1), axis=1)

    future_predictions_rf = scaler_y.inverse_transform(np.array(future_predictions_rf).reshape(-1, 1))

    plot_graph(future_predictions_rf, "Random Forest Future Predictions", "Days", "Stock Price")

    return rf_model, future_predictions_rf

def predict_stock_arima(data, num_days):
    """Predict stock using an ARIMA model."""
    y = data["Close"]

    arima_model = ARIMA(y, order=(5, 1, 0))
    arima_results = arima_model.fit()

    future_predictions_arima = arima_results.predict(start=len(y), end=len(y) + num_days - 1, type='levels')

    plot_graph(future_predictions_arima, "ARIMA Future Predictions", "Days", "Stock Price")

    return np.squeeze(future_predictions_arima)

def predict_stock_xgboost(data, num_days):
    """Predict stock using an XGBoost model"""
    x = data[["Open", "High", "Low", "Volume"]]
    y = data["Close"]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgboost_model.fit(xtrain, ytrain.ravel())

    test_predictions = xgboost_model.predict(xtest)
    test_predictions = scaler_y.inverse_transform(test_predictions.reshape(-1, 1))

    last_data_point = data.tail(1)[["Open", "High", "Low", "Volume"]].values.reshape(1, -1)
    last_data_point = scaler_x.transform(last_data_point)

    future_predictions_xgboost = []

    for _ in range(num_days):
        future_prediction = xgboost_model.predict(last_data_point.reshape(1, -1))
        future_predictions_xgboost.append(future_prediction[0])

        last_data_point = np.append(last_data_point[:, 1:], np.array(future_predictions_xgboost).reshape(-1, 1))
    
    plot_graph(future_predictions_xgboost, "XGBoost Future Preditions", "Days", "Stock Price")

    return xgboost_model, future_predictions_xgboost

def blend_predictions(future_predictions, future_predictions_rf, future_predictions_arima, future_predictions_xgboost):
    """Blend predictions from different models."""
    blended_predictions = (future_predictions + future_predictions_rf + future_predictions_arima + future_predictions_xgboost) / 4.0
    return blended_predictions

def plot_candlestick_chart(all_data, future_predictions, symbols, num_days):
    """Plot a candlestick chart with historical and future predictions."""
    figure = go.Figure()

    for i, stock_data in enumerate(all_data):
        symbol = symbols[i]

        if i < len(future_predictions):
            individual_future_predictions = list(future_predictions[i])
        else:
            individual_future_predictions = []

        figure.add_trace(go.Candlestick(x=stock_data["Date"],
                                        open=stock_data["Open"].astype(float),
                                        high=stock_data["High"].astype(float),
                                        low=stock_data["Low"].astype(float),
                                        close=stock_data["Close"].astype(float),
                                        name=f'Historical Data - {symbol}',
                                        increasing_line_color='green', decreasing_line_color='red'))

        last_date = stock_data["Date"].iloc[-1]
        future_dates = [last_date + timedelta(days=j) for j in range(1, num_days + 1)]

        if individual_future_predictions:
            future_close_values = individual_future_predictions
            future_open_values = [future_close_values[0]] + future_close_values[:-1]
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

def live_stock_feed(symbols, all_data, scaler_x_lstm, scaler_y_lstm, model_lstm, rf_model, scaler_y_rf, arima_results, xgboost_model, num_days):
    """Live stock feed."""
    num_live_features = 4  # Set the number of live features
    scaler_x_lstm.fit(all_data[0][["Open", "High", "Low", "Volume"]].values[:, :num_live_features])
    scaler_y_lstm.fit(all_data[0]["Close"].values.reshape(-1, 1))

    for symbol, data in zip(symbols, all_data):
        print(f"{symbol}")
        print(data)

        x_live = data[["Open", "High", "Low", "Volume"]].values
        x_live = scaler_x_lstm.transform(x_live)

        print(x_live.shape)
        x_live_reshaped = x_live.reshape((x_live.shape[0], num_live_features, 1))

        prediction_lstm = model_lstm.predict(x_live_reshaped)
        prediction_rf = rf_model.predict(x_live)
        prediction_arima = arima_results.predict(start=len(data), end=len(data), type='levels')
        prediction_xgboost = xgboost_model.predict(x_live)

        predicted_close_lstm = scaler_y_lstm.inverse_transform(prediction_lstm.reshape(-1, 1))[0][0]
        predicted_close_rf = scaler_y_rf.inverse_transform(prediction_rf.reshape(-1, 1))[0][0]
        predicted_close_arima = prediction_arima[-1]
        predicted_close_xgboost = scaler_y_lstm.inverse_transorm(prediction_xgboost.reshape(-1, 1))[0][0]

        print(f"Predicted Close (LSTM) for the next day: {predicted_close_lstm}")
        print(f"Predicted Close (RF) for the next day: {predicted_close_rf}")
        print(f"Predicted close (ARIMA) for the next day: {predicted_close_arima}")
        print(f"Predicted close (XGBoost) for the next day: {predicted_close_xgboost}")

        send_email(f"Stock Prediction  ({symbols})",
                   f"Predicted Close (LSTM) for the next day: {predicted_close_lstm}\n"
                   f"Predicted Close (RF) for the next day: {predicted_close_rf}\n"
                   f"Predicted close (ARIMA) for the next day: {predicted_close_arima}\n"
                   f"Predicted close (XGBoost) for the next day: {predicted_close_xgboost}\n")

        time.sleep(86400)

all_data = download_stock_data(SYMBOLS, START_DATE, END_DATE)
scaler_x_lstm = MinMaxScaler()
scaler_y_lstm = MinMaxScaler()
model_lstm = build_lstm_model(input_shape=(4, 1))
future_predictions_lstm = predictions_lstm(model_lstm, scaler_x_lstm, scaler_y_lstm, all_data[0], NUM_DAYS)
future_predictions_rf = predict_stock_random_forest(all_data[0], NUM_DAYS)
future_predictions_arima = predict_stock_arima(all_data[0], NUM_DAYS)
future_predictions_xgboost = predict_stock_xgboost(all_data[0], NUM_DAYS)
blended_predictions = blend_predictions(future_predictions_lstm, future_predictions_rf, future_predictions_arima)
live_stock_feed(SYMBOLS, all_data, scaler_x_lstm, scaler_y_lstm, model_lstm=model_lstm, 
                 rf_model=future_predictions_rf, arima_results=future_predictions_arima, num_days=NUM_DAYS)
plot_candlestick_chart(all_data, [blended_predictions], SYMBOLS, NUM_DAYS)
print("LSTM Future Predictions")
print(future_predictions_lstm)
print("Random Forest Future Predictions")
print(future_predictions_rf)
print("ARIMA Future Predictions")
print(future_predictions_arima)
print("Blended Predictions")
print(blended_predictions)