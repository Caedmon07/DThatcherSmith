import numpy as np
import yfinance as yf
import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px

today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")

stock_tickers=['AAPL', 'GOOG', 'MSFT', 'AMZN']

dfs = []
for ticker in stock_tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False).reset_index()
    data["Ticker"] = ticker
    dfs.append(data)

data=pd.concat(dfs, ignore_index=True)

figure = go.Figure()

for ticker in stock_tickers:
    stock_data = data[data['Ticker'] == ticker]

figure.add_trace(go.Candlestick(x=stock_data['Date'],
                             open=stock_data['Open'],
                             high=stock_data['High'],
                             low=stock_data['Low'],
                             close=stock_data['Close'],
                             text=stock_data['Ticker'] + '<br>Open: ' + stock_data['Open'].astype(str) + 
                                       '<br>Close: ' + stock_data['Close'].astype(str) + 
                                       '<br>High: ' + stock_data['High'].astype(str) + 
                                       '<br>Low: ' + stock_data['Low'].astype(str), 
                                       hoverinfo='text', 
                                       name=data['Ticker']))  

figure.update_layout(title = "Muliple Stock Price Analysis")
figure.update_xaxes(title_text='Date')
figure.update_yaxes(title_text='Stock Price (USD)')

figure.show()
