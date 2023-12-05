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

data = data[["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

print(data)