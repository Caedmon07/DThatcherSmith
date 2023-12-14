import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px

today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")

stock_colors={'AAPL': 'green', 'GOOG': 'blue', 'MSFT': 'orange', 'AMZN': 'red'}

dfs = {}
for ticker, color in stock_colors.items():
    data = yf.download(ticker, start=start_date, end=end_date, progress=False).reset_index()
    data["Ticker"] = ticker
    data["Colour"] = color
    dfs[ticker] = data

data=pd.concat(list(dfs.values()), ignore_index=True)

#Creates a Candlestick Graph
figure = go.Figure()

for ticker in stock_colors.keys():
    subset = data[data['Ticker'] == ticker]
    color = stock_colors[ticker]
    figure.add_trace(go.Candlestick(x=subset['Date'],
                             open=subset['Open'],
                             high=subset['High'],
                             low=subset['Low'],
                             close=subset['Close'],
                             increasing_line_color=color,
                             decreasing_line_color=color,
                             text=subset['Ticker'] + '<br>Open: ' + subset['Open'].astype(str) + 
                                       '<br>Close: ' + subset['Close'].astype(str) + 
                                       '<br>High: ' + subset['High'].astype(str) + 
                                       '<br>Low: ' + subset['Low'].astype(str), 
                                       hoverinfo='text', 
                                       name=ticker))  

figure.update_layout(title = "Muliple Stock Price Analysis")
figure.update_xaxes(title_text='Date')
figure.update_yaxes(title_text='Stock Price (USD)')

#Creates a Bar Plot
figure2 = go.Figure()
for ticker in stock_colors.keys():
    subset = data[data['Ticker'] == ticker]
    color = stock_colors[ticker]
    figure2.add_trace(go.Bar(x=subset['Date'], 
                            y=subset['Close'], 
                            marker_color=color,
                            text=subset['Ticker'] + '<br>Close: ' + subset['Close'].astype(str),
                            hoverinfo='text',
                            name=ticker))
    figure2.update_layout(title = "Muliple Stock Price Analysis (Bar Plot)")
    figure2.update_xaxes(title_text='Date')
    figure2.update_yaxes(title_text='Closing Price (USD)')

#Creates a line graph
figure3=go.Figure()
for ticker, df in dfs.items():
    color = stock_colors[ticker]
    figure3.add_trace(go.Scatter(x=df['Date'],
                             y=df['Close'],
                             mode='lines',
                             line=dict(color=color),
                             text=df['Ticker'] + '<br>Close: ' + data['Close'].astype(str), 
                             hoverinfo='text', 
                             name=ticker))  

figure3.update_xaxes(title_text='Date')
figure3.update_yaxes(title_text="Close Price (USD)")
figure3.update_layout(title_text='Stock Prices Line Graph')

date_ranges = {
    '1 Month': 30,
    '3 Months': 90,
    '6 Months': 180,
    '1 Year': 365
}

updatemenus =[]

for stock in stock_colors.keys():
    buttons = []
    for label, days in date_ranges.items():
        button = dict(label=label,
                      method='relayout',
                      args=[{'xaxis': {'range': [today - timedelta(days=days), today]}}])
        buttons.append(button)

updatemenus.append(dict(type='buttons',
                        showactive=True,
                        buttons=buttons,
                        x=0.1,
                        xanchor='left',
                        y=1.15,
                        yanchor='top'))
figure3.update_layout(updatemenus=updatemenus)

figure.show()
figure2.show()
figure3.show()