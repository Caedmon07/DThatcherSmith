import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("Coffee_export.csv")
print(data.head())
x = data["1990"].values.reshape(-1, 1)
y = data["1995"]

model = LinearRegression()
model.fit(x, y)
x_range = np.linspace(x.min(), x.max(), 1000)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter (data, x="1990", y="1995", opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name="Linear Regression"))

print(y_range)

fig.show()
