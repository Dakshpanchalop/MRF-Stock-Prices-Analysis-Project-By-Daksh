# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:18:23 2024

@author: praka
"""
import pandas as pd

import streamlit as st
from datetime import date
import os
os.system('pip install yfinance')

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set start and end dates
START = "2002-07-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title of the app
st.title('Stock Forecast App')

# Select a stock for prediction
stocks = ('MRF.NS','AMZN')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Select prediction period
n_years = st.slider('Years of prediction:', 1, 22)
period = n_years * 365

# Cache the data loading function to optimize performance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Prepare data for Prophet model
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe and make predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Prediction Section
st.subheader('Make Predictions')
prediction_date = st.date_input("Select a date for prediction", value=date.today())
prediction_days = (prediction_date - data['Date'].max().date()).days

if prediction_days > 0:
    prediction = forecast[forecast['ds'] == pd.Timestamp(prediction_date)]
    
    if not prediction.empty:
        st.write(f"Predicted stock price on {prediction_date}: {prediction['yhat'].values[0]:.2f}")
    else:
        st.write(f"No forecast data available for {prediction_date}")
else:
    st.write(f"Please select a future date for prediction.")

