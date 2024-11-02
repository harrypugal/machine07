from keras.models import load_model
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


model = load_model('stock_clos.h5')


def preprocess_data(file_path):
    if file_path.endswith('.xls') or file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path, engine='openpyxl')
    else:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date')
        
    close_price = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_price)

    time_stamp = 60
    stock_data = []
    for i in range(time_stamp, len(scaled_data)):
        stock_data.append(scaled_data[i - time_stamp:i, 0])
    
    stock_data = np.array(stock_data)
    stock_data = np.reshape(stock_data, (stock_data.shape[0], stock_data.shape[1], 1))

    prediction = model.predict(stock_data)
    prediction = scaler.inverse_transform(prediction)
    df = pd.DataFrame(prediction, columns=['Predicted Price'])
    
    # Adding the corresponding dates to the DataFrame
    #last_date = data['Date'].iloc[1]
    last_date = data.index[-1]  # Get the last date from the fetched data

    df['Date'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df), freq='B')
    df = df[['Date', 'Predicted Price']]  # Reorder columns
    df['Date'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df), freq='B')
    df = df[['Date', 'Predicted Price']]  # Reorder columns

    return df


def fetch_stock_data(ticker, start_date, end_date):
    # Fetch stock data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    

    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date')

    close_price = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_price)

    time_stamp = 60
    stock_data = []
    for i in range(time_stamp, len(scaled_data)):
        stock_data.append(scaled_data[i - time_stamp:i, 0])
    
    stock_data = np.array(stock_data)
    stock_data = np.reshape(stock_data, (stock_data.shape[0], stock_data.shape[1], 1))

    prediction = model.predict(stock_data)
    prediction = scaler.inverse_transform(prediction)
    df = pd.DataFrame(prediction, columns=['Predicted Price'])
    
    # Adding the corresponding dates to the DataFrame
    last_date = data.index[-1]  # Get the last date from the fetched data
    df['Date'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df), freq='B')
    df = df[['Date', 'Predicted Price']]  # Reorder columns

    return df
