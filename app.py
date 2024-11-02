from keras.models import load_model
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from predict import fetch_stock_data, preprocess_data


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    # File upload handling
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)

    if file and (file.filename.endswith('.xls') or file.filename.endswith('.xlsx')):
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        prediction = preprocess_data(file_path)
        
        # Convert predictions DataFrame to HTML table
        prediction_html = prediction.to_html(classes='table table-striped', index=False)

        return render_template('result.html', prediction_table=prediction_html)

@app.route('/manual', methods=['POST'])
def manual_input():
    # Stock ticker input handling
    stock_ticker = request.form['ticker']
    start = request.form['start_date']
    end = request.form['end_date']
    prediction = fetch_stock_data(stock_ticker, start, end)

    # Convert predictions DataFrame to HTML table
    prediction_html = prediction.to_html(classes='table table-striped', index=False)

    return render_template('result.html', prediction_table=prediction_html)

if __name__ == '__main__':
    app.run(debug=True)
 

