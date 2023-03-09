

# Create your views here.
import os
import requests
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import csv
import json
import tensorflow as tf
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle


def home(request):
    return render(request, 'home.html')

def get_precip(gooddate):
    urlstart = 'http://api.wunderground.com/api/API_KEY/history_'
    urlend = '/q/Switzerland/Zurich.json'
    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()

    for summary in data['history']['dailysummary']:
        abc = ','.join((gooddate , summary['date']['year'],summary['date']['mon'],summary['date']['mday'],summary['precipm'], summary['maxtempm'], summary['meantempm'],summary['mintempm']))
        df = pd.DataFrame(data=abc)
        df.to_csv('/home/user/Desktop/2013_weather.csv', index=False)


def fetch_data_view(request):
    # Make the API request and get the data
    response = requests.get('https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=GBP&limit=30')
   # data = response.json()

    data = json.loads(response.text)



    # Create the CSV file and write the data to it
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'
    writer = csv.writer(response)
    for item in data['Data']['Data']:
       writer.writerow([item['time'],item['low'],item['high'],item['open'],item['close'],item['volumeto']])
    return response



def predict(request):
    filename = 'C:\\Users\\ahmad\\scaler.pkl'
    with open(filename, 'rb') as f:
        sc = pickle.load(f)
    # close the file
    f.close()

    colnames=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', ]
    data = pd.read_csv("C:\\Users\\ahmad\\Downloads\\data (30).csv", names=colnames, header=None)
    data['Adj Close'] = data['Close']
    # data['RSI'] = ta.rsi(data.Close, length=14)
    # data['EMAF'] = ta.ema(data.Close, length=20)
    # data['EMAM'] = ta.ema(data.Close, length=100)
    # data['EMAS'] = ta.ema(data.Close, length=150)
    #
    # data['Target'] = data['Adj Close'] - data.Open
    # data['Target'] = data['Target'].shift(-1)
    #
    # data['TargetClass'] = [1 if data.Target[i] > 0 else 0 for i in range(len(data))]
    #
    # data['TargetNextClose'] = data['Adj Close'].shift(-1)
    #
    # data.dropna(inplace=True)
    # data.reset_index(inplace=True)
    # data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    #
    # data_set = data.iloc[:, 0:11]  # .values
    # pd.set_option('display.max_columns', None)
    # data.shape
    #
    # # scale
    # data_set_scaled = sc.transform(data_set)
    #
    # # multiple feature from data provided to the model
    # X = []
    # # print(data_set_scaled[0].size)
    # # data_set_scaled=data_set.values
    # backcandles = 30
    # # print(data_set_scaled.shape[0])
    # for j in range(8):  # data_set_scaled[0].size):#2 columns are target not X
    #     X.append([])
    #     for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
    #         X[j].append(data_set_scaled[i - backcandles:i, j])
    #
    # # move axis from 0 to position 2
    # X = np.moveaxis(X, [0], [2])
    #
    # # Erase first elements of y because of backcandles to match X length
    # # del(yi[0:backcandles])
    # # X, yi = np.array(X), np.array(yi)
    # # Choose -1 for last column, classification else -2...
    # X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    # # y=np.reshape(yi,(len(yi),1))
    #
    # X = np.array([data_set_scaled[i - backcandles:i, :4].copy() for i in range(backcandles, len(data_set_scaled))])
    # model = tf.keras.models.load_model("C:\\Users\\ahmad\\PycharmProjects\\pythonProject15\\crypto\\cryptoApp\\bitcoinModel.h5")
    # print('hello world from predict')
    # predictions = model.predict(data)
    print ( data.shape)
    return render(request, 'home.html')



