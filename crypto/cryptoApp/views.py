

# Create your views here.
import requests
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import csv
import json
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle


def home(request):
    return render(request, 'test.html')

def get_precip(gooddate):
    urlstart = 'http://api.wunderground.com/api/API_KEY/history_'
    urlend = '/q/Switzerland/Zurich.json'
    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()

    for summary in data['history']['dailysummary']:
        abc = ','.join((gooddate,summary['date']['year'],summary['date']['mon'],summary['date']['mday'],summary['precipm'], summary['maxtempm'], summary['meantempm'],summary['mintempm']))
        df = pd.DataFrame(data=abc)
        df.to_csv('/home/user/Desktop/2013_weather.csv', index=False)


def fetch_data_view(request):
    # Make the API request and get the data
    response = requests.get('https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=GBP&limit=10')
   # data = response.json()

    data = json.loads(response.text)



    # Create the CSV file and write the data to it
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'
    writer = csv.writer(response)
    for item in data['Data']['Data']:
       writer.writerow([item['time'],item['low'],item['high'],item['open'],item['close'],item['volumeto']])
    return response



def pedict(request):
    model = tf.load('bitcoinModel.h5')
    print('hello world from predict')
    return render(request, 'test1.html')



