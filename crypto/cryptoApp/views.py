

# Create your views here.
import requests
import pandas as pd
from django.shortcuts import render

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
