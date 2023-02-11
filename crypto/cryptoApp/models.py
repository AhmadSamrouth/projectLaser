import requests
import json
from django.db import models

class OHLCV(models.Model):
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.DecimalField(max_digits=10, decimal_places=2)



api_key = 'e9def2c8d36acfc8d7b5f5a78c17439290058c6c786f5834102df8036b01b1ca'
fsym = 'BTC'
tsym = 'USD'
interval = 'daily'

url = f'https://min-api.cryptocompare.com/data/v2/histo{interval}?fsym={fsym}&tsym={tsym}&api_key={api_key}'
response = requests.get(url)
data = json.loads(response.text)

