'''
utils.py
this is also under
\\wsl.localhost\Ubuntu\home\name\miniconda3\lib\python3.9\site-packages\gym_anytrading\datasets
the commented out lines are the original datasets 
'''

from .utils import load_dataset as _load_dataset


# Load FOREX datasets
#FOREX_EURUSD_1H_ASK = _load_dataset('FOREX_EURUSD_1H_ASK', 'Time')

# Load Stocks datasets
#STOCKS_GOOGL = _load_dataset('STOCKS_GOOGL', 'Date')
#GOOG = _load_dataset('GOOG', 'Date')
Bitstamp_BTCUSD_2017_2022_minute_Capital=_load_dataset('Bitstamp_BTCUSD_2017_2022_minute_Capital', 'Date')
#BTCUSD_2017_2022_caluculated_10=_load_dataset('BTCUSD_2017_2022_caluculated_10', 'Date')
