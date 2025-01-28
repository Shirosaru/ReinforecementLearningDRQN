'''
This is an __inti__.py file that is found on
\\wsl.localhost\Ubuntu\home\name\miniconda3\lib\python3.9\site-packages\gym_anytrading\__init__.py
Need to change this to the following to accept the original file

'''

from gym.envs.registration import register
from copy import deepcopy

from . import datasets




register(
    id='stocks-v0',
    entry_point='gym_anytrading.envs:StocksEnv',
    kwargs={
        'df': deepcopy(datasets.Bitstamp_BTCUSD_2017_2022_minute_Capital),
        'window_size': 30,
        'frame_bound': (30, len(datasets.Bitstamp_BTCUSD_2017_2022_minute_Capital))
    }
)
