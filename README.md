# ReinforecementLearningDRQN

## Reinforcement learning using Open AI gym and the following gym-stock-trading.
[Stock Trading github](https://github.com/duhfrazee/gym-stock-trading)
[Gym anytrading github](https://github.com/AminHP/gym-anytrading)

stocks -v 0 was rewritten to suite BTC-USD with DRQN refinforcement learning.

- Bitstamp_BTCUSD_2017_2022_minute_Capital.h5  
is a file obtained from bistamp recording minute info of BTCUSD
locate this under the data in the following path. 

` gym-anytrading/gym_anytrading/datasets/data `

The above directory was in python site packages so you might want to change it to your desired site. 
To do so, change the init.py in gym-anytrading/gym_anytrading


### Usage
1) make OHLC data from bitstamp. Time frame can be what you want. I tried this in h5 format but can be csv.
2) get gym-anytrading
   ```python
   pip install gym-anytrading
   ```
3) put h5/csv data under  gym-anytrading/gym_anytrading/datasets/data
   a bit hard to locate but is under the python site-packages 
4) Download DItradingtryout_Full_DRQN.py
then 

```python
python DItradingtryout_Full_DRQN.py
```
5) Default is just the stocks-vo Close and timeframe, but this can be adjusted to input more parameters.
   5-7 input parameters was about the sweetspot.
   Will update



# Not an investment/financial advice. Invest at your own risk
