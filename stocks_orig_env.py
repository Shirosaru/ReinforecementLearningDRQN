'''
Put this under env
\\wsl.localhost\Ubuntu\home\name\miniconda3\lib\python3.9\site-packages\gym_anytrading\envs

line 48: signal_features has the input columns. Make your own parameters input them and reinforcement learn them!

'''


import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class StocksstocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        
        print("window_size",window_size)
        print("frame_bound", frame_bound[1])
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
#        print(diff)
        support = self.df.loc[:, 'support'].to_numpy()
        support=np.resize(support, int(self.frame_bound[1]))
        aboveEMA15 = self.df.loc[:, 'aboveEMA15'].to_numpy()
        aboveEMA15=np.resize(aboveEMA15, int(self.frame_bound[1]))
        overResistance = self.df.loc[:, 'overResistance'].to_numpy()
        overResistance=np.resize(overResistance, int(self.frame_bound[1]))
        belowResistance = self.df.loc[:, 'belowResistance'].to_numpy()
        belowResistance=np.resize(belowResistance, int(self.frame_bound[1]))
        STC_INDICATOR = self.df.loc[:, 'STC_INDICATOR'].to_numpy()
        STC_INDICATOR=np.resize(STC_INDICATOR, int(self.frame_bound[1]))

        #Indicators to include overResistance, belowResistance, STC_INDICATOR

        signal_features = np.column_stack((prices, diff, support, aboveEMA15, overResistance, belowResistance, STC_INDICATOR))
#s        signal_features = np.column_stack((prices, diff))


        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
