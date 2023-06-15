from gym import Env
from gym.spaces import Discrete, Box, Tuple, Sequence
import numpy as np
import pandas as pd

from gambling import Gambling
from data_collector import DataCollector
from trader import Return_rate_cal
from state import Observe_State_Change

class FundENV(Env):

    def __init__(self):
        # Variables
        original_value = 10000
        self.time_start = "01-01-2019"
        self.time_end = "31-12-2022"
        currencies = ["BTC", "ETH", "BNB", "XRP", "LTC", "DOGE", "USDT", "USDC", "ADA"]
        self.trader_amount = 8

        # Create an instances
        data_collector = DataCollector()

        # Actions we can take: gambling hand, amount
        self.action_space = Tuple((Discrete(4), Box(low=0, high=1, shape=(1,), dtype=np.float32)))

        # Observations rewards from previous gambling, fund value before gambling, trader behavior
        list_1 = [Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                  Box(low=0, high=100000000, shape=(1,), dtype=np.float32)]

        # there are n traders
        list_2 = [Box(low=-1, high=1, shape=(1,),
                      dtype=np.float32) for _ in range(self.trader_amount)]
        element_space = list_1+list_2
        self.observation_space = Tuple(element_space)

        # Set the value of SCT
        self.value = original_value
        # Set winning
        self.winning = 0
        # Set start state
        state_updater = Observe_State_Change()
        self.state = state_updater.state_update(winning=self.winning, value=self.value,
                                                data_list=[0 for _ in range(self.trader_amount)])

        # Set trading time length
        self.training_time = data_collector.time_cal(time_start=self.time_start, time_end=self.time_end)
        # timer
        self.timer = 0

        # build DataFrame for the market
        data_collector.data_collection(currencies, time_start=self.time_start, time_end=self.time_end)
        df = data_collector.dataframe_producing(currencies, time_start=self.time_start, time_end=self.time_end)
        df['SCT'] = np.nan
        df['SCT'][self.timer] = original_value
        self.df = df

        print('Initialize complete')
        print(f'action_space: {self.action_space}')
        print(f'observation_space: {self.observation_space}')
        print(f'initial state: {self.state}')
        print(f'training_time: {self.training_time}')
        print(f'initial value: {self.value}')
        print('DataFrame of market')
        print(self.df)

    def step(self, action):
        if self.timer == 0:
            self.timer += 1
        original_value = self.value
        # action-gambling
        # apply action gambling
        gambling = Gambling()
        self.winning = gambling.playing_baccarat(action[0], action[1][0], self.value)
        # value calculation
        self.value += self.winning
        # update DataFrame of Market
        data_collector = DataCollector()
        self.df['SCT'][self.timer] = self.value
        # trading
        # the information the trader can get
        df_current = self.df.copy().iloc[0:self.timer+1]
        print(df_current)
        # calculate the investment amount on SCT
        return_rate_cal = Return_rate_cal()
        market_data = return_rate_cal.market_cal(self.timer, df_current, start=self.time_start, end=self.time_end)
        # storing the investments on SCT
        invest_data_list = [0.0 for _ in range(self.trader_amount)]
        for i in range(self.trader_amount):
            invest_data_list[i] = market_data[i].iloc[self.timer, -3]
        # calculate current value
        self.value += sum(invest_data_list)

        # state observing(the percentage of value from gambling, the current-value, the percentage of value from trader
        observe_state_change = Observe_State_Change
        self.state = observe_state_change.state_update(winning=self.winning/original_value,
                                                       value=original_value,
                                                       data_list=[x/original_value for x in invest_data_list])

        # calculating reward
        # calculate SCT invest difference
        invest_dif_list = [0.0 for _ in range(self.trader_amount)]
        for i in range(self.trader_amount):
            invest_dif_list[i] = market_data[i].loc[self.timer, -3] - market_data[i].loc[self.timer-1, -3]
        # set the reward
        self.reward = sum(invest_dif_list)

        # timer
        self.timer += 1

        # Check timer
        if self.timer >= self.training_time:
            Is_Done = True
        else:
            Is_Done = False

        # set placeholder
        info = {}

        # return step information
        return self.value, self.state, self.reward, info, self.timer, Is_Done, self.df
    def render(self):
        # implement with visualization
        pass

    def reset(self):
        original_value = 10000
        time_start = "01-01-2019"
        time_end = "31-12-2022"
        currencies = ["BTC", "ETH", "BNB", "XRP", "LTC", "DOGE", "USDT", "USDC", "ADA"]

        self.trader_amount = 8
        self.value = original_value
        self.winning = 0
        state_updater = Observe_State_Change()
        self.state = state_updater.state_update(winning=self.winning, value=self.value,
                                                data_list=[0 for _ in range(self.trader_amount)])
        data_collector = DataCollector()
        self.training_time = data_collector.time_cal(time_start=time_start, time_end=time_end)
        df = data_collector.dataframe_producing(currencies, time_start=time_start, time_end=time_end)
        self.df = data_collector.add_SCT_to_df(df=df, data=original_value, date=self.timer, time_start=time_start)

        return self.value, self.state, self.training_time, self.df, self.winning