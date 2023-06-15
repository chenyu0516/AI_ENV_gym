from gym import Env
from gym.spaces import Discrete, Box, Tuple, Sequence
import numpy as np

from gambling import Gambling
from data_collector import DataCollector
from trader import MomentumTrading, FixedWeightTrading, RiskParityTrading


class FundENV(Env):

    def __init__(self):
        # Variables
        original_value = 10000
        time_start = "01-01-2019"
        time_end = "31-12-2022"
        currencies = ["BTC", "ETH", "BNB", "XRP", "LTC", "DOGE", "USDT", "USDC", "ADA"]

        # Actions we can take: gambling hand, amount
        self.action_space = Tuple((Discrete[4], Box(low=0, high=1, shape=(1,), dtype=np.float32)))

        # Observations rewards from previous gambling, fund value before gambling, trader behavior
        list_1 = [Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                  Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)]

        # there are n traders
        list_2 = [Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32) for _ in range(10)]
        element_space = list_1+list_2
        self.observation_space = Tuple(element_space)

        # Set the value of SCT
        self.value = original_value
        # Set start state
        self.state = (0, original_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Set trading time length
        self.training_time = DataCollector.time_cal(time_start=time_start, time_end=time_end)
        # timer
        self.timer = 0

        # build DataFrame for the market
        df = DataCollector.dataframe_producing(currencies, time_start=time_start, time_end=time_end)
        self.df = DataCollector.add_SCT_to_df(df=df, data=original_value)

    def step(self, action):
        # action-gambling
        # apply action gambling
        self.winning = Gambling.playing_baccarat(action[0], action[1])
        # value calculation
        self.value += self.winning
        # update DataFrame of Market
        self.df = DataCollector.add_SCT_to_df(df=self.df, data=self.value)

        # trading
        # the information the trader can get
        df = self.df
        df_current = df.copy().loc[0:self.timer]
        # get weight

        # state observing

        # timer
        self.timer += 1

        # calculating reward

        # Check timer
        if self.training_time <= 0:
            Is_Done = True
        else:
            Is_Done = False

        # set placeholder
        info = {}

        # return step information
        # return self.value, self.state, self.training_time, self.reward, self.info
    def render(self):
        # implement with visualization
        pass

    def reset(self):
        original_value = 10000
        time_start = "01-01-2019"
        time_end = "31-12-2022"
        currencies = ["BTC", "ETH", "BNB", "XRP", "LTC", "DOGE", "USDT", "USDC", "ADA"]

        self.value = original_value
        self.state = (0, original_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.training_time = DataCollector.time_cal(time_start=time_start, time_end=time_end)
        df = DataCollector.dataframe_producing(currencies, time_start=time_start, time_end=time_end)
        self.df = DataCollector.add_SCT_to_df(df=df, data=original_value)

        return self.value, self.state, self.training_time, self.df