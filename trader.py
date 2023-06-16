import pandas as pd
import numpy as np
from data_collector import DataCollector

class FixedWeightTrading:
    def __init__(self, df, df_returns, exclude):
        self.df = df
        self.df_returns = df_returns
        self.exclude = exclude

    def equal_weight(self, i):
        weights = pd.Series([0] * len(self.df.columns), index=self.df.columns)
        assets = self.df.columns[(self.df.columns != self.exclude)]
        weights[assets] = 1 / len(assets)
        return weights


class MomentumTrading:
    def __init__(self, df, df_returns, momentum_periods, escape_periods, exclude):
        self.df = df
        self.df_returns = df_returns
        self.momentum_periods = momentum_periods
        self.escape_periods = escape_periods
        self.exclude = exclude

    def dual_momentum(self, i):
        weights = pd.Series([0] * len(self.df.columns), index=self.df.columns)
        if (i >= self.momentum_periods):
            detect_momentum = (self.df.iloc[i - 1] - self.df.iloc[i - self.momentum_periods]) / self.df.iloc[
                i - self.momentum_periods]
            weights = detect_momentum.clip(lower=0)
            weights = np.where(
                ((weights == weights.max()) & ((weights.max() != 0) & (weights.idxmax() != self.exclude))), 1, 0)

        if (i >= self.escape_periods):
            detect_escape = (self.df.iloc[i - 1] - self.df.iloc[i - self.escape_periods]) / self.df.iloc[
                i - self.escape_periods]
            detect_escape = detect_escape.clip(lower=0)
            weights = weights * (np.sum(detect_escape * weights) != 0)

        weights = pd.Series(weights, index=self.df.columns)
        return weights


class RiskParityTrading:
    def __init__(self, df, df_returns, df_normalized, lookback, exclude=['USDT', 'USDC']):
        self.df = df
        self.df_returns = df_returns
        self.df_normalized = df_normalized
        self.lookback = lookback
        self.exclude = exclude

    def sd_risk(self, i):
        weights = pd.Series([0] * len(self.df.columns), index=self.df.columns)

        if (i >= self.lookback):
            std_devs = self.df_normalized.iloc[i - self.lookback:i].std()
            std_devs.drop(self.exclude, inplace=True)

            # Calculate the inverse of the standard deviation
            inv_std_devs = (1 / std_devs).replace([np.inf, -np.inf], 0)
            # Normalize the inverse standard deviations to get the portfolio weights
            weights.update(inv_std_devs / np.sum(inv_std_devs))

        return weights


class Return_rate_cal:
    def __init__(self, traders_number, df):
        self.traders_portfolio = [pd.DataFrame(np.nan, index=[0,1], columns=df.columns.append(
                pd.Index(['Returns', 'Current Asset', 'SCTInvest']))) for _ in range(traders_number)]
        for tp in self.traders_portfolio:
            tp.loc[1] = [0]*len(tp.columns)
            tp['Current Asset'][1] = 100

    def market_cal(self, t, df):

        df_returns = df.copy().pct_change()
        df_normalized = df / df.iloc[0]

        traders = [
            (FixedWeightTrading(df.copy(), df_returns.copy(), ''), 'equal_weight', ''),
            (MomentumTrading(df.copy(), df_returns.copy(), 20, 20, None), 'dual_momentum', '(sig=20, esc=20)'),
            (MomentumTrading(df.copy(), df_returns.copy(), 10, 20, None), 'dual_momentum', '(sig=10, esc=20)'),
            (MomentumTrading(df.copy(), df_returns.copy(), 10, 25, None), 'dual_momentum', '(sig=10, esc=25)'),
            (MomentumTrading(df.copy(), df_returns.copy(), 10, 30, None), 'dual_momentum', '(sig=10, esc=30)'),
            (RiskParityTrading(df.copy(), df_returns.copy(), df_normalized.copy(), 10), 'sd_risk', '(sig=10)'),
            (RiskParityTrading(df.copy(), df_returns.copy(), df_normalized.copy(), 15), 'sd_risk', '(sig=15)'),
            (RiskParityTrading(df.copy(), df_returns.copy(), df_normalized.copy(), 20), 'sd_risk', '(sig=20)'),
        ]

        traders_portfolio = self.traders_portfolio

        inv_coli = [df.columns.get_loc(col) for col in df.columns]
        rt_coli = traders_portfolio[0].columns.get_loc('Returns')

        for i, (q, method_name, _) in enumerate(traders):
            traders_portfolio[i].index = df.index[-2::]
            # Update Previous Weights and Asset and SCT invest
            traders_portfolio[i].iloc[0, inv_coli] = traders_portfolio[i].iloc[1, inv_coli]
            traders_portfolio[i].iloc[0, rt_coli + 1] = traders_portfolio[i].iloc[1, rt_coli + 1]
            traders_portfolio[i].iloc[0, rt_coli + 2] = traders_portfolio[i].iloc[1, rt_coli + 2]
            # Weights
            traders_portfolio[i].iloc[1, inv_coli] = getattr(q, method_name)(t)

            # Returns = recap previous performance (no SCP)
            df_returns_copy = df_returns.copy()
            df_returns_copy['SCT'][-1] = 0
            # print(df_returns_copy)
            traders_portfolio[i].iloc[1, rt_coli] = np.sum(
                traders_portfolio[i].iloc[0, inv_coli] * df_returns_copy.iloc[-1, inv_coli])
            # Current Asset to invest with weights = Previous Asset * Returns
            traders_portfolio[i].iloc[1, rt_coli + 1] = traders_portfolio[i].iloc[0, rt_coli + 1] * (
                    1 + traders_portfolio[i].iloc[1, rt_coli])
            # SCTInvest = how much i invest in SCT = Current Asset * SCT weight
            traders_portfolio[i].iloc[1, rt_coli + 2] = traders_portfolio[i].iloc[1, rt_coli + 1] * \
                                                        traders_portfolio[i].iloc[1, rt_coli - 1]

        self.traders_portfolio = traders_portfolio

        return self.traders_portfolio