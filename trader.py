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
    def market_cal(self, t, df, start, end):

        df_returns = df.pct_change()
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

        traders_portfolio = [pd.DataFrame(np.nan, index=df.index, columns=df.columns.append(
            pd.Index(['SCTInvest', 'Returns', 'Current Asset']))) for _ in range(len(traders))]
        for trader_portfolio in traders_portfolio:
            trader_portfolio['Current Asset'][0] = 100

        inv_coli = [df.columns.get_loc(col) for col in df.columns]
        rt_coli = traders_portfolio[0].columns.get_loc('Returns')

        for i, (q, method_name, _) in enumerate(traders):
            traders_portfolio[i].iloc[t, inv_coli] = getattr(q, method_name)(t)
            traders_portfolio[i].iloc[t, rt_coli - 1] = traders_portfolio[i].iloc[t, rt_coli + 1] * \
                                                        traders_portfolio[i].iloc[t, rt_coli - 2]
            data_collector = DataCollector()
            traders_portfolio[i].iloc[t, rt_coli] = np.sum(
                    traders_portfolio[i].iloc[t-1, inv_coli] * df_returns.copy().iloc[t, inv_coli])
            traders_portfolio[i].iloc[t, rt_coli + 1] = traders_portfolio[i].iloc[t-1, rt_coli + 1] * (
                    1 + traders_portfolio[i].iloc[t, rt_coli])

        return traders_portfolio