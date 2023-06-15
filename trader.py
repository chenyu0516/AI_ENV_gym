import pandas as pd
import numpy as np


class FixedWeightTrading:
    def __init__(self, df, df_returns, exclude):
        self.df = df
        self.df_returns = df_returns
        self.exclude = exclude

    def equal_weight(self, i):
        weights = pd.Series([0] * len(self.df.columns), index=self.df.columns)
        assets = self.df.columns[(self.df.columns != self.exclude) & self.df_returns.iloc[i + 1].notnull()]
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

        def update_return_rate(self, df, t):

            df_returns = df.ptc_change()
            df_normalized = df / df.iloc[0]

            traders = [
                (FixedWeightTrading(df.copy(), df_returns.copy(), ''), 'equal_weight', ''),
                (MomentumTrading(df.copy(), df_returns.copy(), 10, 20, None), 'dual_momentum', '(sig=10, esc=20)'),
                (RiskParityTrading(df.copy(), df_returns.copy(), df_normalized.copy(), 10), 'sd_risk', '(sig=10)'),
            ]
            traders_portfolio = [pd.DataFrame(np.nan, index=df.index, columns=df.columns.append(pd.Index(['Returns'])))
                                 for _ in range(len(traders))]
            # scam = [0] * df.shape[0]

            for i, (q, method_name, _) in enumerate(traders):
                traders_portfolio[i].iloc[t, :-1] = getattr(q, method_name)(t)
                traders_portfolio[i]['Returns'][t + 1] = np.sum(
                    traders_portfolio[i].iloc[t, :-1] * df_returns.iloc[t + 1, :-1])

