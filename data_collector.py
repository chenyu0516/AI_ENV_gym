import json
from cryptocmd import CmcScraper
from datetime import date
import pandas as pd
import numpy as np


class DataCollector(json, CmcScraper, date, pd, np):
    def data_collection(self, currencies, time_start, time_end):

        data_collection = {}  # Dictionary to store the collected data

        for currency in currencies:
            scraper = CmcScraper(currency, time_start, time_end)
            data_tuple = scraper.get_data()

            peroid = self.time_cal(time_start, time_end)
            data_list = [0] * peroid
            for i in range(peroid):
                data_list[i] = data_tuple[1][-i - 1][4]

            # Store the data in the collection dictionary
            data_collection[currency] = {
                "currency": currency,
                "start_date": time_start,
                "end_date": time_end,
                "data": data_list
            }

            return data_collection

    def time_cal(self, time_start, time_end):

        time_start = time_start.split('-')
        time_end = time_end.split('-')

        start_date = date(int(time_start[2]), int(time_start[1]), int(time_start[0]))
        end_date = date(int(time_end[2]), int(time_end[1]), int(time_end[0]))

        delta = end_date - start_date
        return delta.days

    def time_transform(self, input):

        input = input.split('-')
        return input[2]+input[1]+input[0]

    def data_transform(self, input_data):
        output_data = [1.0] * len(input_data)
        for i in range(len(input_data)):
            if i != 0:
                fraction = input_data[i] / input_data[i - 1]
                output_data[i] = output_data[i - 1] * fraction
        return output_data

    def dataframe_producing(self, currencies, time_start, time_end):

        start = self.time_transform(time_start)
        time_peroid = self.time_cal(time_start,time_end)
        data = self.data_collection(currencies, time_start=time_start, time_end=time_end)

        dates = pd.date_range(start, periods=time_peroid)
        data_array = np.empty((len(dates), len(currencies)))

        for i, coin in enumerate(currencies):
            data = data[coin]['data']
            data_trans = self.data_transform(data)
            data_array[:, i] = data_trans

        return pd.DataFrame(data_array, index=dates, columns=currencies)

    def add_SCT_to_df(self, df, data):
        return df.iloc[data, 'SCT']
