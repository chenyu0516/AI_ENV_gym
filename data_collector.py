import datetime

from cryptocmd import CmcScraper
from datetime import date
import pandas as pd
import numpy as np
import json


class DataCollector:
    def data_collection(self, currencies, time_start, time_end):

        data_collection = {}  # Dictionary to store the collected data
        print('Start market data collecting...')
        for currency in currencies:
            print(f'Start collecting {currency}...')
            scraper = CmcScraper(currency, time_start, time_end)
            data_tuple = scraper.get_data()

            peroid = self.time_cal(time_start, time_end)
            data_list = [0] * peroid
            for i in range(peroid):
                data_list[i] = data_tuple[1][-i - 1][4]

            print(f'Collecting {currency} success')
            print('Storing...')
            # Store the data in the collection dictionary
            data_collection[currency] = {
                "currency": currency,
                "start_date": time_start,
                "end_date": time_end,
                "data": data_list
            }
            print(f'Storing {currency} success')

        with open("crypto_data.json", "w") as json_file:
            json.dump(data_collection, json_file)

    def time_cal(self, time_start, time_end):

        time_start = time_start.split('-')
        time_end = time_end.split('-')

        start_date = date(int(time_start[2]), int(time_start[1]), int(time_start[0]))
        end_date = date(int(time_end[2]), int(time_end[1]), int(time_end[0]))

        delta = end_date - start_date
        return delta.days + 1

    def time_transform(self, input):

        input = input.split('-')
        return input[2]+input[1]+input[0]

    def time_transform_1(self, input):
        input = input.split('-')
        return input[2]+'-'+input[1]+'-'+input[0]


    def data_read(self):
        with open('crypto_data.json', 'r') as f:
            data_json = f.read()
        return json.loads(data_json)

    def dataframe_producing(self, currencies, time_start, time_end):
        start = self.time_transform(time_start)
        time_period = self.time_cal(time_start, time_end)
        data_dict = self.data_read()
        print(time_period)

        dates = pd.date_range(start, periods=time_period)
        data_array = np.empty((len(dates), len(currencies)))

        for i, coin in enumerate(currencies):
            data_array[:, i] = data_dict[coin]['data']  # Access the data list for the current currency

        return pd.DataFrame(data_array, index=dates, columns=currencies)

    def add_SCT_to_df(self, df, data, time_start, date):
        time = time_start.split('-')
        start = datetime.date(int(time[2]), int(time[1]), int(time[0]))
        the_date = start + datetime.timedelta(days=date)
        the_date = str(the_date) + ' 00:00:00'
        df.at[the_date, "SCT"] = data
        return df

