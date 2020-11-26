import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from collections import deque
from polygon import RESTClient
from datetime import date
import pickle
import config


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def is_leap_year(year):
    return int(year) % 4 == 0


def get_polygon_tick_data(ticker, start_year, end_year):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    all_data = pd.DataFrame()
    current_year = start_year
    with RESTClient(config.alpaca_key) as client:
        while current_year <= end_year:
            for month, days in enumerate(days_in_month, start=1):
                # Check if we are past the current date
                current_date = date.today()
                if current_year >= current_date.year and month > current_date.month:
                    break
                if month == 2 and is_leap_year(current_year):
                    days = 29
                print(f'Downloading data from polygon for {current_year}-{month:02}')
                resp = client.stocks_equities_aggregates(ticker=ticker, timespan='minute', multiplier='1',
                                                         from_=f'{current_year}-{month:02}-01',
                                                         to=f'{current_year}-{month:02}-{days}',
                                                         limit='50000')
                data = pd.DataFrame(resp.results)
                data.set_index('t')
                all_data.dropna(inplace=True)
                data.fillna(method="ffill", inplace=True)
                all_data = all_data.append(data)
            current_year += 1
    return all_data


def classify(current, future):
    target = 1 if future > current else 0
    return target


# Download tick data and reformat to a flatter dataframe
def create_data(ticker, start_year, end_year, future_prediction, time_frame, validation_split):
    all_data = get_polygon_tick_data(ticker=ticker, start_year=start_year, end_year=end_year)

    # Create target (sell or buy) based on future price
    future = all_data['c'].shift(-future_prediction)
    all_data[f'target'] = list(map(classify, all_data['c'], future))

    all_data.dropna(inplace=True)

    # Once all the data has been collated we don't care about the time stamp
    all_data.drop('t', axis=1, inplace=True)
    all_data.drop('n', axis=1, inplace=True)

    validation_cutoff_index = -int(validation_split * len(all_data))

    train_data = all_data[:validation_cutoff_index]
    validation_data = all_data[validation_cutoff_index:]

    train_x, train_y = preprocess_data(train_data, time_frame=time_frame)
    validation_x, validation_y = preprocess_data(validation_data, time_frame=time_frame)

    return train_x, train_y, validation_x, validation_y


def preprocess_data(data, time_frame):
    formatted_data = pd.DataFrame()

    for col in data.columns:
        if not 'target' == col:
            formatted_data[col] = data[col].pct_change()
            formatted_data.replace(np.inf, np.nan, inplace=True)
            formatted_data[col] = preprocessing.scale(formatted_data[col].values)
        else:
            formatted_data[col] = data[col]

    print(formatted_data, file=open('lol.txt', 'a'))

    # Drop any NaNs that might have crept through
    formatted_data.dropna(inplace=True)

    # Create segmented sections of data
    sequential_data = []
    previous = deque(maxlen=time_frame)

    # Include all but last (target) row in data and separate it out
    for i in formatted_data.values:
        previous.append([n for n in i[:-1]])
        if len(previous) == time_frame:
            sequential_data.append([np.array(previous), i[-1]])

    # Balance data to have equal number of buys and sells
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        else:
            buys.append([seq, target])

    smaller_set_index = min(len(buys), len(sells))
    buys = buys[:smaller_set_index]
    sells = sells[:smaller_set_index]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), np.array(y)


def save_data(train_x, train_y, validation_x, validation_y):
    pickle.dump(train_x, open('train_x.pickle', 'wb'))
    pickle.dump(train_y, open('train_y.pickle', 'wb'))
    pickle.dump(validation_x, open('validation_x.pickle', 'wb'))
    pickle.dump(validation_y, open('validation_y.pickle', 'wb'))

