import pendulum
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def change_date_to_datetime(data):
    if type(data) == dict:
        for name, df in data.items():
            df = df.copy()
            date_name = df.iloc[:,0].name # time column name
            df['date'] = df[date_name].apply(lambda x: pendulum.from_format(str(x),'YYYYMMDD'))
            #df['date'] = pd.to_datetime(df[date_name], format='%Y%m%d') # same code
            #df['date'] = df[date_name].apply(lambda x:datetime.strptime(str(x), '%Y%m%d')) # same code
            df = pd.concat([df['date'], df.iloc[:, -20:-1]], axis=1) # time index + 19 features
            data[name] = df

    elif type(data) == pd.DataFrame:
        date_name = data.iloc[:,0].name
        data[date_name] = data[date_name].apply(lambda x: pendulum.from_format(str(x),'YYYYMMDD'))

    return data

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def make_time_rolling_data():
    '''
    This function has a closure area
    When roll the data window, the number of rolling is updated in this closure area
    '''
    roll_num = 0
    def rolling(data, train_month, val_month, test_month):
        nonlocal roll_num
        initial_date = data.date.iloc[0]

        start_roll_date = initial_date + relativedelta(months=test_month * roll_num) # first day in rolled month
        last_roll_date = initial_date + relativedelta(months=train_month + val_month + test_month - 1) # to find end day in 1 lag month calculated i.e. 09.01 x / 08.31 o

        years = data.date.apply(lambda x: x.year)
        months = data.date.apply(lambda x: x.month)

        start_roll_work_date = data.date[start_roll_date.year == years][start_roll_date.month == months].iloc[0] # first work day in rolled month
        last_roll_work_date = data.date[last_roll_date.year == years][last_roll_date.month == months].iloc[-1] # last work day

        start_index, last_index = data[data.isin([start_roll_work_date, last_roll_work_date])].index.values

        outdata = data[start_index:last_index+1]

        roll_num += 1

        return outdata


