import pandas as pd
import numpy as np
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from tsfresh import extract_features
from tsfresh import extract_relevant_features


def convert_gas_to_ts_fresh_input(gas_df, target='margin'):
    _gas_df=gas_df.reset_index()
    new_df = {'features': [], 'day': [], 'c_id': [], 'values': [], 'labels': []}
    cols = _gas_df.columns.to_list()
    cols.remove(target)
    cols.remove('day')
    for i in range(len(_gas_df)):
        for i2 in cols:
            new_df['features'].append(i2)
            new_df['day'].append(_gas_df.iloc[i]['day'])
            new_df['values'].append(_gas_df.iloc[i][i2])
            new_df['labels'].append(_gas_df.iloc[i][target])
            new_df['c_id'].append(i)

    ts_fresh_input=pd.DataFrame(data=new_df)

    y_df = ts_fresh_input['labels']
    y_df.index = ts_fresh_input['c_id']
    y_df = y_df[~y_df.index.duplicated(keep='first')]

    x_df = ts_fresh_input.drop('labels', axis=1)

    return x_df,y_df


def tsfresh_feature_extraction(x_df,y_df,features=None):
    if features is None:
        features_df = extract_relevant_features(x_df, y_df, column_id='c_id', column_sort='day',column_value='values')
    else:
        features_df = extract_features(x_df, column_id="c_id", column_sort="day",column_value='values')[features]

    return features_df


def read_rlm_imputation():
    rlm_impu = read_data('/gasPrediciton/data/rlm_imputation.csv')
    rlm_impu.index = rlm_impu["ds"]
    rlm_impu.index.name = 'day'
    rlm_impu.drop(["ds"], inplace=True, axis=1)
    rlm_impu.drop(["Unnamed: 0"], inplace=True, axis=1)
    rlm_impu.index = pd.to_datetime(rlm_impu.index)
    return rlm_impu


def read_gas_submission():
    df_sub = read_data('/gasPrediciton/data/submission_example.csv', sep=';')
    df_sub.index = df_sub["day"]
    df_sub.drop(["day"], inplace=True, axis=1)
    df_sub.index = pd.to_datetime(df_sub.index)
    return df_sub


def create_gas_without_na():
    df_gas = read_gasprediction_gas_data()
    df_weather = read_gasprediction_weather_data(imputation=True)
    df_gas_mod = df_gas.dropna()
    df_weather_mod = df_weather.dropna()

    df_gas_and_weather = pd.merge(df_gas_mod, df_weather_mod, left_index=True, right_index=True, how="inner")

    timestamp_s = df_gas_and_weather.index.map(pd.Timestamp.timestamp)
    weeklySig = createWeeklyTimeSignal(timestamp_s)
    weeklySig.index = df_gas_and_weather.index

    yearlySig = createYearlyTimeSignal(timestamp_s)
    yearlySig.index = df_gas_and_weather.index

    df_gas_and_weather_with_signal = pd.merge(df_gas_and_weather, weeklySig, left_index=True, right_index=True,
                                              how="inner")
    return pd.merge(df_gas_and_weather_with_signal, yearlySig, left_index=True, right_index=True, how="inner")


def create_gas_with_na():
    df_gas = read_gasprediction_gas_data()
    df_weather = read_gasprediction_weather_data(imputation=True)

    df_gas_and_weather = pd.merge(df_gas, df_weather, left_index=True, right_index=True, how="outer")

    timestamp_s = df_gas_and_weather.index.map(pd.Timestamp.timestamp)
    weeklySig = createWeeklyTimeSignal(timestamp_s)
    weeklySig.index = df_gas_and_weather.index

    yearlySig = createYearlyTimeSignal(timestamp_s)
    yearlySig.index = df_gas_and_weather.index

    df_gas_and_weather_with_signal = pd.merge(df_gas_and_weather, weeklySig, left_index=True, right_index=True,
                                              how="outer")
    return pd.merge(df_gas_and_weather_with_signal, yearlySig, left_index=True, right_index=True, how="outer")


def read_gasprediction_gas_data():
    df_train = read_data('/gasPrediciton/data/train.csv')
    df_train.index = df_train["day"]
    df_train.drop(columns=["day"], inplace=True)
    df_train.index = pd.to_datetime(df_train.index)
    return df_train


def read_gasprediction_weather_data(imputation=False):
    weather = read_data('/gasPrediciton/data/de-weather-data-aggregated.csv', sep=',')
    weather.index = weather["day"]
    weather.drop(["day"], inplace=True, axis=1)
    weather.index = pd.to_datetime(weather.index)

    if imputation is True:
        weather = impute_missing_weather_data(weather)

    return weather

def read_prophet_forecasts():
    weather = read_data('/gasPrediciton/data/prophet_forcast.csv')
    weather.index = weather["ds"]
    weather.drop(["ds"], inplace=True, axis=1)
    weather.index = pd.to_datetime(weather.index)

    return weather

def read_xgb_csv():
    weather = read_data('/gasPrediciton/data/submission_xgb.csv')
    weather.index = weather["day"]
    weather.drop(["day"], inplace=True, axis=1)
    weather.index = pd.to_datetime(weather.index)

    return weather


def read_data(filePath, basePath='../../30data', sep=';', decimal='.'):
    pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), basePath + filePath)
    df = pd.read_csv(pth, sep=sep, decimal=decimal)
    return df


def impute_missing_weather_data(weather_df):
    weather_df['wdir_std'] = weather_df['wdir_std'].interpolate(option='time')
    weather_df['wdir_mean'] = weather_df['wdir_mean'].interpolate(option='time')
    return weather_df


# TODO: RENAME THIS BECAUSE OF MISSING INITIAL COLUMNS
def build_lagged_features(s, bonus_cols, lag=5):
    new_dict = {}
    for col_name in bonus_cols:
        # ONLY necessary if we want to include the features of all subsequent days, not only the last one
        # new_dict[col_name]=s[col_name]
        # create lagged Series
        # for l in range(1,lag+1):
        #    new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(-l)
        l = lag
        new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(-l)
    res = pd.DataFrame(new_dict, index=s.index)

    # TODO: remove nans or convert to zero
    return res


def build_lagged_features_xgb(s, lag=10):
    new_dict = {}
    for col_name in s.columns:
        # ONLY necessary if we want to include the features of all subsequent days, not only the last one
        new_dict[col_name] = s[col_name]
        # create lagged Series
        for l in range(1, lag + 1):
            new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(l)
        # l = lag
        # new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(-l)
    res = pd.DataFrame(new_dict, index=s.index)

    # TODO: remove nans or convert to zero
    return res


def createWeeklyTimeSignal(timestamp_s):
    day = 24 * 60 * 60
    week = 7 * day

    return pd.DataFrame(
        data={'Dow sin': np.sin(timestamp_s * (2 * np.pi / week)), 'Dow cos': np.cos(timestamp_s * (2 * np.pi / week))})


def createYearlyTimeSignal(timestamp_s):
    day = 24 * 60 * 60
    year = (365.2425) * day

    return pd.DataFrame(
        data={'Year sin': np.sin(timestamp_s * (2 * np.pi / year)),
              'Year cos': np.cos(timestamp_s * (2 * np.pi / year))})


def train_val_test_split(df, train_size=0.5, val_only=False):
    n = len(df)
    train_df = df[0:int(n * train_size)]

    if val_only is False:
        val_size = int(n * (train_size + ((1 - train_size) / 2)))
        val_df = df[int(n * train_size):val_size]
        test_df = df[val_size:]
    else:
        val_df = df[int(n * train_size):]
        test_df = None

    return train_df, val_df, test_df


def train_val_test_split_experiemental(df, train_size=0.5):
    n = len(df)

    train_start_index = int(n - n * train_size)
    train_df = df[train_start_index:]

    val_end_index = int(n * ((1 - train_size) / 2))
    val_df = df[:val_end_index]

    test_df = df[val_end_index:train_start_index]

    return train_df, val_df, test_df


class Standardiser:
    def __init__(self, train_df, min_max=False) -> None:
        if min_max == False:
            self.substract = train_df.mean()
            self.divide = train_df.std()
        else:
            self.substract = train_df.min()
            self.divide = train_df.max() - train_df.min()

    def __call__(self, train_df, val_df, test_df, *args, **kwds):
        train_df = (train_df - self.substract) / self.divide
        val_df = (val_df - self.substract) / self.divide
        test_df = (test_df - self.substract) / self.divide
        return train_df, val_df, test_df


class Standardiser2:
    def __init__(self, train_df, min_max=False) -> None:
        if min_max == False:
            self.substract = train_df.mean()
            self.divide = train_df.std()
        else:
            self.substract = train_df.min()
            self.divide = train_df.max() - train_df.min()

    def __call__(self, *args, **kwds):
        return [(x - self.substract) / self.divide for x in args]


def standardisierung(train_df, val_df=None, test_df=None, norm_params=None, exclude=None):
    if norm_params is not None:
        train_mean = norm_params[0]
        train_std = norm_params[1]
    else:
        train_mean = train_df.mean()
        train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    if val_df is not None:
        val_df = (val_df - train_mean) / train_std

    if test_df is not None:
        test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df, [train_mean, train_std]


def denorm_stand(df, norm_params):
    return df * norm_params[1] + norm_params[0]


def standardisierung_min_max(train_df, val_df=None, test_df=None, norm_params=None, exclude=None):
    if exclude is None:
        exclude = []
    if norm_params is not None:
        maxs = norm_params[0]
        mins = norm_params[1]
    else:
        mins = train_df.min()
        maxs = train_df.max()

    excluded = {}
    for i in exclude:
        excluded[i] = train_df[i].copy()

    train_df = (train_df - mins) / (maxs - mins)

    for i in excluded.keys():
        train_df[i] = excluded[i]

    if val_df is not None:
        excluded = {}
        for i in exclude:
            excluded[i] = val_df[i].copy()

        val_df = (val_df - mins) / (maxs - mins)

        for i in excluded.keys():
            val_df[i] = excluded[i]

    if test_df is not None:
        excluded = {}
        for i in exclude:
            excluded[i] = test_df[i].copy()

        test_df = (test_df - mins) / (maxs - mins)

        for i in excluded.keys():
            test_df[i] = excluded[i]

    return train_df, val_df, test_df, [maxs, mins]


def denorm_min_max(df, norm_params):
    return df * (norm_params[0] - norm_params[1]) + norm_params[1]


"""
Braucht einen (denormalisierten) pandas dataframe folgender form:
    *      | targets | predictions
date_index |  ...    |   ...
"""


def interactive_timeseries_plot(df_preds, time_res_in_days=365, width=1400, height=800, y_axis=None):
    if y_axis is None:
        y_axis = ['predictions', 'targets']
    import plotly.express as px

    df_preds.index.name = 'day'
    df_plot = df_preds.reset_index(level=0)
    df_plot['day_s'] = df_plot.day
    df_plot = df_plot.astype({'day_s': 'string'})

    fig = px.line(df_plot, x="day_s", y=y_axis,
                  hover_data={"day_s": "|%B %d, %Y"},
                  title='custom tick labels', range_x=[df_plot.iloc[0].day, df_plot.iloc[time_res_in_days].day],
                  width=width, height=height)
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.show()


def augmented_dickey_fuller_statistics(time_series, feat_name):
    data = {'Feature': [], 'ADF Statistic': [], 'p-value': []}

    data['Feature'].append(feat_name)

    result = adfuller(time_series.values, autolag='AIC')

    data['ADF Statistic'].append(result[0])
    data['p-value'].append(result[1])
    for key, value in result[4].items():
        data[key] = []
        data[key].append(value)

    if result[1] <= 0.05:
        data['is_stationär'] = 'True'
    else:
        data['is_stationär'] = 'False'

    return pd.DataFrame(data=data)


def grangers_causality_matrix(X_train, variables, test='ssr_chi2test', max_lag=12, verbose=False):
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(X_train[[r, c]], maxlag=max_lag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(max_lag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            dataset.loc[r, c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]
    return dataset


def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 24)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)
