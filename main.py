
from pandas import DataFrame
from pandas import read_csv, to_datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from pandas import concat
from operator import itemgetter

# https://matplotlib.org/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
# https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
#https://becominghuman.ai/linear-regression-in-python-with-pandas-scikit-learn-72574a2ec1a5



def load_and_format_csv(filename='./data/AAPL.csv'):
    """ Load csv file to dataframe and format Dates as datetime64
    """
    assert os.path.isfile(filename), f'{filename} not found'
    df = read_csv(filename)     # loading csv
    if "Date" in df.columns:
        df['Date'] = to_datetime(df['Date'], format='%Y-%m-%d')
    return df

def split_training_testing_set(x_df=None, y_df=None, size=50, fraction=0.8):
    """
    First, x_df is divided into x-windows of size=size.
    """

    assert len(x_df) == len(y_df), f'x_df must have size of y_df, now {len(x_df)} != {len(y_df)}'

    nb_samples = len(x_df) - size
    idx = int(fraction * nb_samples)

    indices = np.arange(nb_samples).astype(np.int)[:,None] + np.arange(size + 1).astype(np.int)
    data = y_df.values[indices]

    X = data[:, :-1]
    Y = data[:, -1]

    training_testing = namedtuple('TT', 'x_train y_train x_test y_test size idx fraction')
    return training_testing(X[:idx], Y[:idx], X[idx:], Y[idx:], size, idx, fraction)


class DataCleaning:

    @staticmethod
    def rolling_mean(serie, window):
        return serie.rolling(window=window).mean()

from sklearn.metrics import mean_squared_error
class Regression:

    modelname_fun = {
        'LinearRegression': linear_model.LinearRegression,
        'Ridge': linear_model.Ridge,
        'Lasso': linear_model.Lasso,
        'LassoLars': linear_model.LassoLars,
        'BayesianRidge': linear_model.BayesianRidge,
    }

    @staticmethod
    def get_model(model):

        try:
            return Regression.modelname_fun[model]
        except KeyError:
            raise Exception(f'Unknown model {model}')

    
    @staticmethod
    def linear(df, x_df, y_df, model_name):
        sets = split_training_testing_set(x_df, y_df)

        model = Regression.get_model(model_name)()
        model.fit(sets.x_train, sets.y_train)
        
        y_pred_train = model.predict(sets.x_train)
        y_pred = model.predict(sets.x_test)

        result_df = df.copy()
        result_df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        result_df.set_index('Date', inplace=True)
        result_df = result_df.iloc[sets.size:sets.idx]
        result_df[model_name] = y_pred_train[:-sets.size]
        return mean_squared_error(sets.y_test, y_pred), result_df


def compare_rolling_mean(df, col='Adj Close'):
    assert col in df.columns, f'missing {col} in df'
    cols = [col]
    for win_size in range(0, 100, 10):
        colname = f'window: {win_size}'
        df[colname] = DataCleaning.rolling_mean(df['Adj Close'],
                                                window=win_size)
        cols.append(colname)
        
    df[cols].plot(title='Impact of the rolling mean window argument on AdjClose Serie')
    plt.show()


if __name__ == '__main__':


    model_score = {}
    dfs = []
    
    df = load_and_format_csv()

    for modelname in Regression.modelname_fun.keys():
        error, result_df = Regression.linear(df, df['Date'], df['Adj Close'], modelname)
        model_score[modelname] = error
        if dfs:
            result_df.drop('Adj Close', axis=1, inplace=True)
        dfs.append(result_df)
    
    for idx, (model, score) in enumerate(sorted(model_score.items(), key=itemgetter(1)), start=1):
        msg = f"[{idx}][model={model}] score={score}"
        if idx == 1:
            msg += " ** BEST"
        print(msg)
    print("-"*80)

    compare_df = concat(dfs, axis=1)
    compare_df.plot(grid=True, title='Comparison of multiples Linear Regression Models')
    plt.show()
