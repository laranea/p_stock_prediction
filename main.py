
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
from itertools import product
from sklearn.metrics import mean_squared_error

# Sources:
# https://matplotlib.org/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
# https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
# https://becominghuman.ai/linear-regression-in-python-with-pandas-scikit-learn-72574a2ec1a5
# https://nbviewer.jupyter.org/github/rayryeng/make-money-ml-course/blob/master/week2/Week_2_Make_Money_with_Machine_Learning_Homework.ipynb


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
    indices = np.arange(nb_samples).astype(np.int)[:,None] + np.arange(size + 1).astype(np.int)
    data = y_df.values[indices]
    
    X = data[:, :-1]
    Y = data[:, -1]
    idx = int(fraction * nb_samples)

    training_testing = namedtuple('TT', 'x_train y_train x_test y_test size idx fraction')
    return training_testing(X[:idx], Y[:idx], X[idx:], Y[idx:], size, idx, fraction)


class DataCleaning:

    @staticmethod
    def rolling_mean(serie, window):
        return serie.rolling(window=window).mean()


class Regression:

    model_params = namedtuple('MP', 'function params')
    
    # Change params to test if prediction is better or not
    modelname_fun = {
        'LinearRegression': model_params(linear_model.LinearRegression,
                                         {'fit_intercept': [False, True],
                                          'normalize': [True, False]}),
                             
        'Ridge': model_params(linear_model.Ridge,
                              {'fit_intercept': [False, True],
                               'normalize': [True, False],
                               'alpha': np.arange(0.1, 2., 0.2)}),
        
        'Lasso': model_params(linear_model.Lasso,
                              {'fit_intercept': [True, False],
                               'normalize': [True, False],
                               'alpha': np.arange(0.1, 2., 0.2),
                               }),
        'LassoLars': model_params(linear_model.LassoLars,
                                  {'fit_intercept': [True, False],
                                   'normalize': [True, False],
                                   'alpha': np.arange(0.1, 2., 0.2),
                                  }),
        
        'BayesianRidge': model_params(linear_model.BayesianRidge,
                                      {'fit_intercept': [True, False],
                                       'normalize': [True, False],
                                       'n_iter': [300],
                                      })
    }

    @staticmethod
    def get_model(model):

        try:
            return Regression.modelname_fun[model]
        except KeyError:
            raise Exception(f'Unknown model {model}')

        
    @staticmethod
    def linear(df, x_df, y_df, model_name, size, fraction):
        """ Find the best linear regression model comparing configurations
        """
        sets = split_training_testing_set(x_df, y_df, size=size, fraction=fraction)

        results = []
        model_params = Regression.get_model(model_name)
        
        # For a model, get prediction for all configurations
        config_dicts = [ dict(zip(model_params.params, v)) for v in product(*model_params.params.values())]
        for config_dict in config_dicts:

            config = [f'{k}={v}' for k,v in config_dict.items()]
            name = f'{model_name} s={size} frac={fraction}' + " ".join(config)
            model = model_params.function(**config_dict)

            model.fit(sets.x_train, sets.y_train)
            y_pred = model.predict(sets.x_test)

            result_df = df.copy()
            result_df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            result_df.set_index('Date', inplace=True)
            result_df = result_df.iloc[sets.idx + sets.size:]
            result_df[name] = y_pred

            record = (name, np.sqrt(mean_squared_error(sets.y_test, y_pred)), result_df)
            results.append(record)

        return results


def compare_rolling_mean(df, col='Adj Close'):
    assert col in df.columns, f'missing {col} in df'
    cols = [col]
    for win_size in range(0, 50, 5):
        colname = f'window: {win_size}'
        df[colname] = DataCleaning.rolling_mean(df['Adj Close'],
                                                window=win_size)
        cols.append(colname)
        
    df[cols].plot(title='Impact of the rolling mean window argument on AdjClose Serie')
    plt.show()


if __name__ == '__main__':

    # Save RSME for each model + params
    model_score = {}

    # Reading source file
    df = load_and_format_csv()

    # Testing several configurations to understand impact of prediction
    sizes = [10, 15, 30, 45, 55] # windows size for prediction (in days)
    fractions = [0.7, 0.8] # sample fraction for training / testing
    dfs = []

    # Run all combinations and save rmse
    for size, fraction, modelname in product(sizes, fractions, Regression.modelname_fun.keys()):
        for (name, rmse, result_df) in Regression.linear(df, df['Date'], df['Adj Close'], modelname, size, fraction):
            model_score[name] = rmse
            if dfs:
                result_df.drop('Adj Close', axis=1, inplace=True)
            dfs.append(result_df)

    # Sort results according to RMSE (Best is the first)
    top10 = ['Adj Close'] # keep the Top 10
    for idx, (name, score) in enumerate(sorted(model_score.items(), key=itemgetter(1)), start=1):
        msg = f"[{idx}][model={name}] score={score}"
        if idx == 1:
            msg += " ** BEST"
        print(msg)
        if idx <= 10:
            top10.append(name)
    print("-"*80)

    # Display chart
    compare_df = concat(dfs, axis=1)
    top10_df = compare_df[top10]
    top10_df.plot(grid=True, title='Comparison of multiples linear models')
    plt.show()
