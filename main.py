
from pandas import DataFrame
from pandas import read_csv, to_datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

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
    #df.set_index('Date', inplace=True)
    print(df.columns)
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

    
    # train_set = lambda df: df[:idx]
    # test_set = lambda df: df[idx:]

    # print(x_df[:idx])
    
    # x_train, y_train = map(train_set, (x_df, y_df))
    # x_test, y_test = map(test_set, (x_df, y_df))
    # 
    # return training_testing(x_train, y_train, x_test, y_test)


class DataCleaning:

    @staticmethod
    def rolling_mean(serie, window):
        return serie.rolling(window=window).mean()



class Regression:

    @staticmethod
    def get_model(model):

        name_fun = {
            'LinearRegression': linear_model.LinearRegression,
            'Ridge': linear_model.Ridge,
            'Lasso': linear_model.Lasso,
        }
        try:
            return name_fun[model]
        except KeyError:
            raise Exception(f'Unknown model {model}')

    
    @staticmethod
    def linear(df, x_df, y_df, model_name='LinearRegression'):
        sets = split_training_testing_set(x_df, y_df)

        model = Regression.get_model(model_name)()
        model.fit(sets.x_train, sets.y_train)

        y_pred_train = model.predict(sets.x_train)
        y_pred = model.predict(sets.x_test)

        df_ridge = df.copy()
        df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        df_ridge.set_index('Date', inplace=True)
        df_ridge = df_ridge.iloc[sets.size:sets.idx] # Past 32 days we don't know yet
        df_ridge['Adj Close Train'] = y_pred_train[:-sets.size]
        df_ridge.plot(label='TSLA', figsize=(16,8), title=f'Adjusted Closing Price {model_name}', grid=True)
        plt.show()


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

from numpy import geomspace
if __name__ == '__main__':
    df = load_and_format_csv()
    #compare_rolling_mean(df)

    for model in ('LinearRegression', 'Ridge', 'Lasso'):
        Regression.linear(df, df['Date'], df['Adj Close'], model)
        

    # from sklearn import datasets
    # diabetes = datasets.load_diabetes()
    # # Use only one feature
    # diabetes_X = diabetes.data[:, np.newaxis, 2]
    
    # # Split the data into training/testing sets
    # diabetes_X_train = diabetes_X[:-20]
    # diabetes_X_test = diabetes_X[-20:]    

    # print(diabetes_X_train)
    

    # df[['Adj Close']].plot()
    # plt.show()

    
    
