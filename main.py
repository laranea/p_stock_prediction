
from pandas import DataFrame
from pandas import read_csv, to_datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

def split_training_testing_set(df, x_df=None, y_df=None, size=20):

    train_test = lambda df: df[:-size], df[-size:]

    x_df = x_df or df['Open']
    y_df = y_df or df['Close']

    df_x_train, df_x_test  = train_test(x_df)
    df_y_train, df_y_test = train_test(y_df)
    return df_x_train, df_y_train, df_x_test, df_y_test


class DataCleaning:

    @staticmethod
    def rolling_mean(serie, window):
        return serie.rolling(window=window).mean()


    
class Regression:

    @staticmethod
    def linear(df):
        # lr = linear_model.LinearRegression()
        # open_X_train, close_Y_train, open_X_test, close_Y_test = split_training_testing_set(df)
        # lr.fit(open_X_train, close_Y_train)
        # close_Y_predict = lr.predict(open_X_test)
        # print('Coefficients: \n', lr.coef_)
        # # The mean squared error
        # print("Mean squared error: %.2f"
        #       % mean_squared_error(close_y_test, close_y_pred))

        # sns.heatmap(
        #     data=df.corr(),
        #     vmin=-1, vmax=1, center=0,
        #     cmap=sns.diverging_palette(20, 220, n=200),
        #     square=True
        # )
        sns.pairplot(df)
        # plt.scatter(df['Open'], df['Adj Close'])
        plt.show()
        
        # plt.scatter(open_X_test, open_Y_test, color='black')
        # plt.plot(open_X_test, open_Y_predict, color='blue', linewidth=3)
        # plt.show()

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
    compare_rolling_mean(df)

    
    #Regression.linear(df)

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

    
    
