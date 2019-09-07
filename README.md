# Stock Prediction

This code compare several linear models from sklearn librairy in order to predict stock.
/data: APPL.csv.

The training set is based on a window of N days, the target value is the following day.

# Description

This code try several parameters changes to check impact on prediction. All combinations are checked
for all models and only top10 (RMSE score) are displayed. 


# Results

## Chart comparing the 10 best predictions.

  ![Alt text](top10_chart.png?raw=true "Comparison of stock prediction with linear models")

## Root Mean Squarred Error results (sorted: best first)

 - [1][model=LinearRegression s=10 frac=0.7fit_intercept=False normalize=True] score=3.329237228064044 ** BEST
 - [2][model=LinearRegression s=10 frac=0.7fit_intercept=False normalize=False] score=3.329237228064044
 - [3][model=Ridge s=10 frac=0.7fit_intercept=False alpha=1.0] score=3.3293174192861867
 - [4][model=Ridge s=10 frac=0.7fit_intercept=False alpha=1.1] score=3.3293254577040705
 - [5][model=BayesianRidge s=10 frac=0.7fit_intercept=False n_iter=300] score=3.3468441167291743
 - [6][model=LassoLars s=10 frac=0.7fit_intercept=False alpha=1.1] score=3.361863077205247
 - [7][model=LassoLars s=10 frac=0.7fit_intercept=False alpha=1.0] score=3.3628894021644435
 - [8][model=LassoLars s=15 frac=0.7fit_intercept=False alpha=1.1] score=3.399937460735615
 - [9][model=LassoLars s=15 frac=0.7fit_intercept=False alpha=1.0] score=3.4029249158774477
 - [10][model=LinearRegression s=15 frac=0.7fit_intercept=False normalize=True] score=3.4402616157909653
 - [11][model=LinearRegression s=15 frac=0.7fit_intercept=False normalize=False] score=3.4402616157909653
 - [12][model=Ridge s=15 frac=0.7fit_intercept=False alpha=1.0] score=3.4404017547558765
 - [13][model=Ridge s=15 frac=0.7fit_intercept=False alpha=1.1] score=3.4404157852087014
 - [14][model=LassoLars s=20 frac=0.7fit_intercept=False alpha=1.1] score=3.4651546288159705
 - [15][model=LassoLars s=20 frac=0.7fit_intercept=False alpha=1.0] score=3.4691610609351287
 - [16][model=BayesianRidge s=15 frac=0.7fit_intercept=False n_iter=300] score=3.4873511446158356
 - [17][model=LinearRegression s=20 frac=0.7fit_intercept=False normalize=True] score=3.5237182235374593
 - [18][model=LinearRegression s=20 frac=0.7fit_intercept=False normalize=False] score=3.5237182235374593
 - [19][model=Ridge s=20 frac=0.7fit_intercept=False alpha=1.0] score=3.5239041023214632
 - [20][model=Ridge s=20 frac=0.7fit_intercept=False alpha=1.1] score=3.523922701214573
 - [21][model=BayesianRidge s=20 frac=0.7fit_intercept=False n_iter=300] score=3.6032725885753205
 - [22][model=Lasso s=10 frac=0.7fit_intercept=False alpha=1.1] score=4.349009249332183
 - [23][model=Lasso s=10 frac=0.7fit_intercept=False alpha=1.0] score=4.349648099553047
 - [24][model=Lasso s=15 frac=0.7fit_intercept=False alpha=1.1] score=5.907425349359115
 - [25][model=Lasso s=15 frac=0.7fit_intercept=False alpha=1.0] score=5.912282567725009
 - [26][model=Lasso s=20 frac=0.7fit_intercept=False alpha=1.1] score=6.222952923470525
 - [27][model=Lasso s=20 frac=0.7fit_intercept=False alpha=1.0] score=6.229652613383979
































































