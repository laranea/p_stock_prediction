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

##Root Mean Squarred Error results (sorted: best first)

Samples:

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
- [14][model=LassoLars s=55 frac=0.7fit_intercept=False alpha=1.1] score=3.4566733970927497
- [15][model=LassoLars s=55 frac=0.7fit_intercept=False alpha=1.0] score=3.4738485938536816
- [16][model=BayesianRidge s=10 frac=0.8fit_intercept=False n_iter=300] score=3.4865633099831275
- [17][model=BayesianRidge s=15 frac=0.7fit_intercept=False n_iter=300] score=3.4873511446158356
- [18][model=Ridge s=10 frac=0.8fit_intercept=False alpha=1.1] score=3.489907851164658
- [19][model=Ridge s=10 frac=0.8fit_intercept=False alpha=1.0] score=3.489912997254991
- [20][model=LinearRegression s=10 frac=0.8fit_intercept=False normalize=True] score=3.489964738843719
- [21][model=LinearRegression s=10 frac=0.8fit_intercept=False normalize=False] score=3.489964738843719
- [22][model=LassoLars s=30 frac=0.7fit_intercept=False alpha=1.0] score=3.5291854352735186
- [23][model=LassoLars s=30 frac=0.7fit_intercept=False alpha=1.1] score=3.5321217727932126
- [24][model=BayesianRidge s=15 frac=0.8fit_intercept=False n_iter=300] score=3.539748258433491
- [25][model=Ridge s=15 frac=0.8fit_intercept=False alpha=1.1] score=3.5410391205878593

Samples:

- [1][model=LinearRegression s=10 frac=0.7fit_intercept=False normalize=True] score=3.329237228064044 ** BEST
- [2][model=LinearRegression s=10 frac=0.7fit_intercept=False normalize=False] score=3.329237228064044
- [3][model=Ridge s=10 frac=0.7fit_intercept=False normalize=True alpha=0.1] score=3.3292452313618024
- [4][model=Ridge s=10 frac=0.7fit_intercept=False normalize=False alpha=0.1] score=3.3292452313618024
- [5][model=Ridge s=10 frac=0.7fit_intercept=False normalize=True alpha=0.30000000000000004] score=3.3292612485236264
- [6][model=Ridge s=10 frac=0.7fit_intercept=False normalize=False alpha=0.30000000000000004] score=3.3292612485236264
- [7][model=Ridge s=10 frac=0.7fit_intercept=False normalize=True alpha=0.5000000000000001] score=3.3292772797589114
- [8][model=Ridge s=10 frac=0.7fit_intercept=False normalize=False alpha=0.5000000000000001] score=3.3292772797589114
- [9][model=Ridge s=10 frac=0.7fit_intercept=False normalize=True alpha=0.7000000000000001] score=3.3292933250474643
- [10][model=Ridge s=10 frac=0.7fit_intercept=False normalize=False alpha=0.7000000000000001] score=3.3292933250474643


































































