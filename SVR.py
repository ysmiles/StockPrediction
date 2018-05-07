import pandas as pd
import numpy as np
from load import loaddata
from scipy import stats
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


data = loaddata('train.csv', 0.05)

X = data.loc[:, 'Feature_1':'Ret_120'].values
y = data.loc[:, 'Ret_121':'Ret_PlusTwo'].values

X_feature = data.loc[:, 'Feature_1':'Feature_25'].values
X_mins = data.loc[:, 'Ret_2':'Ret_120'].values
X_ret = data.loc[:, 'Ret_MinusTwo':'Ret_MinusOne'].values

daily_weights = data['Weight_Daily'].values
intraday_weights = data['Weight_Intraday'].values

#print(X.shape, y.shape)

imp = Imputer(missing_values='NaN', strategy="mean", axis=0)
X_imp = imp.fit_transform(X)
X_featureimp = imp.fit_transform(X_feature)
X_minsimp = imp.fit_transform(X_mins)
X_retimp = imp.fit_transform(X_ret)

#print(X.shape, y.shape)


def outliers_z_score(ys):  # z-score outlier detection
    threshold = 3.5
    z_scores = stats.zscore(ys)

    return np.where(np.abs(z_scores) > threshold)

#zscore = outliers_z_score(X_imp)
# print type(zscore)


clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.0005, gamma='auto',
          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

n = 2
kf = KFold(n_splits=n, shuffle=True)
r2 = 0
MSE = 0

for train_index, test_index in kf.split(X_imp):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, y_train = X_imp[train_index], y[train_index]
    print("TRAIN:", X_train.shape, "TEST:", y_train.shape)
    X_test = X_imp[test_index]
    y_true = y[test_index]
    y_pred = np.empty([len(test_index), y_true.shape[1]])

    i = 0
    for column in y_train.T:
        y_train1 = np.ravel(y_train[:, i:i+1])
        clf.fit(X_train, y_train1)
        y_pred1 = clf.predict(X_test)
        y_pred = np.column_stack((y_pred, y_pred1))
        i += 1

    y_pred = y_pred[:, y_true.shape[1]:]

    r2 = r2 + r2_score(y_true, y_pred)
    print(r2_score(y_true, y_pred))
    MSE = MSE + mean_squared_error(y_true, y_pred)
    print(mean_squared_error(y_true, y_pred))


r2_avr = r2/n
MSE_avr = MSE/n
print(r2_avr)
print(MSE_avr)


regr = RandomForestRegressor(max_depth=10, random_state=0)

n = 2
kf = KFold(n_splits=n, shuffle=True)
r2 = 0
MSE = 0

for train_index, test_index in kf.split(X_imp):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, y_train = X_imp[train_index], y[train_index]
    print("TRAIN:", X_train.shape, "TEST:", y_train.shape)
    X_test = X_imp[test_index]
    y_true = y[test_index]
    y_pred = np.empty([len(test_index), y_true.shape[1]])

    i = 0
    for column in y_train.T:
        y_train1 = np.ravel(y_train[:, i:i+1])
        regr.fit(X_train, y_train1)
        y_pred1 = regr.predict(X_test)
        y_pred = np.column_stack((y_pred, y_pred1))
        # y_pred.hstack((y_pred,y_pred1))
        i += 1

    y_pred = y_pred[:, y_true.shape[1]:]

    r2 = r2 + r2_score(y_true, y_pred)
    print(r2_score(y_true, y_pred))
    MSE = MSE + mean_squared_error(y_true, y_pred)
    print(mean_squared_error(y_true, y_pred))


r2_avr = r2/n
MSE_avr = MSE/n
print(r2_avr)
print(MSE_avr)
