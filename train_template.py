from load import loaddata
from sklearn.model_selection import KFold

data = loaddata('train.csv', 0.01)

X = data.loc[:, 'Feature_1':'Ret_120'].values
y = data.loc[:, 'Ret_121':'Ret_PlusTwo'].values

daily_weights = data['Weight_Daily'].values
intraday_weights = data['Weight_Intraday'].values

print(X.shape, y.shape)

# KFold(n_splits=2, random_state=None, shuffle=False)
# >>> for train_index, test_index in kf.split(X):
# ...    print("TRAIN:", train_index, "TEST:", test_index)
# ...    X_train, X_test = X[train_index], X[test_index]
# ...    y_train, y_test = y[train_index], y[test_index]
# TRAIN: [2 3] TEST: [0 1]
# TRAIN: [0 1] TEST: [2 3]

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]

    # train and test logics
    # ...
