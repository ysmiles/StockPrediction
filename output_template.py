import numpy as np
import pandas

output_data = []

train_csv = pandas.read_csv('data/train.csv', index_col=0)
test_csv = pandas.read_csv('data/test_2.csv', index_col=0)

train_X = train_csv.drop(train_csv.columns[range(146, 210)], axis=1).values


for i in range(62):  # t=121 to 180, and D+1, D+2
    if i == 60:
        name_of_column = 'Ret_PlusOne'
        name_of_weight = 'Weight_Daily'
    elif i == 61:
        name_of_column = 'Ret_PlusTwo'
        name_of_weight = 'Weight_Daily'
    else:
        name_of_column = 'Ret_' + str(i + 120)
        name_of_weight = 'Weight_Intraday'

    train_y = train_csv[name_of_column].values
    train_weights = train_csv[name_of_weight].values

    test_X = test_csv.values

    # training and predict logics
    # model.train()
    # pred = model.predict()

    for stock_id, val in enumerate(pred):
        output_data.append(
            {'Id': str(stock_id + 1) + '_' + str(i), 'Predicted': val})


output = pandas.DataFrame(data=output_data)
output.sort_values(by='Id', inplace=True)
# print(output.head())
output.to_csv(path_or_buf='data/output.csv', index=False)
