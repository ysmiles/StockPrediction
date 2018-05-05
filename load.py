import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# print(pd.__version__)

def loaddata(ratio=0.05):
    data = pd.read_csv('data/train.csv', index_col=0, nrows=int(ratio * 40000))
    return data


def save_histograms(data):
    # i = 0
    for feature_name in data.columns.values:
        # data[feature_name].hist()
        data.hist(feature_name)
        plt.title(feature_name)
        plt.savefig('img/histogram/' + feature_name + '.png')
        # plt.show()
        plt.clf()  # clear figure but leave figure opened
        # plt.close()  # close the window
        # i += 1
        # if i == 2:
        #     break


data = loaddata(1.0)

# save_histograms(data)
