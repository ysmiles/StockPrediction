# Yu Shu, 2018 Spring
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from load import loaddata


#-------------------------------------------------------------------------------
# read data
#-------------------------------------------------------------------------------
data = loaddata('train.csv', 1.0)
features = data.loc[:, 'Feature_1':'Feature_25'].values
rets = data.loc[:, 'Ret_MinusTwo':'Ret_PlusTwo'].values
daily_weights = data['Weight_Daily'].values
intraday_weights = data['Weight_Intraday'].values

test_data = loaddata('test_2.csv', 1.0)
test_features = test_data.loc[:, 'Feature_1':'Feature_25'].values
test_rets = test_data.loc[:, 'Ret_MinusTwo':'Ret_120'].values

#-------------------------------------------------------------------------------
# preprocess
#-------------------------------------------------------------------------------
imp_axis0 = Imputer(missing_values='NaN', strategy="mean", axis=0)
imp_axis1 = Imputer(missing_values='NaN', strategy="mean", axis=1)

features = imp_axis1.fit_transform(features)
rets = imp_axis0.fit_transform(rets)

test_features = imp_axis1.fit_transform(test_features)
test_rets = imp_axis0.fit_transform(test_rets)

rets_minus = rets[:, :2]
rets_ts = rets[:, 2:-2]  # time series
rets_plus = rets[:, -2:]

test_rets_minus = test_rets[:, :2]
test_rets_ts = test_rets[:, 2:]  # time series


for i in range(rets.shape[0]):
    rets_minus[i] *= daily_weights[i]
    rets_plus[i] *= daily_weights[i]
    rets_ts[i] *= intraday_weights[i]


test_rets_minus *= 1.5e6
test_rets_ts *= 1.5e6

# print(rets_ts.shape)
# 40000, 179

# rets_mid = tf.reduce_sum(rets_ts[:, :119], 1, keepdims=True)  # today summary
# print(rets_minus.shape)
# print(rets_mid.shape)
# # 40000, 1
# rets_ts2 = np.append(rets_minus, rets_mid)
# rets_ts2 = np.append(rets_ts2, rets_plus)
# print(rets_ts2.shape)


pca = PCA(n_components=10)
pca.fit(features)
features = pca.transform(features)
# print(features.shape)
test_features = pca.transform(test_features)

#-------------------------------------------------------------------------------
# train setting
#-------------------------------------------------------------------------------
num_batches = int(rets.shape[0] / 100)        # batch size
num_periods = 119       # number of periods per vector using to predict one period ahead
num_inputs = 1             # number of vectors submitted
num_outputs = 1             # number of output vectors


def get_batch(data, n_batches, n_periods, n_inputs=1, n_outputs=1):
    idx = np.random.randint(data.shape[0], size=n_batches)
    x_batch = data[idx, :n_periods]
    y_batch = data[idx, -n_periods:]

    x_batch = x_batch.reshape(-1, n_periods, n_inputs)
    y_batch = y_batch.reshape(-1, n_periods, n_outputs)

    print('getting \nx_batch with size', x_batch.shape,
          '\ny_batch with size', y_batch.shape)
    return x_batch, y_batch, idx




def embed_features(features, x_batch, idx, n_batches, n_periods):
    batch_features = features[batch_idx]
    print(batch_features.shape)
    temp = np.zeros([n_batches, n_periods, batch_features.shape[1]])
    for j in range(x_batch.shape[0]):
        for kk in range(x_batch.shape[1]):
            temp[j, kk] = batch_features[j]
    x_batch = np.append(x_batch, temp, axis=2)
    return x_batch
    # features = features[idx]
    # # stock_features =  tf.placeholder(tf.int32, [None, features.shape[1]])
    # stock_features =  tf.placeholder(tf.int32, [None, 1])
    # embedding_matrix = tf.Variable(features)
    # stacked_stock_features = tf.tile(stock_features, multiples=[1, n_periods])
    # # stacked_stock_features = tf.tile(stock_features, multiples=[features.shape[1], n_periods])
    # stock_feature_embeds = tf.nn.embedding_lookup(embedding_matrix, stacked_stock_features)
    # inputs_with_embeds = tf.concat([x_batch, stock_feature_embeds], axis=2)
    # print("features shape", features.shape)
    # print("after embeding, the input shape", inputs_with_embeds.shape)
    # return inputs_with_embeds


hidden = 512             # number of neurons we will recursively work through,
epochs = 100          # number of iterations or training cycles
repeats = 1          # number of iterations or training cycles
learning_rate = 0.001  # small learning rate so we don't overshoot the minimum


# We didn't have any previous graph objects running, but this would reset the graphs
tf.reset_default_graph()

# create variable objects (placeholders)
X = tf.placeholder(
    tf.float32, [None, num_periods, num_inputs + features.shape[1]])
y = tf.placeholder(tf.float32, [None, num_periods, num_outputs])

# create our RNN cells
basic_cell = tf.contrib.rnn.BasicRNNCell(
    num_units=hidden, activation=tf.nn.relu)

# get output
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# change the form into a tensor
stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
# specify the type of layer (dense)
stacked_outputs = tf.layers.dense(stacked_rnn_output, num_outputs)
# shape of results
outputs = tf.reshape(stacked_outputs, [-1, num_periods, num_outputs])

# define the cost function which evaluates the quality of our model
loss = tf.reduce_sum(tf.square(outputs - y))
# gradient descent method
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train the result of the application of the cost_function
training_op = optimizer.minimize(loss)


#-------------------------------------------------------------------------------
# start training
#-------------------------------------------------------------------------------
init = tf.global_variables_initializer()  # initialize all the variables

mses = []
with tf.Session() as sess:
    init.run()
    for batch_no in range(repeats):
        x_batch, y_batch, batch_idx = get_batch(
            rets_ts, num_batches, num_periods)
        x_batch = embed_features(
            features, x_batch, batch_idx, num_batches, num_periods)
        # print("now", x_batch.shape)
        # x_batch = sess.run(inputs_with_embeds).reshape(-1, num_periods, num_inputs + features.shape[0])
        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            mse = loss.eval(
                feed_dict={X: x_batch, y: y_batch}) / num_batches / num_periods
            mses.append(mse)
            if ep % 100 == 99:
                # mean square error
                print("after", ep + 1, "epochs", "\tMSE:", mse)

    # test
    num_batches = 1
    x_batch, y_batch, batch_idx = get_batch(rets_ts, num_batches, num_periods)
    x_batch = embed_features(
        features, x_batch, batch_idx, num_batches, num_periods)

    y_pred = sess.run(outputs, feed_dict={X: x_batch})
    y_b = y_batch.reshape(-1)
    y_p = y_pred.reshape(-1)
    print(y_b)
    print(y_p)
    # plt.subplot(1, 2, 1)
    plt.plot(y_b, label='original')
    plt.plot(y_p, label='predicted')
    plt.legend(loc="upper left")
    plt.xlabel("time steps")
    plt.ylabel("returns of a randomly picked stock")
    plt.savefig("img/rnn_sample_run_res.png")
    plt.clf()

    num_batches = test_rets.shape[0]
    data = test_rets
    for i in range(100):
        idx = range(i * int(num_batches/100), (i+1) * int(num_batches/100))
        x_batch = data[idx, :num_periods]
        y_batch = data[idx, -num_periods:]

        x_batch = x_batch.reshape(-1, num_periods, num_inputs)
        y_batch = y_batch.reshape(-1, num_periods, num_outputs)

        x_batch = embed_features(test_features, x_batch, idx, int(num_batches/100), num_periods)

        y_pred = sess.run(outputs, feed_dict={X: x_batch}).reshape(-1, num_periods)
        np.savez_compressed('y_pred_' + str(i) + '.npz', data=y_pred)
        print(y_pred.shape)
    # for i in range(rets.shape[0]):
    #     rets_minus[i] *= daily_weights[i]
    #     rets_plus[i] *= daily_weights[i]
    #     rets_ts[i] *= intraday_weights[i]

epochs = np.arange(epochs * repeats)
mses = np.array(mses)

# plt.subplot(1, 2, 2)
plt.plot(epochs, mses/1e5)
# plt.plot(epochs[150:], mses[150:])
# plt.yscale('log')
plt.xlabel("training epocs")
plt.ylabel("MSE/10^5")
plt.savefig("img/rnn_sample_run_mse.png")

plt.show()

# Prepare to save data
print("prepare to save data")
