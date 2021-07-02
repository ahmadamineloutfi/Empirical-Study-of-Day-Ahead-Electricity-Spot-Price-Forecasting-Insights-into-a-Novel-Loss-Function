# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import sys
import time

sys.path.append(r'D:\Time Series')

from KNN import get_X_Y_B3
from metric import *
from loss import *

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import datetime
import os
from itertools import combinations

from model import model_selection_1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print(time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

tf.config.set_visible_devices([], 'GPU')

np.random.seed(11)
tf.random.set_seed(11)

epochs = 5
opt = RMSprop()

model_name = ['NN','CNN','LSTM','GRU','RNN']

batch_size = 32
step = 1


take0 = list(combinations([0, 1, 2, 3, 4, 5, 6], 0))
take_list = take0

# Data
x, y, t_x, t_y, df = get_X_Y_B3()
x['Spot Prices (Auction) (EUR)'] = y
t_x['Spot Prices (Auction) (EUR)'] = t_y

# Loss functions
loss_name = ['theil2', 'theil2_abs', 'theil2_s', 'mae', 'mse']

# Lookback 12
for lookback in range(1,13):
    def train_generator(data, lookback=lookback, delay=0, min_index=0, max_index=None,
                        shuffle=False, batch_size=32, step=1):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while(1):
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size + 13 >= max_index:
                    # i = min_index + lookback
                    break
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay]
            yield samples, targets[:, 0, :]

    dm3 = []
    
# Modeling    

    for each_model in model_name:
        for zero_index in take_list:

            data = x

            float_data = data.values
            db = tf.data.Dataset.from_generator(lambda: train_generator(float_data), output_types=(tf.float32, tf.float32))
            traindb = db.take(50)
            devdb = db.skip(50)


            tar_dev_batch_list = []
            for b, (src_batch, tar_batch) in enumerate(devdb):
                tar_dev_batch_list.extend(list(tar_batch[:, -1].numpy()))


            print(each_model,time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
            for l, l_name in enumerate(loss_name):

                model = model_selection_1(each_model, lookback, step, float_data)

                loss_list = []
                train_overfit = []
                dev_overfit = []
                for epoch in range(epochs):

# Training 
                    for src_batch, tar_batch in traindb:
                        with tf.GradientTape() as tape:
                            ls = loss_fn(model, src_batch, tar_batch)
                        grads = tape.gradient(ls[l], model.trainable_variables)
                        opt.apply_gradients(zip(grads, model.trainable_variables))
                    loss_list.append(tf.reduce_mean(ls[l]).numpy())
                    train_overfit.append(tf.reduce_mean(ls[4]).numpy())

                    data = x
                    float_data = data.values
                    db = tf.data.Dataset.from_generator(lambda: train_generator(float_data),
                                                        output_types=(tf.float32, tf.float32))
                    devdb = db.skip(50)
                    predict = model.predict(devdb)
                    predict = predict[:,-1]
                    metrics = metrics_fn(tar_dev_batch_list,
                                         predict)

                    dm3.append(metrics[0])
                    
                with open(r"D:\Time Series (1)\backup\dev.csv" , 'a', encoding='utf-8') as dev:
                    dev.write('%s--%s--lookback %d\n' % (each_model, str(l_name), lookback))
                    dev.write(str(train_overfit)[1:-1])
                    dev.write('\n')
                    dev.write(str(dm3)[1:-1])
                    dev.write('\n')
                    dm3 = []
                    train_overfit = []

                with open(r"D:\Time Series (1)\backup\loss.csv" , 'a', encoding='utf-8') as log0:
                    log0.write('%s--%s--lookback %d\n' % (each_model,str(l_name),lookback))
                    log0.write(str(loss_list)[1:-1])
                    log0.write('\n')
                    loss_list = []

# Testing            

                data = t_x
                float_data = data.values

                testdb = tf.data.Dataset.from_generator(lambda: train_generator(float_data,lookback = lookback),
                                                         output_types=(tf.float32, tf.float32))

                m1,m2,m3,m4 = [],[],[],[]
                tar_batch_list = []
                predict_batch_list = []
                for e, (src_batch, tar_batch) in enumerate(testdb):
                    predict = model.predict(src_batch)
                    tar_batch_list.extend(list(tar_batch[:, -1].numpy()))
                    predict_batch_list.extend(list(predict[:, -1]))
                    metrics = metrics_fn(tar_batch[:, -1],
                                          predict[:, -1])
                    m1.append(metrics[0])
                    m2.append(metrics[1])
                ave_metrics = (np.min(m1),np.min(m2))



                with open(r"D:\Time Series (1)\backup\metrics.csv" , 'a', encoding='utf-8') as log1:
                    log1.write('%s--%s--lookback %d\n' % (each_model,str(l_name),lookback))
                    log1.write(str(ave_metrics)[1:-1])
                    log1.write('\n')


                x_axis = [
                    (datetime.datetime.strptime('2018-06-26', '%Y-%m-%d') + datetime.timedelta(
                        days=i)).strftime('%Y-%m-%d') for i in range(len(tar_batch_list))]
                for i,date in enumerate(x_axis):
                    if date =='2018-07-15':
                        ind = i
                        break

                with open(r"D:\Time Series (1)\backup\prediction.csv", 'a', encoding='utf-8') as pre:
                    pre.write("%s--%s--lookback %d\n" %  (each_model, str(l_name),lookback))
                    # pre.write("%s\n" % ','.join(x_axis[lookback-1 + ind:250+lookback-1 + ind]))
                    pre.write("%s\n" % ','.join(x_axis[ind:580 + lookback - 1]))
                    for each in tar_batch_list[ind-lookback +1: 580]:
                        pre.write("%s," % str(each))
                    pre.write("\n")
                    for each in predict_batch_list[ind-lookback +1: 580]:
                        pre.write("%s," % str(each))
                    pre.write("\n")

                x_axis_polt = [
                    (datetime.datetime.strptime('2018-06-26', '%Y-%m-%d') + datetime.timedelta(
                        days=i)).strftime('%Y-%m-%d') for i in range(len(tar_batch_list))]


                fig = plt.figure()
                fig.autofmt_xdate()
                plt.tick_params(axis='x', labelsize=8)
                plt.xticks(range(0, len(tar_batch_list[ind:580 + lookback - 1]), 90))
                plt.plot(x_axis[ind:580 + lookback - 1], tar_batch_list[ind-lookback +1: 580])
                plt.plot(x_axis[ind:580 + lookback - 1], predict_batch_list[ind-lookback +1: 580])
                plt.ylim(22.5, 65)
                plt.savefig(fname=r"D:\Time Series (1)\backup\%s/%s/lookback %d.svg" % (each_model, str(l_name), lookback))
                plt.close()

            print('finish', each_model, zero_index, time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

print('finish all', time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
