# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import sys
import time

sys.path.append(r'D:\Time Series')

from KNN import get_X_Y_B3
from nearest_value import get_X_Y_B5
from cubic_spline import get_X_Y_B2
from ignore import get_X_Y_B1
from mean import get_X_Y_B4

from metric import *
from loss import *

import matplotlib.pyplot as plt
import tensorflow as tf
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

model_name = ['NN']

batch_size = 32
step = 1
lookback = 7


take0 = list(combinations([0, 1, 2, 3, 4, 5, 6], 0))
take_list = take0


get_X_Y = [get_X_Y_B1(),get_X_Y_B2(),get_X_Y_B3(),get_X_Y_B4(),get_X_Y_B5()]


missingname = ['ignor','cubic_spline','KNN','mean','nearest']
for e_XY, X_Y in enumerate(get_X_Y):
    x, y, t_x, t_y, df = X_Y
    x['Spot Prices (Auction) (EUR)'] = y
    t_x['Spot Prices (Auction) (EUR)'] = t_y
    if e_XY == 0:
        datasplit = 30
    else:
        datasplit = 50
    loss_name = ['mse']


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
                if i + batch_size + 21 >= max_index:
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

    for each_model in model_name:
        for zero_index in take_list:
            data = x
            float_data = data.values
            db = tf.data.Dataset.from_generator(lambda: train_generator(float_data), output_types=(tf.float32, tf.float32))
            traindb = db.take(datasplit)
            devdb = db.skip(datasplit)


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
                    # dev
                    data = x
                    float_data = data.values
                    db = tf.data.Dataset.from_generator(lambda: train_generator(float_data),
                                                        output_types=(tf.float32, tf.float32))
                    devdb = db.skip(datasplit)
                    predict = model.predict(devdb)
                    predict = predict[:,-1]
                    metrics = metrics_fn(tar_dev_batch_list,
                                         predict)

                    dm3.append(metrics[0])

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
                    log1.write('%s--%s--%s\n' % (each_model,str(l_name),missingname[e_XY]))
                    log1.write(str(ave_metrics)[1:-1])
                    log1.write('\n')


            print('finish', each_model, zero_index, time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

    print('finish all', time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
