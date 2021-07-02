# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import tensorflow as tf

def metrics_fn(real, prediction):

    real = tf.cast(real,dtype='float32')
    prediction = tf.cast(prediction,dtype='float32')

    # MAE
    mae = tf.keras.metrics.mean_absolute_error(real, prediction)

    # MSE
    mse = tf.keras.metrics.mean_squared_error(real, prediction)

    return mae.numpy(), mse.numpy()