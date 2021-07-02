# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error

def loss_fn(model, inputs, real):
    
    prediction = model(inputs)

    real = tf.cast(real,dtype='float32')
    prediction = tf.cast(prediction,dtype='float32')

    # Theil UII
    theil2 = (K.sqrt(mean_squared_error(real, prediction))) / (K.sqrt(mean_squared_error(real, tf.zeros_like(real))))

    # Theil UII Absolute Value
    theil2_abs = (tf.keras.losses.mean_absolute_error(real, prediction)) \
            / (tf.keras.losses.mean_absolute_error(real, tf.zeros_like(real)))

    # Theil UII squared
    theil2_s = (tf.keras.losses.mean_squared_error(real, prediction)) \
            / (tf.keras.losses.mean_squared_error(real, tf.zeros_like(real)))

    # MAE
    mae = tf.keras.losses.mean_absolute_error(real, prediction)

    # MSE
    mse = tf.keras.losses.mean_squared_error(real, prediction)

    return theil2, theil2_abs, theil2_s, mae, mse