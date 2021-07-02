# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

from tensorflow.keras import layers
import tensorflow as tf

def model_selection_1(each_model,lookback,step,float_data):

    if each_model == 'CNN':
        model = tf.keras.Sequential()
        model.add(
            layers.Conv1D(64, 2, padding='same', activation='relu',
                          input_shape=(lookback // step, float_data.shape[-1])))
        model.add(layers.Flatten())
        model.add(layers.Dense(8))
        return model

    if each_model == 'RNN':
        model = tf.keras.Sequential()
        model.add(
            layers.SimpleRNN(64, activation='relu', input_shape=(lookback // step, float_data.shape[-1])))
        model.add(layers.Dense(8))
        return model
        
        
    if each_model == 'LSTM':
        model = tf.keras.Sequential()
        model.add(
            layers.LSTM(64, activation='relu', input_shape=(lookback // step, float_data.shape[-1])))
        model.add(layers.Dense(8))
        return model

    if each_model == 'NN':
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
        model.add(
            layers.Dense(64, activation='relu'))
        model.add(layers.Dense(8))
        return model

    if each_model == 'GRU':
        model = tf.keras.Sequential()
        model.add(
            layers.GRU(64, activation='relu', input_shape=(lookback // step, float_data.shape[-1])))
        model.add(layers.Dense(8))
        return model