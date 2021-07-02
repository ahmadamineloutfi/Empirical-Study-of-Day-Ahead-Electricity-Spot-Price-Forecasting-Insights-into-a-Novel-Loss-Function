# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import sys
import time

sys.path.append(r'D:\Time Series')

import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_X_Y_B3():
    
    df = pd.read_excel('D:\Time Series\#E_data.xlsx')
    
    dataset = df[['Consumption Prognosis (MVH)','Production Prognosis (MVH)','Wind prognosis (MVH)'
        ,'Prices OQF (EUR)','Prices OYF (EUR)','Brent Oil (EUR)'
        ,'Coal Price (EUR)','Spot Prices (Auction) (EUR)']]

    dataset = pd.DataFrame(data=KNNImputer(n_neighbors=7).fit_transform(dataset),
                      index=dataset.index, columns=dataset.columns)
    dataset = dataset[1:-1]

    train_dataset = dataset.head(2000)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("Spot Prices (Auction) (EUR)")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop("Spot Prices (Auction) (EUR)")
    test_labels = test_dataset.pop("Spot Prices (Auction) (EUR)")

    def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    origins = dataset
    origins['Date'] = df['Date']
    return normed_train_data, train_labels, normed_test_data, test_labels, origins

if __name__ == '__main__':

    x, y, t_x, t_y = get_X_Y_B3()

    def build_model():
      model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[7]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
      ])

      optimizer = tf.keras.optimizers.RMSprop(0.001)

      model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
      return model

    model = build_model()
    model.summary()
    EPOCHS = 100
    history = model.fit(x, y,
      epochs=EPOCHS, validation_split = 0.2)

    test_predictions = model.predict(t_x).flatten()

    plt.scatter(t_y, test_predictions)
    plt.xlabel('True Values [price]')
    plt.ylabel('Predictions [price]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.show()