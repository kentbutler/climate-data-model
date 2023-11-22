# -*- coding: utf-8 -*-
"""Model_LSTMv1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ta_ams7ca-TZuPW83F7RHDAKFd7n9jHc

---

## LSTM Model v1

---

Basic usage of an LSTM - single variable.
"""

import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense,RepeatVector, LSTM, Dropout
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Dropout
from keras.models import Sequential
from keras.utils import plot_model

class Model_LSTMv1:
  """
  Constructs a model instance. Input data should be a dataframe containing
  only the data to be modeled.
  """
  debug = False

  def __init__(self, window_size=30, label_window=1, num_labels=1, num_epochs=300, debug=False):

    self.WINDOW_SIZE = window_size
    self.LABEL_WINDOW = label_window
    self.NUM_LABELS = num_labels
    self.NUM_EPOCHS = num_epochs
    self.debug = debug

    self.MODEL_NAME = "LSTMv1"

    if self.debug:
      print(f'### Building Model{self.MODEL_NAME}::')

  def get_name(self):
    return self.MODEL_NAME

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Declare model and train.
    Based on https://keras.io/examples/timeseries/timeseries_classification_from_scratch/.
    """
    from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 7)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30,1)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(30))
    model.add(LSTM(units=100, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='relu'))
    model.add(LSTM(units=100, return_sequences=True, activation='relu'))
    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(self.NUM_LABELS*self.LABEL_WINDOW))
    if (self.LABEL_WINDOW > 1):
      # reshape as => [batch, out_steps, labels]
      model.add(Reshape([self.LABEL_WINDOW, self.NUM_LABELS]))

    model.compile(loss='mse', optimizer='adam')

    self.model_hist = model.fit(X_train, y_train, epochs=self.NUM_EPOCHS, verbose=1, callbacks = [early_stop] )
    self.model = model

  def get_model_name(self, serial=None):
    return f"{dt.today().strftime('%Y%m%d-%H%M')}-{self.MODEL_NAME}-{serial if serial is not None else '0'}.hdf5"

  def save_model(self, path, serial=None):
    """
    Save the current model under the given drive path.
    Timestamp the model name.
    """
    fname = f'{path}{self.get_model_name(serial)}'
    print(f'Saving model to: {fname}')
    return self.model.save(fname)

  def predict(self, X_in):
    return self.model.predict(X_in)