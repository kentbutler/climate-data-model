# -*- coding: utf-8 -*-
"""Model_LSTMv3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MjxNrA02yP7bUY2likyhArio4_kZJxdm

---

## LSTM Model v3

---

Multivariate usage of an LSTM.
"""

from datetime import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense,RepeatVector, LSTM, Dropout
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Dropout, Reshape
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

class Model_LSTMv3:
  """
  """
  def __init__(self, window_size=30, label_window=1, num_labels=1, num_epochs=300, debug=False):
    self.WINDOW_SIZE = window_size
    self.LABEL_WINDOW = label_window
    self.NUM_LABELS = num_labels
    self.NUM_EPOCHS = num_epochs
    self.debug = debug

    self.MODEL_NAME = "LSTMv3"

    if self.debug:
      print(f'### Building Model{self.MODEL_NAME}::')

  def get_name(self):
    return self.MODEL_NAME

  def __repr__(self):
    """
    Print object stats.
    """
    return '\n'.join([
         f'window_size: {self.WINDOW_SIZE}',
         f'label_window: {self.LABEL_WINDOW}',
         f'num_labels: {self.NUM_LABELS}',
         f'num_epochs: {self.NUM_EPOCHS}'
        ])

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Build and train the model.
    """
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 25)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(self.WINDOW_SIZE, num_features)))
    #model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(RepeatVector(self.WINDOW_SIZE))
    model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Flatten())
    model.add(Dense(self.NUM_LABELS*self.LABEL_WINDOW))
    if (self.LABEL_WINDOW > 1):
      # reshape as => [batch, out_steps, labels]
      model.add(Reshape([self.LABEL_WINDOW, self.NUM_LABELS]))

    model.compile(loss='mae', optimizer='adam')

    if (dataset is not None):
      self.model_hist = model.fit(dataset, epochs=self.NUM_EPOCHS, callbacks = [early_stop], verbose=(1 if self.debug else 0))
    else:
      self.model_hist = model.fit(X_train, y_train, epochs=self.NUM_EPOCHS, verbose=1, callbacks = [early_stop] )
    self.model = model
    return self.model_hist

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