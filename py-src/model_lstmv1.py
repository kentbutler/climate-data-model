# -*- coding: utf-8 -*-
"""Model_LSTMv1.ipynb
---

## LSTM Model v1
---
Basic usage of an LSTM - single variable.
"""

from keras.layers import Dense,RepeatVector, LSTM, Dropout, Reshape
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from model_base import Base_Model

class Model_LSTMv1(Base_Model):
  """
  Constructs a model instance. Input data should be a dataframe containing
  only the data to be modeled.
  """

  def get_name(self):
    return "LSTMv1"

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Declare model and train.
    Based on https://keras.io/examples/timeseries/timeseries_classification_from_scratch/.
    """
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 7)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(self.WINDOW_SIZE, num_features)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(30))
    model.add(LSTM(units=100, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(LSTM(units=100, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(self.NUM_LABELS*self.LABEL_WINDOW, kernel_initializer='glorot_uniform'))

    if (self.LABEL_WINDOW > 1):
      # reshape as => [batch, out_steps, labels]
      model.add(Reshape([self.LABEL_WINDOW, self.NUM_LABELS]))

    model.compile(loss='mse', optimizer='adam')

    if (dataset is not None):
      self.model_hist = model.fit(dataset, epochs=self.NUM_EPOCHS, callbacks = [early_stop], verbose=(1 if self.debug else 0))
    else:
      self.model_hist = model.fit(X_train, y_train, epochs=self.NUM_EPOCHS, verbose=1, callbacks = [early_stop] )
    self.model = model
    return self.model_hist
