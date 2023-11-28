# -*- coding: utf-8 -*-
"""Model_LSTMv3.ipynb
---

## LSTM Model v3

---

Multivariate usage of an LSTM.
"""

from datetime import datetime as dt
from keras.layers import Dense,RepeatVector, LSTM, Dropout
from keras.layers import Flatten, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from model_base import Base_Model

class Model_LSTMv32(Base_Model):

  def get_name(self):
    return "LSTMv32"

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Build and train the model.
    """
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 25)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='gelu', input_shape=(self.WINDOW_SIZE, num_features)))
    #model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(RepeatVector(self.WINDOW_SIZE))
    model.add(LSTM(units=128, return_sequences=True, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    #model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Flatten())
    model.add(Dense(128, activation='gelu'))
    model.add(Flatten())
    model.add(Dense(self.NUM_LABELS*self.LABEL_WINDOW))
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
