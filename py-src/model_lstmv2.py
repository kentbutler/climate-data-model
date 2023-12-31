# -*- coding: utf-8 -*-
"""Model_LSTMv2.ipynb

## LSTM Model v2

---

Multivariate usage of an LSTM.
"""

from keras.layers import Dense,RepeatVector, LSTM, Dropout, Reshape
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from model_base import Base_Model

class Model_LSTMv2(Base_Model):

  def get_name(self):
    return "LSTMv2"

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Build and train the model.
    """
    LSTM_SIZE=128
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 25)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(self.WINDOW_SIZE, num_features)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(self.WINDOW_SIZE))
    model.add(LSTM(units=LSTM_SIZE, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=LSTM_SIZE, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=LSTM_SIZE, return_sequences=True, activation='tanh'))
    model.add(LSTM(units=LSTM_SIZE, return_sequences=True, activation='tanh'))
    model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Dense(LSTM_SIZE, activation='relu'))
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
