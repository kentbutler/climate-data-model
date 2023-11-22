# -*- coding: utf-8 -*-
"""Model_Densev1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SFDRnJ5y-Ig7EVF4KGeOm_0hdEolwOzC

---

## Dense Model v1

---

Multivariate usage of a Dense NN.
"""

from keras.layers import Dense,MaxPooling1D
from keras.layers import Flatten,RepeatVector,Reshape
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from model_base import Base_Model

class Model_Densev11(Base_Model):

  def get_name(self):
    return "Densev11"

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    """
    Build and train the model.
    """
    early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 25)
    model = Sequential()
    model.add(MaxPooling1D(pool_size=2, input_shape=(self.WINDOW_SIZE, num_features)))
    model.add(Dense(128, activation='gelu'))
    model.add(Dense(128, activation='gelu'))
    model.add(Dense(128, activation='gelu'))
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
