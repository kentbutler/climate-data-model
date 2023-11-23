# -*- coding: utf-8 -*-
"""

## Base Model

---
Base class of models.
"""

from datetime import datetime as dt

class Base_Model():
  """
  """
  def __init__(self, window_size=30, label_window=1, num_labels=1, num_epochs=300, debug=False):
    self.WINDOW_SIZE = window_size
    self.LABEL_WINDOW = label_window
    self.NUM_LABELS = num_labels
    self.NUM_EPOCHS = num_epochs
    self.debug = debug

    if self.debug:
      print(f'### Building Model{self.get_name()}::')

  def __repr__(self):
    """
    Print object stats.
    """
    return '\n'.join([
         f'Model: {self.get_name()}',
         f'\twindow_size: {self.WINDOW_SIZE}',
         f'\tlabel_window: {self.LABEL_WINDOW}',
         f'\tnum_labels: {self.NUM_LABELS}',
         f'\tnum_epochs: {self.NUM_EPOCHS}'])

  def get_name(self):
    """
    This model name.  Not unique.
    :return:
    """
    return 'Base'

  def get_model(self):
    """
    Return wrapped TF model class.
    :return:
    """
    return self.model

  def print_summary(self):
    """
    Print summary of internal wrapped model. Pass-through.
    :return:
    """
    if (not self.model or self.model is None):
      return 'No model'
    self.model.summary()

  def set_model(self, model):
    """
    Set the internal model. Used when loading from storage.
    :param model:
    :return:
    """
    if (not model or model is None):
      raise AssertionError('Model given is empty')
    self.model = model

  def train(self, X_train=None, y_train=None, num_features=None, dataset=None):
    raise NotImplementedError

  def predict(self, X_in):
    return self.model.predict(X_in)

  def save_model(self, path, serial=None):
    """
    Save the current model under the given drive path.
    Timestamp the model name.
    """
    if (not self.model or self.model is None):
      raise AssertionError('No model')

    fname = f'{path}{self.get_model_filename(serial)}'
    print(f'Saving model to: {fname}')
    return self.model.save(fname)

  def get_model_filename(self, serial=None):
    """
    Return a unique filename for saving this model's state.
    :param serial:
    :return:
    """
    return f"{dt.today().strftime('%Y%m%d-%H%M')}-{self.get_name()}-{serial if serial is not None else '0'}.hdf5"

