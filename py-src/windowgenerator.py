# -*- coding: utf-8 -*-
"""WindowGenerator.ipynb

##WindowGenerator

Windowing calculator - adapted from TensorFlow tutorial at:

https://www.tensorflow.org/tutorials/structured_data/time_series

Defines a windowing tool which can be applied to any given 2D DataFrame with the configured label column(s). Supports single or multi-label targets.

Transforms input into a time-stepped dataset prepared for supervised learning analysis.

Outputs a 3D dataset:  (batch, time, features)

where `time` is data per timestep, i.e. if training on a 1-year monthly lookback, then the `time` dimension should be 12.
"""

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

class TfWindowGenerator():
  """
  Construct a TfWindowGenerator that operates with the following params:
  * input_width - the number of time steps to use as the input window
  * label_width - the number of time steps to use as output
  * shift - offset that places the output prediction along the window; equal to
  or greater than the label_width
  * label_columns - name(s) of the label columns
  """
  def __init__(self, input_width, label_width, shift=1, batch_size=32, debug=False):
    # GUARDs
    if (input_width is None or label_width is None):
      raise AssertionError('input_width and label_width are required')

    self.debug = debug

    # Assess window parameters
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.batch_size = batch_size

    # Total number of data rows to extract per window/frame
    self.total_window_size = input_width + label_width
    # Index within a window/frame to look for the label(s)
    self.label_start = input_width


  def __repr__(self):
    """
    Print object stats.
    """
    return '\n'.join([
        'Window Generator::',
        f'\tInput size: {self.input_width}',
        f'\tLabel size: {self.label_width}',
        f'\tShift: {self.shift}',
        f'\tBatch size: {self.batch_size}',
        f'\tTotal window size: {self.total_window_size}',
        f'\tLabel start: {self.label_start}',
        ])


  def generate(self, df, label_cols, retain_labels=False):
    """
    Given a feature set of normalized values, create a windowed stack of
    inputs with output labels split out.

    If retain_labels is set, do not drop labels from features.

    Returns (inputs, outputs).
    """
    df_labels = df[label_cols]
    if (retain_labels):
      df_features = df
    else:
      df_features = df.drop(columns=label_cols)
    return self.generate_from_arrays(df_features.values, df_labels.values)

  def generate_from_arrays(self, input_arr, label_arr):
    # Restack data into frames the size of our total window
    # NOTE this currently retains all target columns
    LAST_WINDOW_START = input_arr.shape[0] - self.total_window_size

    print("LAST_WINDOW_START = input_arr.shape[0]-self.total_window_size")
    print(f"{LAST_WINDOW_START} = {input_arr.shape[0]} - {self.total_window_size}")

    # Convert incoming to np arrays
    input_arr = np.asarray(input_arr).astype('float32')
    label_arr = np.asarray(label_arr).astype('float32')
    print(f'input_arr: {input_arr.shape}')
    print(f'label_arr: {label_arr.shape}')

    frames = []
    labels = []

    #   Apply for each row
    for r in range(0, LAST_WINDOW_START):
      frames.append(input_arr[r:r+self.input_width])
      #print(f'### Slicing labels as: ({self.label_start+r}:{self.label_start+r+self.label_width})')
      labels.append(label_arr[self.label_start+r:self.label_start+r+self.label_width])

    frames = np.asarray(frames).astype('float32')
    labels = np.asarray(labels).astype('float32')

    if (self.debug):
      print (f'Frames: {frames.shape}')
      print (f'First frame:\n{frames[0]}')
      print (f'Last frame:\n{frames[-1]}')
      print (f'Labels: {labels.shape}')
      print (f'First label:\n{labels[0]}')
      print (f'Last label:\n{labels[-1]}')


    if (self.debug):
      print('\n'.join([
          'Returned shapes::',
          f'\tInputs: {frames.shape} {type(frames)}',
          f'\tLabels: {labels.shape} {type(labels)}'
      ]))

    return frames, labels

  def get_dataset(self, df, label_cols):
    inputs, labels = self.generate(df, label_cols)
    print(f'## Inputs: {inputs.shape} {type(inputs)}')
    print(f'## Labels: {labels.shape} {type(labels)}')

    inputs = tf.data.Dataset.from_tensor_slices(inputs)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds =  Dataset.zip(inputs, labels)
    ds = ds.batch(self.batch_size)
    return ds

  def get_ds_from_arrays(self, input_arr, label_arr):
    inputs, labels = self.generate_from_arrays(input_arr, label_arr)

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    print (f'#### converting labels: {type(labels)}\n{labels[0]}')
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    print(f'## Inputs: {inputs.shape} {type(inputs)}')
    print(f'## Labels: {labels.shape} {type(labels)}')

    inputs = tf.data.Dataset.from_tensor_slices(inputs)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds =  Dataset.zip(inputs, labels)
    ds = ds.batch(self.batch_size)
    return ds

"""---

**Unit testing**

---
"""

WG_UNIT_TEST = True

if WG_UNIT_TEST:
  from datetime import datetime as dt
  import datetime
  import pandas as pd

  a = []
  b = []
  c = []
  idx = []
  NUM_PTS = 22
  date_start = dt.strptime("1/1/11", "%m/%d/%y")

  for t in range(1,NUM_PTS+1):
    # just append some letter to t
    a.append(chr(96+t))
    b.append(t)
    c.append(t * 0.125)
    idx.append(date_start + datetime.timedelta(days=t))

  df = pd.DataFrame({'A':a,
                    'B': b,
                    'C':c},
                  index=idx)

  #print (df)

  # Label encode our target vals - going to use plain old py char vals
  df['A'] = df['A'].apply(lambda x: ord(x))

  # Scale values
  #df_mean = df.mean()
  #df_std = df.std()
  #df = (df - df_mean) / df_std

  print (df)
  print(f'Shape: {df.shape}')

if WG_UNIT_TEST:
  # Case 1: FAIL: Create failed windower - raises AssertionError
  IN_WIDTH=None
  LAB_WIDTH=1
  SHIFT=1
  LAB_COLS=['A']
  print('--- Case 1 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  try:
    win = TfWindowGenerator(input_width=None, label_width=2)
  except AssertionError:
    print ('Correct outcome - assert error')

if WG_UNIT_TEST:
  # Case 2: Create windower for single label timeframe, single label column
  IN_WIDTH=4
  LAB_WIDTH=1
  SHIFT=1
  LAB_COLS=['A']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4, debug=True)
  print('--- Case 2 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  # Split X/y
  inputs, labels = win.generate(df, LAB_COLS)

  print(f'Generated inputs: {inputs.shape} , labels: {labels.shape}')
  for batch in zip(inputs,labels):
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 3 Create dataset, single label column
  IN_WIDTH=4
  LAB_WIDTH=1
  SHIFT=1
  LAB_COLS=['A']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4, debug=True)
  print('--- Case 3 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  print('Making/iterating dataset...')
  ds = win.get_dataset(df, LAB_COLS)
  for batch in ds:
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 4 Create dataset, multi label columns
  IN_WIDTH=4
  LAB_WIDTH=2
  SHIFT=1
  LAB_COLS=['A','B']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4,debug=True)
  print('--- Case 4 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  print('Making/iterating dataset...')
  ds = win.get_dataset(df, LAB_COLS)
  for batch in ds:
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 5 Get windows input/outputs, multi label columns
  IN_WIDTH=4
  LAB_WIDTH=2
  SHIFT=1
  LAB_COLS=['A']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4, debug=True)
  print('--- Case 5 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  print('Making/iterating dataset...')
  ins,labs = win.generate(df, LAB_COLS)
  for batch in zip(ins,labs):
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 6 Get windows input/outputs, single label, multi label step
  IN_WIDTH=4
  LAB_WIDTH=4
  SHIFT=1
  LAB_COLS=['A']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4,debug=True)
  print('--- Case 6 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  print('Making/iterating dataset...')
  ins,labs = win.generate(df, LAB_COLS)
  for batch in zip(ins,labs):
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 7 Get windows input/outputs, multi label columns
  IN_WIDTH=4
  LAB_WIDTH=2
  SHIFT=1
  LAB_COLS=['A','B']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4,debug=True)
  print('--- Case 7 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  print('Making/iterating dataset...')
  ds = win.get_dataset(df, LAB_COLS)
  for batch in ds:
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break

if WG_UNIT_TEST:
  # Case 8 Get ds from arrays, multi label columns
  IN_WIDTH=4
  LAB_WIDTH=2
  SHIFT=1
  LAB_COLS=['A','C']
  win = TfWindowGenerator(input_width=IN_WIDTH, label_width=LAB_WIDTH, shift=SHIFT, batch_size=4,debug=True)
  print('--- Case 8 ------\n', f'input_width={IN_WIDTH}, label_width={LAB_WIDTH}, shift={SHIFT}, label_columns={LAB_COLS}')
  print(win)

  val_arr = df[['A','B']].values
  lab_arr = df[['A','C']].values

  print('Get ds from arrays...')
  ds = win.get_ds_from_arrays(val_arr, lab_arr)
  for batch in ds:
    X, y = batch
    print(f'-----X-----------\n{X}')
    print(f'-----y-----------\n{y}')
    break