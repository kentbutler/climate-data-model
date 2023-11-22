# -*- coding: utf-8 -*-
"""ModelExecutor.ipynb

## Data606 - Capstone Project
```
Group H
Malav Patel, Kent Butler
Prof. Unal Sokaglu
```

This project is about performing time-series analysis on climate data analysis data.

# Research

### References

Some explanations of earth sciences statistics:
https://pjbartlein.github.io/REarthSysSci/ltms-and-anomalies.html

NOAA PSL NCEP-NCAR datasets:  https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

NOAA PSL, other recognized data sources directory: https://psl.noaa.gov/data/help/othersources/

Global environmental policy timeline, https://www.boell.de/en/2022/05/28/international-environmental-policy-timeline

OECD convergence of policy, climate,and economy: https://www.oecd.org/

NASA climate time machine: https://climate.nasa.gov/interactives/climate-time-machine

### Factoids

* All of the plastic waste produced in the world in 2019 alone weighs as much as 35,000 Eiffel Towers – 353 million tons  - [*Organization for Economic Cooperation and Development (OECD)*](https://www.boell.de/en/2022/05/28/international-environmental-policy-timeline)

## Application Parameters

Note: algorithm tuning is done with declaration of the model.
"""

import pandas as pd
from datetime import datetime as dt
import datetime
import numpy as np
import math
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, \
  QuantileTransformer, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns; sns.set()
plt.rcParams["figure.figsize"] = (10,6)
import warnings

# Import local source
from projectutil import *
from windowgenerator import TfWindowGenerator
from modelfactory import ModelFactory
from dataset_merger import Dataset_Merger

DRIVE_PATH = "/content/drive/MyDrive/data606"

# Set the location of this script in GDrive
SCRIPT_PATH = DRIVE_PATH + "/py-src/"

"""**Configure Predictions**"""

"""# Data Load"""



# Commented out IPython magic to ensure Python compatibility.
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

# To suppress rendering of graphics
#plt.ion()

"""---

**Initial Data Load**

---
"""

class ModelExecutor():

  def __init__(self, data_path, log_path, journal_log, start_date, end_date, input_window, label_window, shift, test_ratio, val_ratio, num_epochs, target_label, model_name, debug=False):
    self.debug = debug
    self.DATA_PATH = data_path
    self.LOG_PATH = log_path
    self.JOURNAL_LOG = journal_log
    self.START_DATE = start_date
    self.END_DATE = end_date
    self.INPUT_WINDOW = input_window
    self.LABEL_WINDOW = label_window
    self.SHIFT = shift
    self.TEST_RATIO = test_ratio
    self.VAL_RATIO = val_ratio
    self.NUM_EPOCHS = num_epochs
    self.TARGET_LABEL = target_label
    self.TARGET_LABELS = [target_label] # for future expansion
    self.MODEL_NAME = model_name

    # Device to run on
    self.run_on_device =  'cpu' # 'cuda'
    # Other internal state
    self.model_factory = None
    self.model = None

  def load_initial_dataset(self, ds_name, feature_map, date_map=None, date_col=None):
    # Declare a merger compatible with our source data and our target dataset we want to merge into
    self.merger = Dataset_Merger(data_path=self.DATA_PATH,
                            start_date=self.START_DATE, end_date=self.END_DATE,
                            debug=self.debug)

    # Start by merging initial dataset
    if (date_map is not None):
      self.df_merge = self.merger.merge_dataset(ds_name, feature_map, date_map=date_map, add_cyclic=True)
    elif (date_col is not None):
      self.df_merge = self.merger.merge_dataset(ds_name, feature_map, date_col=date_col, add_cyclic=True)

    print(f'#### Feature map: {type(feature_map)} {feature_map.values()}')
    self.STEP_FREQ = self.merger.assess_granularity(self.df_merge, list(feature_map.values()))

    print(assess_na(self.df_merge))


  def load_datasets(self, ds_list):
    for dataset in ds_list:
      if ('date_map' in dataset):
        self.df_merge = self.merger.merge_dataset(dataset['filename'],
                                        feature_map=dataset['feature_map'],
                                        df_aggr=self.df_merge,
                                        date_map=dataset['date_map'])
      else:
        self.df_merge = self.merger.merge_dataset(dataset['filename'],
                                    feature_map=dataset['feature_map'],
                                    df_aggr=self.df_merge,
                                    date_col=dataset['date_col'])

      if (self.debug):
        print(self.df_merge)
        print(assess_na(self.df_merge))


  def print_correlations(self):
    # Assess correlations between all data columns
    df_corr = self.df_merge.corr()

    # Identify the columns which have medium to strong correlation with target
    df_corr_cols = df_corr[df_corr[TARGET_LABEL] > 0.5]

    # Drop the target from the correlation results in case we want to use this reduced set
    #    in place of the full set
    df_corr_cols = df_corr_cols.drop(columns=[])

    # Extract just the column names
    corr_cols = df_corr_cols.index.values

    if self.debug:
      print(corr_cols)

    # Add labels
    labels = np.where(np.abs(df_corr) > 0.75, 'S',
                      np.where(np.abs(df_corr) > 0.5, 'M',
                              np.where(np.abs(df_corr) > 0.25, 'W', '')))
    # Plot the matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(df_corr, mask=np.eye(len(df_corr)), square=True,
                center=0, annot=labels, fmt = '', linewidths = .5,
                cmap='vlag', cbar_kws={'shrink':0.8});
    plt.title('Heatmap of correlation among variables', fontsize=20)


  def current_time_ms(self):
    """
    Return a numeric based on current millisecond.
    """
    return dt.now().microsecond

  def train(self):
    return self.process()

  def load_model(self, model_fullpath):
    # Warning: defaulting num labels - currently supporting multi output labels is TBD
    mf = self.get_model_factory(num_labels=1)
    print(f"## Loading model: {model_fullpath}")
    model = mf.get_saved(model_fullpath)
    print(f'Model type: {type(model)}')
    model.print_summary()
    self.model = model
    return model

  def predict(self):
    # Now, set up last step of existing data
    # and ensure we can predict past it - single shot
    print(f'## Predicting {self.LABEL_WINDOW} steps ahead')
    if (self.model is None):
      raise AssertionError('Model is not loaded; please train or load a model first')

    # It's time to set date as index and remove from dataset
    if (self.merger.DATE_COL in self.df_merge.columns):
      self.df_merge.set_index(self.merger.DATE_COL, inplace=True, drop=True)

    # Remove piecemeal date fields as well
    self.df_merge.drop(columns=['day','month','year'], inplace=True)

    NUM_FEATURES = len(self.df_merge.columns)
    print(self.df_merge)

    ## """** Extract just last section as input **"""

    df_input = self.df_merge.iloc[-self.INPUT_WINDOW:, :]
    # use existing y values to create a scaler we can use on results
    y_train = df_input[self.TARGET_LABEL]

    if self.debug:
      print(f'df_input: {df_input.shape}')

    ## """**Scale data**

    # Doing this **after** the split means that training data doesn't get unfair advantage of looking ahead into the 'future' during test & validation.

    # Create small pipeline for numerical features
    numeric_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy='mean')),
                                        ('scale', MinMaxScaler())])

    # get names of numerical features
    con_lst = df_input.select_dtypes(include='number').columns.to_list()

    # Transformer for applying Pipelines
    column_transformer = ColumnTransformer(transformers = [('number', numeric_pipeline, con_lst)])

    # Transform data features
    X_train_tx = column_transformer.fit_transform(df_input)

    # Transform labels
    label_scaler = MinMaxScaler()
    y_train_tx = label_scaler.fit_transform(y_train.values.reshape(-1, 1))

    if self.debug:
      print(f'X_train_tx {X_train_tx.shape}: {X_train_tx[0]}')
      print(f'y_train_tx {y_train_tx.shape}: {y_train_tx[0]}')

    ## """**Extract X**

    # Normally we would do this by explicitly extracting data from our df.
    # However for a time series, we're going to create many small supervised learning sets, so a set of X and y pairs.
    # We should end up with data in a shape ready for batched network input:
    # `batches X time_steps X features`
    NUM_LABELS = 1

    ## **Modeling**

    # These are the features we are going to be modeling
    COLS = list(self.df_merge.columns)

    ## """**Slice into Batches**"""

    # windower = TfWindowGenerator(input_width=self.INPUT_WINDOW,
    #                             label_width=self.LABEL_WINDOW,
    #                             shift=self.SHIFT,
    #                             batch_size=self.INPUT_WINDOW,
    #                             debug=False)
    # print(windower)

    # Build TF Dataset from arrays
    # ds = windower.get_ds_from_arrays(X_train_tx, y_train_tx)

    X_in = np.asarray(X_train_tx).astype('float32')

    ## """**Prep GPU**"""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ##"""**Build model**"""

    #model = ModelLSTMv2(window_size=self.INPUT_WINDOW, num_epochs=self.NUM_EPOCHS, debug=True)

    print(f'Using loaded model: {self.model.get_name()}')
    model = self.model

    ##"""**Train model**"""

    #model.train(X, y, NUM_FEATURES)
    print(f'## Prepping for predicting with input {X_in.shape}')
    # Ensure we match original windowed model shape
    X_in = X_in.reshape( -1, self.INPUT_WINDOW, X_in.shape[1])
    print(f'## Predicting in model with windowed input {X_in.shape}')
    batch_preds = model.predict(X_in)

    print(f'## Preds rcvd: {batch_preds.shape}')
    if (len(batch_preds.shape) > 2):
      batch_preds = batch_preds.reshape(self.LABEL_WINDOW, -1)

    # Re-Scale
    preds = label_scaler.inverse_transform(batch_preds)

    # Reduce to single array
    preds = np.squeeze(preds)
    print(f'## Preds: {preds.shape}')

    ##"""**Determine time step calulator**"""

    # Defaults to a single Day
    STEP_OFFSET = pd.DateOffset()

    if (self.STEP_FREQ == 'M'):
      STEP_OFFSET = pd.DateOffset(months=1)
    else:
      STEP_OFFSET = pd.DateOffset(years=1)

    ##"""**Show Predictions**"""

    num_predictions = len(preds)
    print(f'Num Exp. Predictions: {num_predictions}')

    # Get date frame for injecting predicted results
    # start step is data last step + 1
    last_step = df_input.index[-1]
    print(f'## Last data step: {last_step}')
    first_pred = last_step + STEP_OFFSET
    print(f'## First pred step: {first_pred}')
    last_pred = first_pred
    for i in range(num_predictions):
      last_pred = last_pred + STEP_OFFSET
    print(f'## Last pred step: {last_pred}')

    df_results = create_timeindexed_df(first_pred, last_pred, self.STEP_FREQ)
    df_results['preds'] = preds

    # Combine input and preds
    #TODO - how to show as different yet continuous marks
    df_input = df_retain(df_input, self.TARGET_LABEL)
    df_aggr = df_input.merge(df_results, how='outer', left_index=True,right_index=True, suffixes=[None,'net'])

    print(df_aggr)

    # Finally, we have to reduce to a simple index to make the graphing work nicely
    df_aggr.reset_index(inplace=True, drop=False, names='pred_dates')
    # But we need a date label col
    date_labels = df_aggr['pred_dates'].apply(lambda x: x.strftime('%Y-%m-%d')).values
    # and now we can drop it
    df_aggr.drop(columns=['pred_dates'], inplace=True)

    ##"""**Analyze results**"""

    #print(df_results)

    # Plot results - Y vs. Pred
    TICK_SPACING=6
    fig, ax = plt.subplots(figsize=(8,6), layout="constrained")
    sns.lineplot(data=df_aggr, ax=ax)
    ax.set_xticks(df_aggr.index, labels=date_labels, rotation=90)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(TICK_SPACING))
    plt.xlabel('Time steps')
    plt.ylabel('Temp in degrees C')
    plt.legend(('Recent','Predicted'))
    plt.title(f'Prediction of next {self.LABEL_WINDOW} {self.STEP_FREQ} Steps', fontsize=16)
    plt.show()

    # write pred results out
    # df_results['pred_dates'] = date_labels
    # df_results.to_csv(self.LOG_PATH + f'model-preds-{serial}.csv', index_label='index')

    # Clear axis
    ax.clear()


  def process(self):

    # It's time to set date as index and remove from dataset
    if (self.merger.DATE_COL in self.df_merge.columns):
      self.df_merge.set_index(self.merger.DATE_COL, inplace=True, drop=True)

    # Remove piecemeal date fields as well
    self.df_merge.drop(columns=['day','month','year'], inplace=True)

    NUM_FEATURES = len(self.df_merge.columns)
    print(self.df_merge)

    # Keep rows aside for post validation?
    TOTAL_ROWS = self.df_merge.shape[0]
    NUM_VALIDATION = math.floor(TOTAL_ROWS * self.VAL_RATIO)
    WORKING_ROWS = TOTAL_ROWS - NUM_VALIDATION

    # Split non-validation rows into train/test
    NUM_TEST = math.floor(WORKING_ROWS * self.TEST_RATIO)
    NUM_TRAIN = WORKING_ROWS - NUM_TEST

    print(f'Num features: {NUM_FEATURES}')
    print(f'Total rows: {TOTAL_ROWS}')
    print(f'Validation rows: {NUM_VALIDATION}')
    print(f'Train rows: {NUM_TRAIN}')
    print(f'Test rows: {NUM_TEST}')

    ## """**Split into Train/Test**"""

    df_train = self.df_merge.iloc[:NUM_TRAIN, :]
    df_val = self.df_merge.iloc[NUM_TRAIN:NUM_TRAIN+NUM_VALIDATION, :]
    df_test = self.df_merge.iloc[NUM_TRAIN+NUM_VALIDATION:, :]

    y_train = df_train[self.TARGET_LABEL]
    y_val = df_val[self.TARGET_LABEL]
    y_test = df_test[self.TARGET_LABEL]

    if self.debug:
      print(f'df_train: {df_train.shape}')
      print(f'y_train: {y_train.shape}')
      print(f'df_test: {df_test.shape}')
      print(f'y_test: {y_test.shape}')
      print(f'df_val: {df_val.shape}')
      print(f'y_val: {y_val.shape}')


    ## """**Scale data**

    # Doing this **after** the split means that training data doesn't get unfair advantage of looking ahead into the 'future' during test & validation.

    # Create small pipeline for numerical features
    numeric_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy='mean')),
                                        ('scale', MinMaxScaler())])

    # get names of numerical features
    con_lst = df_train.select_dtypes(include='number').columns.to_list()

    # Transformer for applying Pipelines
    column_transformer = ColumnTransformer(transformers = [('number', numeric_pipeline, con_lst)])

    # Transform data features
    X_train_tx = column_transformer.fit_transform(df_train)
    X_test_tx = column_transformer.transform(df_test)
    X_val_tx = column_transformer.transform(df_val)
    #X_train_tx.shape, X_test_tx.shape, X_val_tx.shape

    # Transform labels
    label_scaler = MinMaxScaler()
    y_train_tx = label_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Slice labels - we cannot predict anything inside the first INPUT_WINDOW
    # WindowGenerator does this for us now
    #y_train_tx = y_train_tx[INPUT_WINDOW:]

    if self.debug:
      print(f'X_train_tx {X_train_tx.shape}: {X_train_tx[0]}')
      print(f'y_train_tx {y_train_tx.shape}: {y_train_tx[0]}')

    ## """**Extract X and y**

    # Normally we would do this by explicitly extracting data from our df.
    # However for a time series, we're going to create many small supervised learning sets, so a set of X and y pairs.
    # We should end up with data in a shape ready for batched network input:
    # `batches X time_steps X features`
    NUM_LABELS = y_train_tx.shape[1]

    ## **Modeling**

    # These are the features we are going to be modeling
    COLS = list(self.df_merge.columns)

    ## """**Slice into Batches**"""

    windower = TfWindowGenerator(input_width=self.INPUT_WINDOW,
                                label_width=self.LABEL_WINDOW,
                                shift=self.SHIFT,
                                batch_size=self.INPUT_WINDOW,
                                debug=False)
    print(windower)

    # Build TF Dataset from arrays
    ds = windower.get_ds_from_arrays(X_train_tx, y_train_tx)

    ## """**Prep GPU**"""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ##"""**Build model**"""

    #model = ModelLSTMv2(window_size=self.INPUT_WINDOW, num_epochs=self.NUM_EPOCHS, debug=True)

    # Use factory for flexible selection
    mf = self.get_model_factory(NUM_LABELS)
    print (mf)

    print(f'Initializing model: {self.MODEL_NAME}')
    model = mf.get(self.MODEL_NAME)

    ##"""**Train model**"""

    #model.train(X, y, NUM_FEATURES)
    print(f'## Training model with {NUM_FEATURES} features and {NUM_LABELS} labels')
    model_history = model.train(dataset=ds, num_features=NUM_FEATURES)

    # Capture stat
    num_epochs = len(model_history.history['loss'])

    ##"""**Test Predictions**"""

    num_predictions = y_test.shape[0]-self.INPUT_WINDOW-self.LABEL_WINDOW
    print(f'Num Exp. Predictions: {num_predictions} == {y_test.shape[0]} - {self.INPUT_WINDOW}')

    preds = []
    pred_dates = []
    y_test_vals = []

    # Defaults to a single Day
    STEP_OFFSET = pd.DateOffset()

    if (self.STEP_FREQ == 'M'):
      STEP_OFFSET = pd.DateOffset(months=1)
    else:
      STEP_OFFSET = pd.DateOffset(years=1)

    for p in range(num_predictions):
      # Prepare inputs
      #print(f'Pred range: x_test_tx[{p}:{p+self.INPUT_WINDOW}]')
      X_pred = X_test_tx[p:p+self.INPUT_WINDOW,:].reshape(-1, self.INPUT_WINDOW, NUM_FEATURES)

      # Prepare outputs
      label_start_index = p+self.INPUT_WINDOW
      #print(f'Exp output: y_test[{label_start_index}:{label_start_index + self.LABEL_WINDOW}]')
      y_test_vals.append(y_test[label_start_index:label_start_index + self.LABEL_WINDOW])

      if (self.LABEL_WINDOW == 1):
        print(f'Pred date: {df_test.index[label_start_index]}')
      else:
        print(f'Pred dates: {df_test.index[label_start_index]} + {self.LABEL_WINDOW-1} steps')

      # Predict
      batch_preds = model.predict(X_pred)
      #print(f'## Batch step: {batch_preds.shape}')
      if (len(batch_preds.shape) > 2):
        #batch_preds = batch_preds[0]
        batch_preds = batch_preds.reshape(self.LABEL_WINDOW, -1)

      # Re-Scale
      pred_vals = label_scaler.inverse_transform(batch_preds)
      # Reduce to single array
      pred_vals = np.squeeze(pred_vals)

      if (self.LABEL_WINDOW == 1):
        preds.append(pred_vals.ravel())
        pred_dates.append(df_test.index[label_start_index])
      else:
        # Add one row per label output; we need to increment the date manually
        pred_start_date = df_test.index[label_start_index]
        step_date = pred_start_date
        for val in pred_vals.tolist():
          # add current result values
          #print(f'## val:  type: {type(val)}  value: {val}')
          preds.append(val)
          pred_dates.append(step_date)
          #print(f'## Pred: {step_date} {val}')
          # move to next step
          step_date = (step_date + STEP_OFFSET)

    df_all_results = pd.DataFrame({'preds': preds,
                                  'pred_dates':pred_dates,
                                   }, index=range(len(preds)))

    if (self.LABEL_WINDOW > 1):
      # There is probably overlap of output due to this condition
      #   Combine predicted outputs for the same dates
      # This will put date into the index
      df_results = df_all_results.groupby(['pred_dates']).mean()
    else:
      df_results = df_all_results

    # Move dates out of column and into index, if it exists
    if ('pred_dates' in df_results.columns):
      df_results.set_index('pred_dates', drop=True, inplace=True)

    # Reduce df_test to just the columns and dates necessary
    df_y = df_retain(df_test, self.TARGET_LABELS)
    df_y = df_y[df_results.index.min():df_results.index.max()]

    # And merge y values into preds
    df_results = df_y.merge(df_results, how='inner', left_index=True, right_index=True, suffixes=['', '_dft'])
    df_results = df_results.rename({self.TARGET_LABEL:'y_test'}, axis=1)

    # Finally, we have to reduce to a simple index to make the graphing work nicely
    df_results.reset_index(inplace=True, drop=False, names='pred_dates')
    # But we need a date label col
    date_labels = df_results['pred_dates'].apply(lambda x: x.strftime('%Y-%m-%d')).values
    # and now we can drop it
    df_results.drop(columns=['pred_dates'], inplace=True)

    ##"""**Analyze results**"""

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.metrics import mean_absolute_percentage_error
    import csv

    # Timestamp for result set
    serial = self.current_time_ms()

    # Save model
    model.save_model(self.LOG_PATH, serial)

    print(df_results)

    # Plot results - Y vs. Pred
    TICK_SPACING=6
    fig, ax = plt.subplots(figsize=(8,6), layout="constrained")
    sns.lineplot(data=df_results, ax=ax)
    ax.set_xticks(df_results.index, labels=date_labels, rotation=90)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(TICK_SPACING))
    plt.xlabel('Time steps')
    plt.ylabel('Temp in degrees C')
    plt.legend(('Test','Predicted'))

    # write pred results out
    df_results['pred_dates'] = date_labels
    df_results.to_csv(self.LOG_PATH + f'model-preds-{serial}.csv', index_label='index')

    # Clear axis
    ax.clear()

    ## """**Error Calculations**"""

    y_test_vals = df_results['y_test'].values
    preds = df_results['preds'].values

    # Calculate MAPE
    m = tf.keras.metrics.MeanAbsolutePercentageError()
    try:
      m.update_state(y_test_vals, preds)
    except ValueError as ve:
      print(f'ValueError calculating MAPE: {ve}')

    mse = mean_squared_error(y_test_vals, preds)
    mae = mean_absolute_error(y_test_vals, preds)
    mape = m.result().numpy()/100  # adjust Keras output to match scikit
    sk_mape = mean_absolute_percentage_error(y_test_vals, preds)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    print(f'SKMAPE: {sk_mape}')

    ## """**Journal entry**"""
    with open(self.JOURNAL_LOG, 'a') as csvfile:
      writer = csv.writer(csvfile)
      #writer.writerow(['DateTime','Serial','Model','TargetLabel','NumFeatures','InputWindow','LabelWindow','TestPct','NumEpochs','MSE','MAE','MAPE','SKMAPE','Columns'])
      writer.writerow([dt.today().strftime("%Y%m%d-%H%M"),serial,self.MODEL_NAME,self.TARGET_LABEL,NUM_FEATURES,self.INPUT_WINDOW,self.LABEL_WINDOW,self.TEST_RATIO,num_epochs,mse,mae,mape,sk_mape,COLS])

    return serial

  def get_model_factory(self, num_labels):
    if (self.model_factory is None):
      self.model_factory = ModelFactory(window_size=self.INPUT_WINDOW,label_window=self.LABEL_WINDOW,num_labels=num_labels,num_epochs=self.NUM_EPOCHS,debug=True)
    return self.model_factory