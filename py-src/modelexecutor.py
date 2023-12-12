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
import importlib
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

  # Count of time steps a the given scale that constitute a typical full cycle in the data
  #   Easy example is years
  CYCLE_LENGTH = {
    'Y': 1,
    'M': 12,
    'D': 365
  }
  FREQ_LABELS = {
    'D':'Day',
    'M':'Month',
    'Y':'Year'
  }

  def __init__(self, data_path, log_path, journal_log, start_date, end_date, input_window, label_window, shift, test_ratio, val_ratio, num_epochs, target_label, model_name, scaler, alpha=1e-4, plot=False, debug=False):
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
    self.TARGET_LABELS = [target_label]
    self.MODEL_NAME = model_name
    self.SCALER_NAME = scaler
    self.ALPHA = alpha
    self.PLOT = plot

    # Device to run on
    self.run_on_device =  'cpu' # 'cuda'
    # Other internal state
    self.plot_ds = False
    self.model_factory = None
    self.model = None
    self.column_transformer = None

  def load_initial_dataset(self, ds_name, feature_map, date_map=None, date_col=None):
    # Declare a merger compatible with our source data and our target dataset we want to merge into
    self.merger = Dataset_Merger(data_path=self.DATA_PATH,
                            start_date=self.START_DATE, end_date=self.END_DATE,
                            plot=self.plot_ds, debug=self.debug)

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
      # Resolve date format - usually this will be empty
      date_fmt = None
      if ('date_fmt' in dataset):
        date_fmt = dataset['date_fmt']

      if ('date_map' in dataset):
        self.df_merge = self.merger.merge_dataset(dataset['filename'],
                                        feature_map=dataset['feature_map'],
                                        df_aggr=self.df_merge,
                                        date_map=dataset['date_map'])
      else:
        self.df_merge = self.merger.merge_dataset(dataset['filename'],
                                    feature_map=dataset['feature_map'],
                                    df_aggr=self.df_merge,
                                    date_col=dataset['date_col'],
                                    date_fmt=date_fmt)

      if (self.debug):
        print(self.df_merge)
        print(assess_na(self.df_merge))


  def print_correlations(self):
    # Assess correlations between all data columns
    df_corr = self.df_merge.corr()

    # Identify the columns which have medium to strong correlation with target
    df_corr_cols = df_corr[df_corr[self.TARGET_LABEL] > 0.5]

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

  def load_model(self, path, filename):
    # Warning: defaulting num labels - currently supporting multi output labels is TBD
    mf = self.get_model_factory(num_labels=1)
    model_fullpath = f'{path}{"" if path[-1]=="/" else "/"}{filename}'
    # Save model name for later graphing
    self.MODEL_FILENAME = filename
    print(f"## Loading model: {model_fullpath}")
    model = mf.get_saved(model_fullpath)
    print(f'Model type: {type(model)}')
    model.print_summary()
    self.model = model
    return model

  def get_step_offset(self):
    """
    Converts self's STEP_FREQ, which is D/M/Y, into a pd DateOffset object
    useful for date calculations.
    :return: pd.DateOffset
    """
    # Defaults to a single Day
    offset = pd.DateOffset()
    if (self.STEP_FREQ == 'M'):
      offset = pd.DateOffset(months=1)
    elif (self.STEP_FREQ == 'Y'):
      offset = pd.DateOffset(years=1)

    return offset

  def init_date_cols(self, df):

    if (self.merger.DATE_COL in df.columns):
      # Move date into index
      df.set_index(self.merger.DATE_COL, inplace=True, drop=True)
      # df.rename(columns={self.merger.DATE_COL:'pred_dates'}, inplace=True)

    df.drop(columns=['day','month','year'], inplace=True)


  def predict(self, num_label_windows=1):

    print(f'## Predicting {self.LABEL_WINDOW} steps ahead')
    if (self.model is None):
      raise AssertionError('Model is not loaded; please train or load a model first')

    # --- Trim dataset ---
    # We only need enough data to allow for:
    #     N label windows from the end PLUS enough space for the input window,
    #        where N is the number of windows to predict
    df_input = self.df_merge

    # Ensure inputs are sort chronologically, to ensure the index math
    df_input.sort_index(inplace=True)

    # Start/end of input data block
    start_idx = df_input.shape[0] - ((num_label_windows * self.LABEL_WINDOW) + self.INPUT_WINDOW)
    input_end_idx = start_idx + self.INPUT_WINDOW - 1

    # ...and, just truncate from here
    df_input = df_input.iloc[start_idx: , :]

    # ...and, that resets our starting point - avoid confusion and mistakes
    start_idx = 0

    # first label start position, so we know how to extract our Y values
    label_start_idx = start_idx + self.INPUT_WINDOW

    self.init_date_cols(df_input)

    # How we will count between timesteps
    STEP_OFFSET = self.get_step_offset()

    # --- Get stats ---

    # how many actual values will we predict that we'll have Y data for
    num_labels = (num_label_windows * self.LABEL_WINDOW)
    # we're predicting num_label_windows AFTER end of dataset; add the SAME amount BEFORE end of dataset;
    #    this allows for a balanced output graph (visual)
    num_predictions = num_label_windows * 2
    num_features = len(df_input.columns)

    print(f"## df_input:\n{df_input}")

    if (label_start_idx >= df_input.shape[0]):
      # we are predicting past the end of the known data -- set start date as first step beyond!!
      pred_start_date = (df_input.index[-1] + STEP_OFFSET)
    else:
      # pull first prediction start date from data
      pred_start_date = df_input.index[label_start_idx]

    print(f'### start_idx: {start_idx}')
    print(f'### label_start_idx: {label_start_idx}')
    print(f'### pred_start_date: {pred_start_date}')
    print(f'### num_predictions: {num_predictions}')

    # save these starting points
    first_label_date = pred_start_date
    first_label_index= label_start_idx

    ## --- Scale ---
    y_vals = df_input[self.TARGET_LABEL]
    X_tx, y_tx = self.scale(df_input, y_vals)

    #NUM_LABELS = 1
    NUM_LABELS = y_tx.shape[1]
    COLS = list(df_input.columns)

    if self.debug:
      print(f'Num features: {len(COLS)}')
      print(f'Num labels: {NUM_LABELS}')
      print(f'X_tx:: {X_tx.shape}: {X_tx[0]}')
      print(f'y_tx:: {y_tx.shape}: {y_tx[0]}')

    # --- Prepare to Run Model ---
    print(f'Using loaded model: {self.model.get_name()}')
    model = self.model

    # Extract X input values as np array
    X_input = np.asarray(X_tx).astype('float32')

    # last index & date of X -- beyond this is regression territory
    last_data_idx = len(X_input) - 1
    last_data_date = df_input.index[-1]
    print(f'## last_data_idx: {last_data_idx}')
    print(f'## last_data_date: {last_data_date}')

    # --- Predict ---
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(f'## Begin predicting X_input: {X_input.shape}')
    all_preds = []

    # ======== Start Prediction loop ========
    for p in range(num_predictions):
      preds = []
      pred_dates = []

      # --- Boundary checks ---
      # 1. Does our block end align w/ the end of the dataset?  If it goes over, we've done something wrong.
      if (input_end_idx > last_data_idx):
        # Some portion of the prediction output is beyond the "have data" line
        #   set flag so we know that we are in prediction-only space!  from here we need to start rolling
        #   the data back into X_input - i.e, autoregressing
        print(f'WARN: Input block out of alignment - ending on: {start_idx+self.INPUT_WINDOW} when last idx is {last_data_idx}')

      # Extract just the input block we need for this iteration
      #   NOTE that if we are autoregressing, we should have the data we need available from the previous iter
      print(f'## Slicing X_input: {start_idx}:{start_idx+self.INPUT_WINDOW}')
      X_pred = X_input[start_idx:start_idx+self.INPUT_WINDOW,:].reshape(-1, self.INPUT_WINDOW, num_features)

      print(f'### X_pred: {X_pred.shape}')
      # print(f'### X_pred: {X_pred.shape}\n{X_pred}')

      # Predict
      batch_preds = model.predict(X_pred)
      print(f'## batch_preds: {batch_preds.shape}')
      if (len(batch_preds.shape) > 2):
        batch_preds = batch_preds.reshape(self.LABEL_WINDOW, -1)
        print(f'## batch_preds reshaped: {batch_preds.shape}')

      # Re-Scale
      pred_vals = self.label_scaler.inverse_transform(batch_preds)
      # Reduce to single array
      pred_vals = np.squeeze(pred_vals)
      # print(f'## pred_vals: {pred_vals.shape}  type: {type(pred_vals)}')

      if (self.LABEL_WINDOW == 1):
        preds.append(pred_vals.ravel())
        all_preds.append(pred_vals.ravel())
        pred_dates.append(pred_start_date)
        pred_start_date = pred_start_date + STEP_OFFSET
        print(f'### NEXT pred_start_date: {pred_start_date}')
      else:
        # Add one row per label output; we need to increment the date manually
        for val in pred_vals.tolist():
          # add current result values
          print(f'## pred value: {val}')
          preds.append(val)
          all_preds.append(val)
          pred_dates.append(pred_start_date)
          print(f'## ADDING pred_start_date: {pred_start_date}')
          # capture last recorded prediction date
          last_label_date = pred_start_date
          # move to next step
          pred_start_date = (pred_start_date + STEP_OFFSET)

      # ---- Reset Pointers -----
      # input shifts forward same as label windows...because, we want our label windows to be contiguous
      start_idx = start_idx + self.LABEL_WINDOW
      input_end_idx = start_idx+self.INPUT_WINDOW - 1
      # next label starting from PREVIOUS label window start....just move forward another label window
      label_start_idx = label_start_idx + self.LABEL_WINDOW


      print(f'## NEXT input_start_idx(start_idx): {start_idx}')
      print(f'## NEXT input_end_idx: {input_end_idx}')
      print(f'## NEXT label_start_idx: {label_start_idx}')
      print(f'## NEXT label_start_date(pred_start_date): {pred_start_date}')

      print(f'## last_data_idx: {last_data_idx}')
      print(f'## last_data_date: {last_data_date}')

      # If any portion of the INPUT block is over the line, we're auto-regressing
      #    Need to add some input data for the next iteration
      print(f'## IS {start_idx} + {self.INPUT_WINDOW} - 1 ({start_idx + self.INPUT_WINDOW - 1}) > {last_data_idx}')
      if (input_end_idx > last_data_idx):
        # Capture predicted values as input to the next iteration
        #   NOTE!!! We shouldn't have to do this -- we should be predicting ALL FEATURES
        # SOL: since right now we're not predicting multiple labels....
        #    WORKAROUND - just replicate last known datapoints, and lay prediction in aside of those
        #           use STEP_FREQ to find a suitable input window
        #           want start to be same
        if (pred_vals.shape[-1] != X_pred.shape[-1]):
          # features input != features predicted
          #   we need to fake it - copy all input features to output (but NOT THE SAME SIZE) (what???)
          #    roll the prediction start_date back to align with the new start index
          # we really need to predict from the INPUT side of things - so go back before the label window
          missing_date_start = pred_start_date - (self.LABEL_WINDOW * STEP_OFFSET)
          # cur_step = missing_date_start

          if (missing_date_start < last_data_date):
            # Hey! we actually have this data; move the pointer forward, to empty slot after last data point
            missing_date_start = last_label_date + STEP_OFFSET

          # NOTE that the missing input chunk will always be a label width, due to how the labels are aligned and contiguous
          missing_data_len = self.LABEL_WINDOW
          print(f'## missing_date_start: {missing_date_start}')
          print(f'## missing_data_len: {missing_data_len}')

          cur_start_idx = start_idx
          found = False
          print(f'Going back in time to find data to fill from missing start date: {missing_date_start}')
          while (not found):
            cur_start_idx = cur_start_idx - self.CYCLE_LENGTH[self.STEP_FREQ]
            print(f'Trying cur_start_idx: {cur_start_idx}')
            # did we go back far enough to fit in a label-width of data?
            if ((start_idx - cur_start_idx) >= missing_data_len):
              found = True

          cur_end_idx = cur_start_idx + missing_data_len
          print(f'Pulling data from {cur_start_idx} to {cur_end_idx}')
          df_extract = df_input.iloc[cur_start_idx:cur_end_idx,:]
          # print(f'## df_extract: {df_extract.shape}')
          # print(f'## df_extract last: {df_extract.index[-1]}')

          # Create the proper date index for the chunk we want to add -
          #   and inject into df_input for later selection
          missing_date_end = missing_date_start + ((self.LABEL_WINDOW-1) * STEP_OFFSET)
          date_index = pd.date_range(missing_date_start, missing_date_end, freq='MS', normalize=True)
          # print(f'## date_index: {date_index}')
          df_extract['newidx'] = date_index
          df_extract.set_index('newidx', drop=True, inplace=True)
          fdate = df_extract.index[0]

          # Now, update target values w/ the predicted, for better AR
          for i,date in enumerate(pred_dates):
            if (date > last_data_date):
              # This is an actual future prediction -- we DO want to overwrite the df_input
              #   temp w/ our prediction, so we can autoregress better
              df_extract.loc[date.normalize()][self.TARGET_LABEL] = preds[i]

          # Add to X_input
          # print(f'Extract and scale data from df_extract')
          X_inject,_ = self.scale(df_extract)
          print(f'## X_inject: {X_inject.shape}')
          # print(f'## X_inject:\n{X_inject}')
          # print(f'Injecting data into X_input')
          X_input = np.concatenate((X_input, X_inject))
          print(f'## X_input: {X_input.shape}')

          # Add to df_input
          # print(f'## df_extract: {df_extract.shape}')
          # print(f'## df_input BEFORE: {df_input.shape}')
          print(f'## To df_input ending at {df_input.index[-1]} concat df_extract starting at {df_extract.index[0]}')
          df_input = pd.concat([df_input,df_extract])
          # print(f'## df_input AFTER: {df_input.shape}')
          print(f'## df_input AFTER:\n{df_input}')

          # FINALLY - adjust last data index
          last_data_idx = last_data_idx + X_inject.shape[0]
          print(f'## last_data_idx: {last_data_idx}')


    ## --- End of Prediction loop ---


    ##"""**Show Predictions**"""

    num_predictions = len(all_preds)
    print(f'Num Exp. Predictions: {num_predictions}')
    print(f'First pred: {first_label_date}')
    print(f'Last pred: {last_label_date}')

    # Create timestamped container to contain all predictions
    df_results = create_timeindexed_df(first_label_date, last_label_date+STEP_OFFSET, self.STEP_FREQ)
    df_results['preds'] = all_preds

    # Remove input data that won't have predicted labels
    df_input = df_input.iloc[first_label_index:,:]

    # Combine input and preds
    df_input = df_retain(df_input, self.TARGET_LABEL)
    print(f'## df_input PRE_SLICE: {df_input}')
    df_input = df_input.loc[first_label_date:last_label_date,:]

    # Blank out all post-data "known temps" to ensure preds are graphed alone - it's all prediction at that point
    #TODO comment this out while testing the injection of pred values, s.t. we get "better" AR inputs
    df_input.loc[last_data_date+STEP_OFFSET:, self.TARGET_LABEL] = np.nan

    print(f'## MERGING ##\n## df_input.index: {df_input.index}')
    print(f'## df_results.index: {df_results.index}')
    df_aggr = df_input.merge(df_results, how='inner', left_index=True,right_index=True, suffixes=['net',None])

    print(f'## df_aggr: {df_aggr}')
    # Make a backup copy of this df, for other graphing
    df_results = df_aggr.copy()

    # Finally, we have to reduce to a simple index to make the graphing work nicely
    df_aggr.reset_index(inplace=True, drop=False, names='pred_dates')
    # But we need a date label col
    date_labels = df_aggr['pred_dates'].apply(lambda x: x.strftime('%Y-%m-%d')).values
    # and now we can drop it
    df_aggr.drop(columns=['pred_dates'], inplace=True)

    # --- Plot 1 ---

    # Plot results - Y vs. Pred
    TICK_SPACING=6
    width = 10 + (num_label_windows * 2)
    fig, ax = plt.subplots(figsize=(width,6), layout="constrained")
    sns.lineplot(data=df_aggr, ax=ax)
    ax.set_xticks(df_aggr.index, labels=date_labels, rotation=90)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(TICK_SPACING))
    plt.xlabel('Time steps')
    plt.ylabel('Temp in degrees C')
    title = f'Prediction of Next {num_label_windows * self.LABEL_WINDOW} {self.freq_to_english(self.STEP_FREQ)} Timesteps'
    ax.set_title(title, fontsize=16, pad=20)
    ax.annotate(f'Generated from {self.MODEL_NAME} model {self.MODEL_FILENAME}                      ',
                xy=(1, 1),  # point to annotate - see xycoords for units
                xytext=(-50, 10),  # offset from xy - units in textcoords
                xycoords='axes fraction',  # how coords are translated?
                textcoords='offset pixels',  # 'axes fraction', 'offset pixels'
                horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=-0.07', fc='none', ec='none')
                )
    ax.texts[0].set_size(10)
    plt.show()

    ax.clear()

    # --- Plot 2 ---
    # Avg Temp box-plot

    # Before post-data predictions, all temps reflect actual measurements
    # We will graph from a single column, going as far back as possible



    # Move date out of index b/c that will impede the upcoming apply
    df_results.reset_index(inplace=True, drop=False, names='date')

    print(f'## df_results BEFORE FILL:\n{df_results}')
    # Fill in missing temp values from predictions
    df_results[self.TARGET_LABEL] = df_results.apply(lambda x: x.preds if np.isnan(x.airPrefAvgTemp) else x.airPrefAvgTemp, axis=1)
    print(f'## df_results AFTER FILL:\n{df_results}')

    df_results['Decade'] = [round(dt.year, -1) for dt in df_results['date']]
    df_results.drop(columns=['date','preds'], inplace=True)
    df_results.rename({self.TARGET_LABEL:'Mean Degrees Celsius'}, axis=1, inplace=True)
    print(f'## df_results AFTER CLEANUP:\n{df_results}')

    fig, ax = plt.subplots(figsize=(10,6), layout="constrained")
    sns.boxplot(data=df_results, x="Decade", y='Mean Degrees Celsius', ax=ax)
    title = f'Global Mean Temp - the Next {num_label_windows * self.LABEL_WINDOW} {self.freq_to_english(self.STEP_FREQ)}s by Decade'
    ax.set_title(title, fontsize=16, pad=20)
    # ax.axvspan(5.5, 7.5, alpha=0.2)  # add shading
    plt.show()

    ax.clear()

    # --- Plot 3 ---
    # Avg Temp box-plot going ALL the way back

    # Get original data; truncate at start of predictions
    df_all = df_retain(self.df_merge, [self.TARGET_LABEL,self.merger.DATE_COL])
    df_all.rename({self.merger.DATE_COL:'date'}, axis=1, inplace=True)
    df_all['Decade'] = [round(dt.year, -1) for dt in df_all['date']]
    df_all.drop(columns=['date'], inplace=True)
    df_all.rename({self.TARGET_LABEL:'Mean Degrees Celsius'}, axis=1, inplace=True)

    # truncate existing data
    df_all = df_all[df_all['Decade'] < 1990]
    print(f'## df_all AFTER TRUNC:\n{df_all}')

    df_graph = pd.concat([df_all,df_results])

    fig, ax = plt.subplots(figsize=(10,6), layout="constrained")
    sns.boxplot(data=df_graph, x="Decade", y='Mean Degrees Celsius', ax=ax)
    title = f'Global Mean Temp - the Next {num_label_windows * self.LABEL_WINDOW} {self.freq_to_english(self.STEP_FREQ)}s by Decade'
    ax.set_title(title, fontsize=16, pad=20)
    # ax.axvspan(5.5, 7.5, alpha=0.2)  # add shading
    plt.show()

    ax.clear()

    # --- Plot 4 ---
    # Trailing temp band
    print(f'## df_graph :\n{df_graph}')

    # g = sns.lmplot(x="Decade", y="Mean Degrees Celsius", col="Mean Degrees Celsius", hue="Mean Degrees Celsius", data=df_all,
    #                 y_jitter=.02, logistic=True, truncate=False)
    # g.set(xlim=(0, 80), ylim=(-.05, 1.05))
    # sns.regplot(data=df_graph, x="Decade", y="Mean Degrees Celsius")
    # plt.show()
    # sns.lmplot(data=df_graph, x="Decade", y="Mean Degrees Celsius")
    # plt.show()

    g = sns.relplot(
      data=df_graph, x="Decade", y="Mean Degrees Celsius",  #col="year", hue="year",
      kind="line", palette="crest", linewidth=6, zorder=1,
      height=6, aspect=1.5, legend=True
    )
    # Iterate over each subplot to customize further
    for year, ax in g.axes_dict.items():
      # Add the title as an annotation within the plot
      ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")
      # Plot every year's time series in the background
      sns.lineplot(
        data=df_graph, x="Decade", y="Mean Degrees Celsius",  # col="year", hue="year",
        units='year', estimator=None, color=".7", linewidth=5, ax=ax
      )
    ax.set_xticks(ax.get_xticks()[::10])

    g.tight_layout()
    plt.show()



  def freq_to_english(self, step):
    return self.FREQ_LABELS[step]

  def scale(self, df, y_vals=None):

    if (self.column_transformer is None):
      # Set up global dataset transformers -- for this reason we need the Y data as well!!
      if (y_vals is None):
        print('WARN: scale() has no Y data, cannot set up Y scaler')
      # Dynamically build a scaler from name
      #TODO: separate out YeoJohnson into its own local class; but have to rework this a bit
      module = importlib.import_module('sklearn.preprocessing')
      ScalerClass = getattr(module, self.SCALER_NAME)
      num_scaler = ScalerClass()
      print(f'## Scaler type: {type(num_scaler)}')

      # Create small pipeline for numerical features
      numeric_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy='mean')),
                                          ('scale', num_scaler)])
      # Create Transformer from pipeline
      con_lst = df.select_dtypes(include='number').columns.to_list()
      self.column_transformer = ColumnTransformer(transformers = [('number', numeric_pipeline, con_lst)])
      self.column_transformer.fit(df)
      # Construct Y/label scaler
      self.label_scaler = ScalerClass()
      self.label_scaler.fit(y_vals.values.reshape(-1, 1))

    X_tx = self.column_transformer.transform(df)
    y_tx = None
    if (y_vals is not None):
      y_tx = self.label_scaler.transform(y_vals.values.reshape(-1, 1))
    return X_tx, y_tx

  def train(self):

    self.init_date_cols(self.df_merge)

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
    if (self.VAL_RATIO > 0):
      df_val = self.df_merge.iloc[NUM_TRAIN:NUM_TRAIN+NUM_VALIDATION, :]
    df_test = self.df_merge.iloc[NUM_TRAIN+NUM_VALIDATION:, :]

    y_train = df_train[self.TARGET_LABEL]
    if (self.VAL_RATIO > 0):
      y_val = df_val[self.TARGET_LABEL]
    y_test = df_test[self.TARGET_LABEL]

    if self.debug:
      print(f'df_train: {df_train.shape}')
      print(f'y_train: {y_train.shape}')
      print(f'df_test: {df_test.shape}')
      print(f'y_test: {y_test.shape}')
      if (self.VAL_RATIO > 0):
        print(f'df_val: {df_val.shape}')
        print(f'y_val: {y_val.shape}')


    ## """**Scale data**

    X_train_tx, y_train_tx = self.scale(df_train, y_train)
    X_test_tx,_ = self.scale(df_test)
    if (self.VAL_RATIO > 0):
      X_val_tx,_ = self.scale(df_val)

    NUM_LABELS = y_train_tx.shape[1]
    COLS = list(self.df_merge.columns)

    if self.debug:
      print(f'Num features: {len(COLS)}')
      print(f'Num labels: {NUM_LABELS}')
      print(f'X_train_tx {X_train_tx.shape}: {X_train_tx[0]}')
      print(f'y_train_tx {y_train_tx.shape}: {y_train_tx[0]}')

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

    # Use factory for flexible selection
    mf = self.get_model_factory(NUM_LABELS)
    print (mf)

    print(f'Initializing model: {self.MODEL_NAME}')
    model = mf.get(self.MODEL_NAME)

    ##"""**Train model**"""

    print(f'## Training model with {NUM_FEATURES} features and {NUM_LABELS} labels')
    model_history = model.train(dataset=ds, num_features=NUM_FEATURES)

    # Capture stat
    num_epochs = len(model_history.history['loss'])

    ##"""**Test Predictions**"""

    num_predictions = max(y_test.shape[0]-self.INPUT_WINDOW-self.LABEL_WINDOW, 1)
    print(f'Num Exp. Predictions: {num_predictions} == {y_test.shape[0]}-{self.INPUT_WINDOW}-{self.LABEL_WINDOW}')

    preds = []
    pred_dates = []
    y_test_vals = []

    # How we will count between timesteps
    STEP_OFFSET = self.get_step_offset()

    for p in range(num_predictions):
      # Prepare inputs
      #print(f'Pred range: x_test_tx[{p}:{p+self.INPUT_WINDOW}]')
      X_pred = X_test_tx[p:p+self.INPUT_WINDOW, :].reshape(-1, self.INPUT_WINDOW, NUM_FEATURES)

      # Prepare outputs
      label_start_idx = p+self.INPUT_WINDOW
      #print(f'Exp output: y_test[{label_start_idx}:{label_start_idx + self.LABEL_WINDOW}]')
      y_test_vals.append(y_test[label_start_idx:label_start_idx+self.LABEL_WINDOW])

      if (self.LABEL_WINDOW == 1):
        print(f'Pred date: {df_test.index[label_start_idx]}')
      else:
        print(f'Pred dates: {df_test.index[label_start_idx]} + {self.LABEL_WINDOW-1} steps')

      # Predict
      batch_preds = model.predict(X_pred)
      #print(f'## Batch step: {batch_preds.shape}')
      if (len(batch_preds.shape) > 2):
        #batch_preds = batch_preds[0]
        batch_preds = batch_preds.reshape(self.LABEL_WINDOW, -1)

      # Re-Scale
      pred_vals = self.label_scaler.inverse_transform(batch_preds)
      # Reduce to single array
      pred_vals = np.squeeze(pred_vals)

      if (self.LABEL_WINDOW == 1):
        preds.append(pred_vals.ravel())
        pred_dates.append(df_test.index[label_start_idx])
      else:
        # Add one row per label output; we need to increment the date manually
        pred_start_date = df_test.index[label_start_idx]
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
    rmse = np.sqrt(mse) if (mse > 0) else 0
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
      #writer.writerow(['DateTime','Serial','Model','TargetLabel','NumFeatures','InputWindow','LabelWindow','Scaler','Alpha','TestPct','NumEpochs','RMSE','MSE','MAE','MAPE','SKMAPE','Columns'])
      writer.writerow([dt.today().strftime("%Y%m%d-%H%M"),serial,self.MODEL_NAME,self.TARGET_LABEL,NUM_FEATURES,self.INPUT_WINDOW,self.LABEL_WINDOW,self.SCALER_NAME,self.ALPHA,self.TEST_RATIO,num_epochs,rmse,mse,mae,mape,sk_mape,COLS])

    return serial

  def get_model_factory(self, num_labels):
    if (self.model_factory is None):
      self.model_factory = ModelFactory(window_size=self.INPUT_WINDOW,label_window=self.LABEL_WINDOW,num_labels=num_labels,num_epochs=self.NUM_EPOCHS,alpha=self.ALPHA,debug=True)
    return self.model_factory