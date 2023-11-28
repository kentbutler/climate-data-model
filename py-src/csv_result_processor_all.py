# -*- coding: utf-8 -*-
"""CSV_result_processor-all.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DmEBBRyRX2sLf7lXj8wXi-LmNm343Q1X

## CSV Results Processor

Graph results of climate prediction data captured via CSV data.
"""

## ------------------------------

DRIVE_PATH = "/data/projects/climate-data-model/"

# Set the location of this script in GDrive
SCRIPT_PATH = DRIVE_PATH + "py-src/"

# Location of run data
JOURNAL_LOG = SCRIPT_PATH + "cv-results.csv"
DATA_ROOT = DRIVE_PATH + "data/preds/"

## ###############################
## Run parameters
debug = False
# Plot a certain result??  0 for all
SHOW_SERIAL = 0   # set to 0 to show just the best
# -- UNCOMMENT to load a particular result set --
# DATA_ROOT = DRIVE_PATH + "data/preds-s11/"
# JOURNAL_LOG = DATA_ROOT + "cv-results.csv"

MSE_THRESHOLD = 0.1
## ###############################

# Colors for rendering
colors = 'rbygm'

# Visualization params
METRIC = 'MSE'

GROUP_COLS = ['TargetLabel','Model','InputWindow','LabelWindow','TestPct','Columns','NumFeatures','Scaler']
TGT_LABEL = 0
WIND_SIZE = 1
TEST_PCT = 2
COLS = 3

import glob
import os
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# Attempt to offboard graphics to Qt5
#import matplotlib
#matplotlib.use('Qt5Agg')
#from display_window import DispApp
#disp = DispApp(fig)

DATE_COL = 'pred_dates'
TICK_SPACING=6

# Load CSV overall results
df = pd.read_csv(JOURNAL_LOG)
# Delete rows w/o a real serial
df = df[df['Serial'] > 10]

#--------- Plot 1 ------------
# All MSE bar graph
plt.rcParams["figure.figsize"] = [18,6]
sns.barplot(x=df['Serial'], y=df[METRIC])
#plt.plot(df[COLS])
plt.xlabel('Serial')
plt.xticks(rotation=90)
plt.ylabel(METRIC)
plt.title(f'All {METRIC} per Serial')

#--------- Plot 2 ------------
#  Pull highlights per group
df_net = df.groupby(GROUP_COLS).mean()
# get values out of index
df_net.reset_index(inplace=True)
# create X labels
df_net['label'] = df_net.apply(lambda x: f"{x['Model']}-{x.InputWindow}-{x.LabelWindow}-{x.NumFeatures}-{x.Scaler}", axis=1)
fig, ax = plt.subplots(1, 1, figsize=(6, 5), layout="constrained")
sns.barplot(x=df_net['label'], y=df_net[METRIC], ax=ax)
ax.set_xticks(df_net.index, labels=df_net.label, rotation=90)
plt.title(f'Mean {METRIC} per Param Set')

#--------- Plot 3 ------------
# Selection of serial results as line plots
# HILITE_COLS = ['Model']

for i,s in enumerate(df.index.values):
  cur_row = df.loc[s]
  serial = cur_row['Serial']
  #print(f'### serial: {serial} ###')
  if (serial <= 10):
    continue

  mse = round(float(cur_row['MSE']), 4)
  mae = round(float(cur_row['MAE']), 4)
  model = cur_row['Model']
  epochs = cur_row['NumEpochs']
  scaler = cur_row['Scaler']

  if (SHOW_SERIAL > 0):
    if (serial != SHOW_SERIAL):
      continue
  elif (mse > MSE_THRESHOLD):
    continue

  fig, ax = plt.subplots(1, 1, figsize=(11,5), layout="constrained")

  # Load Data
  df_stats = pd.read_csv(DATA_ROOT + f'model-preds-{serial}.csv')
  # Condition
  df_stats.drop(columns=['index'],inplace=True)
  df_stats.set_index('pred_dates', drop=True, inplace=True)
  df_stats.rename(columns={'preds':'Predictions'},inplace=True)
  # Plot
  sns.lineplot(data=df_stats, ax=ax, markers=['o','v'])
  # Annotate

  ax.set_xticks(df_stats.index, labels=df_stats.index, rotation=90)
  ax.xaxis.set_major_locator(plticker.MultipleLocator(TICK_SPACING))
  plt.xlabel('Time steps')
  plt.ylabel('Temp in degrees C')
  title_str = f'In/Out Window: {cur_row["InputWindow"]}/{cur_row["LabelWindow"]}   Model: {cur_row["Model"]}\n\n'
  # title_str_p = [f'{HILITE_COLS[t]}: {cur_row[HILITE_COLS[t]]}\n' for t in range(len(HILITE_COLS))]
  # title_str = title_str + title_str_p
  # title_str = ''.join(title_str)
  ax.set_title(f'({i}) Pred vs. Actual for Serial {serial}\n{title_str}')
  ax.annotate(f'MSE: {mse}   MAE: {mae} Epochs: {epochs} Scaler: {scaler}            \n{cur_row["Columns"]}',
              xy=(1,1),  # point to annotate - see xycoords for units
              xytext=(50, 10),  # offset from xy - units in textcoords
              xycoords='axes fraction',  # how coords are translated?
              textcoords='offset pixels', # 'axes fraction', 'offset pixels'
              horizontalalignment='right'
              )
  ax.texts[0].set_size(10)
plt.show()
