# -*- coding: utf-8 -*-
"""TeamH_Project_MainDriver.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DX-y7IzyW7Q0CiR8ipr3WSCsa3sCVwcF

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

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

# Import local source
from projectutil import *
from modelexecutor import ModelExecutor


DRIVE_PATH = "/data/projects/climate-data-model"
# Set the location of this script in GDrive
SCRIPT_PATH = DRIVE_PATH + "/py-src/"
# Root Path of the data on the cloud drive
DATA_ROOT = DRIVE_PATH + "/data/"
# Location of logged output prediction data
LOG_PATH = DATA_ROOT + "/preds/"
# Journal file
JOURNAL_LOG = SCRIPT_PATH + "cv-results.csv"

"""**Dataset Definitions**"""

# Label to predict
TARGET_LABEL = 'airPrefAvgTemp'

# Datasets
TEMP_DATA = {'filename':'GlobalTemperatures.csv',
             'feature_map':{'LandAndOceanAverageTemperature':'landSeaAvgTemp'},
             'date_col':'dt'}

AIR_TEMP_DATA = {'filename':'berkeley-earth-temperature.Global_Land_and_Ocean_Air_Temps-groomed.csv',
             'feature_map':{'GlobalAverageTemp':'airPrefAvgTemp'},
             'date_col':'date'}

SEA_TEMP_DATA = {'filename':'berkeley-earth-temperature.Global_Land_and_Ocean_Sea_Temps-groomed.csv',
             'feature_map':{'GlobalAverageTemp':'seaPrefAvgTemp'},
             'date_col':'date'}

CO2_DATA = {'filename':"atmospheric-co2.csv",
            'feature_map':{'Carbon Dioxide (ppm)':'co2', 'Seasonally Adjusted CO2 (ppm)':'co2_seas'},
            'date_map':{'Year':'year','Month':'month'}}

# Contains data older than year 1645 - provide date format for using python dates
CO2_ICE_DATA = {'filename':"co2-daily-millenia-groomed.csv",
            'feature_map':{'co2':'co2'},
                'date_col': 'date',
                'date_fmt': '%d/%m/%Y'}

GHG_HIST_DATA = {'filename':'owid-co2-data-groomed.csv',
           'feature_map':{'share_global_cumulative_luc_co2':'share_global_cumulative_luc_co2',
                          'share_global_luc_co2':'share_global_luc_co2',
                          'share_of_temperature_change_from_ghg':'share_of_temperature_change_from_ghg',
                          'temperature_change_from_co2':'temperature_change_from_co2',
                          'land_use_change_co2':'land_use_change_co2',
                          'cumulative_luc_co2':'cumulative_luc_co2'},
           'date_map':{'year':'year'}}

SEAICE_DATA = {'filename':"seaice.csv",
               'feature_map':{'     Extent':'ice_extent'},
               'date_map':{' Month':'month','Year':'year',' Day':'day'}}

WEATHER_DATA = {'filename':"finalDatasetWithRain.csv",
                'feature_map':{'air_x':'air_x','air_y':'air_y','uwnd':'uwnd'},
                'date_col':'time'}

VOLCANO_DATA = {'filename':'eruptions-conditioned.csv',
                'feature_map':{'vei':'volcanic_idx'},
                'date_map':{'start_year':'year','start_month':'month'}}

FOREST_DATA = {'filename':'WorldForestCover-Interpolated.csv',
               'feature_map':{'PctCover-Int':'pct_forest_cover'},
               'date_col':'date'}

SUNSPOT_DATA = {'filename':'sunspotnumber.csv',
               'feature_map':{'suns_spot_number':'sunspot_num'},
               'date_map':{'year':'year'}}

POLICY_DATA = {'filename':'GlobalEnvPolicies.csv',
               'feature_map':{'EventRating':'policy_rating'},
               'date_col':'date'}


#GHG_DATA = {'filename':'greenhouse_gas_inventory_data.csv',
#            'feature_map':{''},
#            'date_map':{'Year':'year'}}

"""**Run Parameters**"""

debug = False
# show_graphics = True
predict = False
NUM_LOOPS = 20

"""**Hyperparams**"""

SHIFT = 1
# Ratio of test data to train data - used for split
TEST_RATIO = 0.2
# 0..1 percent of data to use as validation
VALIDATION_RATIO = 0
# Num epochs
NUM_EPOCHS = 300

# NOTE: this is a tempting workaround to stop graphic popups, but it is deceptive.
#     This will launch an empty frame per loop
#     Rather, just feature flag plotting
# if (not show_graphics):
#   plt.ion()

# History lookback in network
#INPUT_WINDOWS = [30,48,60]
#INPUT_WINDOWS = [24,36,48]
INPUT_WINDOWS = [60]
LABEL_WINDOWS = [60]
# ALPHAS = [5e-3,1e-4,5e-4,1e-5,5e-5]
# ALPHAS = [1e-4,5e-4,5e-5]
ALPHAS = [1e-4,5e-4]

# Dynamically build a scaler from name
# SCALERS = ['StandardScaler','PowerTransformer','QuantileTransformer','RobustScaler']
SCALERS = ['StandardScaler','MinMaxScaler','RobustScaler']
#  Note that 'Normalizer' is not a scaler per se, it is essentially just a function
#    to reverse it you need to retain, and multiply by, w
  # w = np.sqrt(sum(x**2))
  # x_norm2 = x/w
  # print x_norm2

# Pair
# SCALERS = ['RobustScaler']
# MODEL_NAMES = ['Densev11']
# Pair
# SCALERS = ['StandardScaler']
# MODEL_NAMES = ['LSTMv32']
# Pair
# SCALERS = ['MinMaxScaler']
# MODEL_NAMES = ['TXERv1']

# Models to CV
# 'Densev1',
# MODEL_NAMES = ['Densev1','Densev11','TXERv1','LSTMv3','LSTMv31','LSTMv32']
# MODEL_NAMES = ['Densev1','Densev11','LSTMv3']
# MODEL_NAMES = ['LSTMv3','LSTMv31','LSTMv32']
MODEL_NAMES = ['Densev1','TXERv1','LSTMv32']


# Start/stop including data from these dates
# START_DATE =  pd.to_datetime('1950-01-01')
# END_DATE = pd.to_datetime('2023-10-01')
START_DATE =  pd.to_datetime('1850-01-01')
END_DATE = pd.to_datetime('2023-10-01')

# Base everything on this dataset
INITIAL_DATASET = AIR_TEMP_DATA

# Use case 1
# ALL_DATASETS=[[SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA]]

# Use case 2
# ALL_DATASETS=[[CO2_ICE_DATA, GHG_HIST_DATA, SEA_TEMP_DATA, AIR_TEMP_DATA],
#               [CO2_ICE_DATA, SEA_TEMP_DATA, AIR_TEMP_DATA]]
ALL_DATASETS=[[CO2_ICE_DATA, GHG_HIST_DATA, SEA_TEMP_DATA, AIR_TEMP_DATA]]

# ALL_DATASETS = [
#   [AIR_TEMP_DATA],
#  [VOLCANO_DATA,POLICY_DATA],
#  [FOREST_DATA,POLICY_DATA],
#  [SEAICE_DATA,VOLCANO_DATA,FOREST_DATA,WEATHER_DATA],
#  [SEAICE_DATA,VOLCANO_DATA,FOREST_DATA,WEATHER_DATA,CO2_DATA],
#  [SEAICE_DATA,VOLCANO_DATA,FOREST_DATA,WEATHER_DATA,CO2_DATA,SUNSPOT_DATA],
#   [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA],
#  [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA],
  # [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA],
  # [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA],
  # [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA],
  # [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA],
# ]
# ALL_DATASETS=[[SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA]]
#ALL_DATASETS=[[SEAICE_DATA]]
# ALL_DATASETS=[[SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA]]


"""# Execute Trainer"""
for n in range(NUM_LOOPS):
  for i,win in enumerate(INPUT_WINDOWS):
    for j, lab in enumerate(LABEL_WINDOWS):
      for k,model in enumerate(MODEL_NAMES):
        for m,scaler in enumerate(SCALERS):
          for o,alpha in enumerate(ALPHAS):
            for z,ds_list in enumerate(ALL_DATASETS):
              fnames = [ds['filename'] for ds in ds_list]
              print(f'============================ Executing {i+j+k+m+n+z} ===================================\n{model}-{win}/{lab}-{fnames}')

              # re-construct the model exec b/c it contains some state
              exec = ModelExecutor(data_path=DATA_ROOT, log_path=LOG_PATH, journal_log=JOURNAL_LOG, start_date=START_DATE, end_date=END_DATE,
                                  input_window=win, label_window=lab, shift=SHIFT, test_ratio=TEST_RATIO, val_ratio=VALIDATION_RATIO,
                                  num_epochs=NUM_EPOCHS, target_label=TARGET_LABEL, model_name=model, scaler=scaler, alpha=alpha, debug=True)

              exec.load_initial_dataset(INITIAL_DATASET['filename'], INITIAL_DATASET['feature_map'], date_map=None, date_col=INITIAL_DATASET['date_col'])

              exec.load_datasets(ds_list)
              #exec.print_correlations()
              exec.process()

  print(f'=========================== Completed {i+j+k+m+n+z} ===================================')