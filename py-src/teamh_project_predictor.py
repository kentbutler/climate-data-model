# -*- coding: utf-8 -*-
## Data606 - Capstone Project
# ```
# Group H
# Malav Patel, Kent Butler
# Prof. Unal Sokaglu
# ```
# Make predictions based on the given model.
#
import pandas as pd
from datetime import datetime as dt
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

# Import local source
from projectutil import *
from modelexecutor import ModelExecutor

debug = False
DRIVE_PATH = "/data/projects/climate-data-model/"
# Set the location of this script in GDrive
SCRIPT_PATH = DRIVE_PATH + "py-src/"
# Root Path of the data on the cloud drive
DATA_ROOT = DRIVE_PATH + "data/"
# Location of logged output prediction data
LOG_PATH = DATA_ROOT + "preds/"
# Journal file
JOURNAL_LOG = SCRIPT_PATH + "cv-results.csv"
"""**Parameters**"""

# ---- Serialized model w/ params -----
# INPUT_WINDOW = 60
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0400-Densev1-823252.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s39/"
# MODEL_NAME = 'Densev1'
# SCALER = 'MinMaxScaler'

# 685055 - NOT GREAT
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231210-2334-Densev1-685055.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s38/"
# MODEL_NAME = 'Densev1'
# SCALER = 'MinMaxScaler'

# 685055 - NOT BAD
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231210-2334-Densev1-685055.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s38/"
# MODEL_NAME = 'Densev1'
# SCALER = 'MinMaxScaler'

# 956965 - DECENT, BIG STEP UP
# INPUT_WINDOW = 60
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0347-TXERv1-956965.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s39/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 971088 - DECENT
# INPUT_WINDOW = 60
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0235-TXERv1-971088.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s39/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

#  263281 - POOR
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0719-TXERv1-263281.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s40/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'StandardScaler'

# 50120 -- FLAT
# INPUT_WINDOW = 60
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0621-TXERv1-50120.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s40/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 382025
# INPUT_WINDOW = 120
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0828-TXERv1-382025.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s40/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 451509
# INPUT_WINDOW = 60
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231211-0400-LSTMv32-451509.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s39/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 738809 -- FLAT
# INPUT_WINDOW = 120
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231212-0231-LSTMv32-738809.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s41/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 610789 -- FLAT
# INPUT_WINDOW = 120
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231212-0205-LSTMv32-610789.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s41/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 370528
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231212-0236-LSTMv32-370528.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s41/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 960250
# INPUT_WINDOW = 120
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0046-TXERv1-960250.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s42/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 237996 - best TXER
# INPUT_WINDOW = 36
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0202-TXERv1-237996.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 469862 - 2nd-best TXER
# INPUT_WINDOW = 36
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0323-TXERv1-469862.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 143389 - series best Dense
# INPUT_WINDOW = 36
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0059-Densev1-143389.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'Densev1'
# SCALER = 'MinMaxScaler'

# 358649 - wide TXER
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0103-TXERv1-358649.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 615785
# INPUT_WINDOW = 120
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0106-TXERv1-615785.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 17695
# INPUT_WINDOW = 84
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0104-LSTMv32-17695.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 643981
# INPUT_WINDOW = 36
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0058-LSTMv32-643981.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s43/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 466828
# INPUT_WINDOW = 24
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0908-TXERv1-466828.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s44/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 940023
# INPUT_WINDOW = 24
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0936-LSTMv32-940023.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s44/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 23628
# INPUT_WINDOW = 24
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-0902-Densev1-23628.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s44/"
# MODEL_NAME = 'Densev1'
# SCALER = 'MinMaxScaler'

# 963309
# INPUT_WINDOW = 24
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-1017-LSTMv32-963309.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s45/"
# MODEL_NAME = 'LSTMv32'
# SCALER = 'MinMaxScaler'

# 30193
# INPUT_WINDOW = 24
# LABEL_WINDOW = 60
# MODEL_FILENAME = '20231213-1016-TXERv1-30193.hdf5'
# MODEL_PATH = DATA_ROOT + "preds-s45/"
# MODEL_NAME = 'TXERv1'
# SCALER = 'MinMaxScaler'

# 722188
INPUT_WINDOW = 24
LABEL_WINDOW = 60
MODEL_FILENAME = '20231213-1057-TXERv1-722188.hdf5'
MODEL_PATH = DATA_ROOT + "preds-s46/"
MODEL_NAME = 'TXERv1'
SCALER = 'MinMaxScaler'

# --------------------------------------

ALPHA = 1e-4

# Start/stop including data from these dates
# UC1 - global temp pred
# START_DATE =  pd.to_datetime('1950-01-01')
# END_DATE = pd.to_datetime('2015-12-01')
# UC2 - historical model
START_DATE =  pd.to_datetime('1850-01-01')
END_DATE = pd.to_datetime('2023-10-01')

# How many Label Windows to predict ahead?  Balances this number w/ Label Windows BEFORE end of data;
#   so, plan to have at least (NUM_PREDICTION_WINDOWS*LABEL_WINDOW)+INPUT_WINDOW years of data available
NUM_PREDICTION_WINDOWS=8

"""**Dataset Definitions**"""


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

"""**Run Parameters**"""

# Label to predict
# UC1
# GRAPH_LABEL = 'landSeaAvgTemp'
# TARGET_LABELS = 'landSeaAvgTemp'
# UC3
GRAPH_LABEL = 'airPrefAvgTemp'
# TARGET_LABELS = ['airPrefAvgTemp']
TARGET_LABELS = ['airPrefAvgTemp','co2','seaPrefAvgTemp','share_global_cumulative_luc_co2','land_use_change_co2','share_of_temperature_change_from_ghg','temperature_change_from_co2']

# TARGET_LABELS = ['airPrefAvgTemp','co2']
# TARGET_LABELS = ['airPrefAvgTemp','co2','seaPrefAvgTemp']

# Base everything on this dataset
# INITIAL_DATASET = TEMP_DATA     #UC1
INITIAL_DATASET = AIR_TEMP_DATA  #UC3

# Use case 1
# DATASETS=[[SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA]]
#   [SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA]
# ]

# Use case 3
DATASETS=[CO2_ICE_DATA, GHG_HIST_DATA, SEA_TEMP_DATA, AIR_TEMP_DATA]
# DATASETS=[CO2_ICE_DATA, SEA_TEMP_DATA, AIR_TEMP_DATA]

"""# Set up and run Predictions """
exec = ModelExecutor(data_path=DATA_ROOT, log_path=LOG_PATH, journal_log=JOURNAL_LOG, start_date=START_DATE, end_date=END_DATE,
                    input_window=INPUT_WINDOW, label_window=LABEL_WINDOW, shift=1, test_ratio=0, val_ratio=0,
                    num_epochs=3, target_labels=TARGET_LABELS, graph_label=GRAPH_LABEL, model_name=MODEL_NAME, scaler=SCALER, alpha=ALPHA, plot=True, debug=True)

exec.load_initial_dataset(INITIAL_DATASET['filename'], INITIAL_DATASET['feature_map'], date_map=None, date_col=INITIAL_DATASET['date_col'])
exec.load_datasets(DATASETS)
exec.load_model(MODEL_PATH, MODEL_FILENAME)
exec.predict(num_label_windows=NUM_PREDICTION_WINDOWS)
