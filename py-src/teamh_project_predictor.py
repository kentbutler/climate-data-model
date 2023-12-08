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
plt.rcParams["figure.figsize"] = (10,6)
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

INPUT_WINDOW = 60
LABEL_WINDOW = 60
MODEL_FILENAME = '20231201-2020-Densev1-568003.hdf5'
MODEL_PATH = DATA_ROOT + "preds-s24/"
MODEL_NAME = 'Densev1'
SCALER = 'MinMaxScaler'  # 'RobustScaler' 'MinMaxScaler' 'StandardScaler'
ALPHA = 1e-4
# Start/stop including data from these dates
# UC1 - global temp pred
START_DATE =  pd.to_datetime('1950-01-01')
END_DATE = pd.to_datetime('2015-12-01')
# UC2 - historical model
# START_DATE =  pd.to_datetime('1850-01-01')
# END_DATE = pd.to_datetime('2023-10-01')

"""**Dataset Definitions**"""

# Label to predict
TARGET_LABEL = 'landSeaAvgTemp'

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

"""**Run Parameters**"""
INITIAL_DATASET = TEMP_DATA
DATASETS=[SEAICE_DATA, VOLCANO_DATA, FOREST_DATA, SUNSPOT_DATA, CO2_DATA, WEATHER_DATA, POLICY_DATA]

"""# Set up and run Predictions """
exec = ModelExecutor(data_path=DATA_ROOT, log_path=LOG_PATH, journal_log=JOURNAL_LOG, start_date=START_DATE, end_date=END_DATE,
                    input_window=INPUT_WINDOW, label_window=LABEL_WINDOW, shift=1, test_ratio=0, val_ratio=0,
                    num_epochs=3, target_label=TARGET_LABEL, model_name=MODEL_NAME, scaler=SCALER, alpha=ALPHA, plot=True, debug=True)

exec.load_initial_dataset(INITIAL_DATASET['filename'], INITIAL_DATASET['feature_map'], date_map=None, date_col=INITIAL_DATASET['date_col'])

exec.load_datasets(DATASETS)

exec.load_model(f'{MODEL_PATH}{MODEL_FILENAME}')
exec.predict()
