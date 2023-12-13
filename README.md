# climate-data-model

# Summary

This project implements an agile data modeling framework with a focus on climate data. Its purpose is to provide a flexible and relatively simple platform for testing many models, hyperparamter tuning, dataset consumption and pre-processing, and result inspection and comparison.


The class structure looks like

![alt text](https://github.com/kentbutler/climate-data-model/blob/main/ClassDiagram.jpg?raw=true)

and you can see descriptions of the CLI in the `Scripts` section.

There are also `ipynb` notebooks which can be run for doing EDA and running individual models.

# Scripts

Summary of the **Python scripts** intended to run directly from the CLI:

| Script                  | Description            |
|--------------------------|-----------------------|
| teamh_project_maindriver.py | Simple Cross-Validation script. Specify datasets, models, and hyperparameters to test with. Prepares sthe `cv-results.csv` output summary. |
|csv_result_processor_all.py  | Execute against an entire results directory; reads `results-cv.csv` and graphs summary info and the top N results based on METRIC. Set directory name directly in script before running. |
| teamh_project_predictor.py | Given the descriptor of a saved model, execute predictions using that model for the input parameters. |
|overall_results_crawler.py  | Execute against several directories and summarize model performances based on hyperparameters. Set directory names directly in script before running. |
| netcdf_reader.py | For reading and parsing of netCDF-formatted data.  This is a prototype. |
| teamh_project_main.py | Copy of the `ipynb` discovery notebook for running individual models. Not maintained. |


Summary of the **Python notebooks** available:

| Script                  | Description            |
|--------------------------|-----------------------|
| Dataset_EDA.ipynb     | Easy EDA of new datasets. Use for CSV. |
| netCDF_reader.ipynb   | Prototype EDA of netCDF data. |
| TeamH_Project_Main.ipynb | Single model executor for experimentation. |

There are also a number of other `ipynb` notebooks which were original implementations of the flexible framework, but have not been maintained. YMMV.

# Data

Due to the size of the data it is housed in Google Drive.

See this link:  https://drive.google.com/drive/folders/1uXEOk5igGhPBu_T8z5Hf63MWbDfSgHOO?usp=sharing

# Examples

Example results from this framework.

## Temperature Prediction over 40 Years using CO2 and historical AirTemp & SeaTemp datasets

Using py script `teamh_project_maindriver.py` with input data configuration:
```
AIR_TEMP_DATA = {'filename':'berkeley-earth-temperature.Global_Land_and_Ocean_Air_Temps-groomed.csv',
             'feature_map':{'GlobalAverageTemp':'airPrefAvgTemp'},
             'date_col':'date'}

SEA_TEMP_DATA = {'filename':'berkeley-earth-temperature.Global_Land_and_Ocean_Sea_Temps-groomed.csv',
             'feature_map':{'GlobalAverageTemp':'seaPrefAvgTemp'},
             'date_col':'date'}


# Contains data older than year 1645 - provide date format for using python dates
CO2_ICE_DATA = {'filename':"co2-daily-millenia-groomed.csv",
            'feature_map':{'co2':'co2'},
                'date_col': 'date',
                'date_fmt': '%d/%m/%Y'}

GHG_HIST_DATA = {'filename':'owid-co2-data-groomed.csv',
           'feature_map':{'share_global_cumulative_luc_co2':'share_global_cumulative_luc_co2',
                          'share_global_luc_co2':'share_global_luc_co2',
                          'share_of_temperature_change_from_ghg':'share_of_temperature_change_from_ghg',
                          'temperature_change_from_co2':'temperature_change_from_co2'},

           'date_map':{'year':'year'}}

```
and parameters 
```
INPUT_WINDOWS = [30,48,60]
LABEL_WINDOWS = [60]
ALPHAS = [1e-4,5e-4,5e-5]
SCALERS = ['StandardScaler','MinMaxScaler','RobustScaler','PowerTransformer','QuantileTransformer']
MODEL_NAMES = ['Densev1','Densev11','TXERv1','LSTMv3','LSTMv31','LSTMv32']

START_DATE =  pd.to_datetime('1850-01-01')
END_DATE = pd.to_datetime('2023-10-01')

ALL_DATASETS=[[CO2_ICE_DATA, SEA_TEMP_DATA, GHG_HIST_DATA, AIR_TEMP_DATA]]
INITIAL_DATASET = AIR_TEMP_DATA
GRAPH_LABEL = 'airPrefAvgTemp'
TARGET_LABELS = ['airPrefAvgTemp','co2','seaPrefAvgTemp','share_global_cumulative_luc_co2','share_global_luc_co2','share_of_temperature_change_from_ghg','temperature_change_from_co2']
```
the script will execute all combinations of input parameters, and create summarized output in `data/cv-results.csv`.

Running 
```
csv_result_processor_all.py
```
will then produce a graphical summary of results.  Model files and metadata/metrics for each indivual run is also recorded.

Observing the details of individual training results can be done in this same script by serial number, or by setting a METRIC threshold, which will locate and graph all results better than the threshold.

Locating a desired model for creation of inference can be done by running
```
teamh_project_predictor.py
```
with a configuration such as
```
INPUT_WINDOW = 60
LABEL_WINDOW = 60
MODEL_FILENAME = '20231211-0235-TXERv1-971088.hdf5'
MODEL_PATH = DATA_ROOT + "preds-s39/"
MODEL_NAME = 'TXERv1'
SCALER = 'MinMaxScaler'

```
where the `MODEL_FILENAME` references a particular trained model based on generated serial number.

See the `data/preds` folder for all output artifacts.






