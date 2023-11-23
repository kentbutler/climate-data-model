# -*- coding: utf-8 -*-
"""ProjectUtil.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sP9YW9fUk5xkp7ny_TGlml4Iu2obSjwi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import Day, MonthBegin, YearBegin

def get_df_types(df, debug=False):
  # Create lists of column names by data type
  floats=[]
  ints=[]
  strings=[]
  other=[]

  for col in df.columns:
    if debug:
      print(f'## {col}:\t\t{df[col].dtype}')
    t = df[col].dtype.name
    if t.find('float') >= 0:
      floats.append(col)
    elif t.find('int') >= 0:
      ints.append(col)
    elif t.find('object') >= 0:
      strings.append(col)
    else:
      other.append(col)

  if debug:
    print(f'Types::\n\tInts: {ints}'),print(f'\tFloats: {floats}'),print(f'\tStrings: {strings}'),print(f'\tOther: {other}')

  return floats,ints,strings,other


def df_to_arrays(df, target_label, debug=False):
  """
  Given a DataFrame, convert into a 2D array of numerics.
  Target variable is returned as y.

  Returns a 2D ndarray as X, ndarray as y, and optional encoder for y
  if encoding was necessary.
  """
  target_encoder = None
  X = []
  y = []

  # Numericize non-numerics
  for alpha_col in alphas:
    if debug:
      print(f'Label encoding col: {alpha_col}')
    label_enc = LabelEncoder()
    enc_col = label_enc.fit_transform(df[alpha_col].values)
    if alpha_col == target_label:
      target_encoder = label_enc
      y.append(enc_col)
    else:
      X.append(enc_col)

  for numeric_col in numerics:
    if numeric_col == target_label:
      y.append(df[numeric_col].values)
    else:
      X.append(df[numeric_col].values)

  return np.array(X), np.array(y), target_encoder

def df_retain(df, columns, debug=False):
  """
  Given a DataFrame and a list of column names, retain only the listed columns.
  Returns a dataframe with all but the listed columns removed.
  """
  if (df is None):
    return None

  drop_cols = []
  for col in df.columns:
    if (col not in columns):
      drop_cols.append(col)

  if (debug):
    print(f'Dropping columns: {drop_cols}')

  return df.drop(columns=drop_cols)

def create_timeindexed_df(start_date, end_date, freq='1M'):
  """
  Create an empty DataFrame with index set up with dates in a regular pattern
  for the currently set start/end dates and given frequency.
  See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
  for the full frequency specification.
  Default creates an index point at every first day of the month in the given range.
  """
  if (not start_date or not end_date):
    raise AssertionError('Time range required')

  # See if we need an offset
  offset = Day(0)  # Days do not, unless you're looking to back off Time units to midnight
  if (freq[-1] == 'M'):
    offset = MonthBegin(1)
  elif (freq[-1] == 'Y'):
    offset = YearBegin(1)

  # Generate index
  dates = pd.date_range(start_date, end_date, freq=freq) - offset

  # Create empty df
  return pd.DataFrame(index=dates)

def clean_df(df, purge_suffixes=[]):
  """
  Perform following on the given df:
  * drop any column with a name ending in a given purge_suffix
  * remove surrounding parands from any column name
  * rename columns ending in ", mean" to "-Mean"
  """
  drop_list = []
  tuple_list = {}
  has_tuples = False

  for col in df.columns:
    if (isinstance(col, tuple)):
      # rename tuple to just first entry
      tuple_list[col] = col[0]
      has_tuples = True
    else:
      for s in purge_suffixes:
        if (col.endswith(s)):
          drop_list.append(col)
          break

  # Drop columns w/ suffixes from list
  df.drop(columns=drop_list, inplace=True)

  # Rename tuples
  if (has_tuples):
    # This is a workaround, b/c the regular pd.rename() and rename_axis()
    #   are unable to replace tuple names on dfs which are created via a groupby
    #   with multiple criteria.  Hence this could be expensive.
    df_new = pd.DataFrame(index=df.index)

    for col in df.columns:
      if (isinstance(col, tuple)):
        df_new[col[0]] = df[col]
      else:
        df_new[str(col)] = df[col]
    df = df_new

  return df

def diff_df_rows(df1, df2, col='index', debug=False):
  """
  Difference the given dataframes on the given column.
  If 'index' specified, applies to the indexes.
  Returns count of difference.
  """
  if (col == 'index'):
    diff1 = set(df1.index) - set(df2.index)
    if (debug):
      print (f'In df1 only: {diff1 if len(diff1) < 11 else len(diff1)}')

  else:
    diff1 = set(df1[col].values) - set(df2[col].values)
    if (debug):
      print (f'In df1 only: {diff1 if len(diff1) < 11 else len(diff1)}')

  return len(diff1)

def assess_na(df):
  """
  Given a df, return the percentage of each column that is NaN as a df.
  Return a df w/ same cols, percent values.
  """
  df_na = df.isna()
  df_stats = pd.DataFrame(columns=[df_na.columns])
  for col in df_na.columns:
    s = df_na[col].value_counts()
    # code around error accessing when keys do not exist
    t=0
    f=0
    if (True in s):
      t=s[True]
    if (False in s):
      f=s[False]

    df_stats.loc[0,col] = t/(t+f)

  return df_stats

"""---

**Unit Tests**

---
"""

PU_UNIT_TEST = False

# Unit Testing
if PU_UNIT_TEST:
  def get_test_df():
    df = pd.DataFrame({'angles': [0, 3, 4],
                      'degreesRT': [360.32, 180.31, 360.114],
                      'code':['A','B','C']},
                      index=['circle', 'triangle', 'rectangle'])
    print(f'DataFrame: \n{df}')
    return df

  print('--------------------------------------------')
  print('Case 1: get_df_types()')
  df = get_test_df()

  # Determine data types in given columns
  floats,ints,strings,other = get_df_types(df, True)

  numerics = set(floats).union(set(ints))
  alphas = set(strings).union(set(other))

  print(f'Numeric cols: {numerics}')
  print(f'Alpha cols: {alphas}')

  print('--------------------------------------------')
  print('Case 2: df_to_arrays() - alpha target')
  df = get_test_df()

  X, y, enc = df_to_arrays(df, 'code', debug=True)
  print(f'X: {X}')
  print(f'y: {y}')
  print(f'enc: {enc}')

  print('--------------------------------------------')
  print('Case 3: df_to_arrays() - alpha col, numer. target')
  df = get_test_df()

  X, y, enc = df_to_arrays(df, 'angles', debug=True)
  print(f'X: {X}')
  print(f'y: {y}')
  print(f'enc: {enc}')

  print('--------------------------------------------')
  print('Case 4: df_retain() - retain only "angles"')
  df = get_test_df()

  df = df_retain(df, 'angles', debug=True)
  print(f'df: {df.columns}')

from sre_constants import error
if PU_UNIT_TEST:
  import pandas as pd
  from datetime import datetime as dt
  import datetime

  START_DATE =  pd.to_datetime(dt.fromisoformat('1999-01-01'))
  # Stop including data after this date
  END_DATE = pd.to_datetime(dt.fromisoformat('2002-12-31'))

  print('--------------------------------------------')
  print('Case 5a: create_timeindexed_df() - no date')
  try:
    # Error expected
    df = create_timeindexed_df(START_DATE, None, freq='1M')
  except AssertionError as e:
    print (f'Caught expected error: {e}')

  print('--------------------------------------------')
  print('Case 5b: create_timeindexed_df() - 1M')
  df = create_timeindexed_df(START_DATE, END_DATE, freq='1M')
  print(df.index)

  print('--------------------------------------------')
  print('Case 5c: create_timeindexed_df() - 1D')
  df = create_timeindexed_df(START_DATE, END_DATE, freq='1D')
  print(df.index)

  print('--------------------------------------------')
  print('Case 5c: create_timeindexed_df() - 1Y')
  df = create_timeindexed_df(START_DATE, END_DATE, freq='1Y')
  print(df.index)

if PU_UNIT_TEST:
  print('--------------------------------------------')
  print('Case 6: - create_timeindexed_df')
  df = create_timeindexed_df(START_DATE, END_DATE, freq='1M')
  df['date'] = df.index
  print(df.columns)
  print(df.head())
  df['year'] = df['date'].apply(lambda x: x.year)
  df['month'] = df['date'].apply(lambda x: f'{x.month:02}')

if PU_UNIT_TEST:
  print('--------------------------------------------')
  print('Case 7a: - clean_df - mixed tuple/string series names')
  day_data = ['Mon','Tue','Wed','Thurs','Fri','Wed','Tue','Fri']
  df = pd.DataFrame({'days': day_data,
                   ('vals', 'mean'): range(len(day_data)),
                     'valdf':range(1,len(day_data)+1)})
  print(df.info())
  print('=========== cleaning ===========')
  df = clean_df(df, purge_suffixes=['df'])
  print(df.info())

if PU_UNIT_TEST:
  print('--------------------------------------------')
  print('Case 7b: - clean_df - all tuple series names')

  # Create a realistic timeseries df
  years = [2020,2020,2021,2021,2020,2021]
  months= [1,1,2,2,3,4]
  df = pd.DataFrame({'year': years,
                     'month':months,
                     'val1': [(f * 0.125) for f in range(1,len(months)+1)],
                     'val2': [(f * 0.777) for f in range(11,11+len(months))]
                     })
  df['date'] = df.apply(lambda x: pd.to_datetime(f"{int(x.month):02}/01/{int(x.year)}".format_map(x)), axis=1)
  df.set_index('date', drop=False, inplace=True)
  print(df)

  # Providing an array of element entries will create a df named entirely w/ tuples
  #    and this is not easy to clean!!
  df_net = df.groupby(['year','month'])['val1','val2'].agg(['mean'])

  print('=========== before cleaning ===========')
  print(df_net.info())
  print('=========== after cleaning ===========')
  df_net = clean_df(df_net)
  print(df_net.info())
  print(df_net.index)

if PU_UNIT_TEST:
  print('--------------------------------------------')
  print('Case 8: - diff_df_rows')

  # Create a realistic timeseries df
  years = [2020,2020,2020,2021,2021,2021]
  months= [1,2,3,4,5,6]
  df = pd.DataFrame({'year': years,
                     'month':months,
                     'val1': [(f * 0.125) for f in range(1,len(months)+1)]
                     })
  df['date'] = df.apply(lambda x: pd.to_datetime(f"{int(x.month):02}/01/{int(x.year)}".format_map(x)), axis=1)
  df.set_index('date', drop=False, inplace=True)
  print('-----------df ------------\n',df)

  # Create a realistic timeseries df
  years = [2020,2020,2020,2021,2021,2021]
  months= [1,2,3,4,5,7]
  df2 = pd.DataFrame({'year': years,
                     'month':months,
                     'val2': [(f * 0.777) for f in range(11,11+len(months))]
                     })
  df2['date'] = df2.apply(lambda x: pd.to_datetime(f"{int(x.month):02}/01/{int(x.year)}".format_map(x)), axis=1)
  df2.set_index('date', drop=False, inplace=True)
  print('-----------df2 ------------\n',df2)

if PU_UNIT_TEST:
  diff = diff_df_rows(df, df2, debug=True)
  print(f'Diff (0 means no diff): {diff}')

if PU_UNIT_TEST:
  diff = diff_df_rows(df2, df, col='date', debug=True)
  print(f'Diff (0 means no diff): {diff}')

if PU_UNIT_TEST:
  print('--------------------------------------------')
  print('Case 9: - assess na')

  # Create a realistic timeseries df
  years = [2020,2020,2020,2021,np.nan,np.nan]
  months= [np.nan,2,3,4,5,6]
  df = pd.DataFrame({'year': years,
                     'month':months,
                     'val1': [(f * 0.125) for f in range(1,len(months)+1)]
                     })
  print('-----------df ------------\n',df)
  print('-----------df_stats----------\n', assess_na(df))
