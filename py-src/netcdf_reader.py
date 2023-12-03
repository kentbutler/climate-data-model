# -*- coding: utf-8 -*-
"""netCDF_reader.ipynb
"""

# !pip install netCDF4


"""**Meteorilogical Source Data**

This data is formatted in netCDF format using SCI scientific units such as degrees Kelvin.  This is highly technical and is collected in  many different types of datasets.  The data aggregated into working datasets such as can be found at Kaggle represents a summarization of the most relevant subsets of this data as what makes data analysis practical without becoming an expert in the field.
"""

import netCDF4 as nc
import numpy as np
import pandas as pd

class NetCDFLoader:
  DRIVE_PATH = "/data/projects/climate-data-model"

  # Set the location of this script in GDrive
  SCRIPT_PATH = DRIVE_PATH + "/py-src/"

  # Root Path of the data on the cloud drive
  DATA_PATH = DRIVE_PATH + "/data/"


  def load_airpress_2m(self):
    """
    **Air Pressure, 2 meters elev.**
    """
    # Sample code to read netCDF data
    # This is air pressure recorded at 2meters above surface elevation, using long term mean averaging
    f = nc.Dataset(self.DATA_PATH+'air.2m.mon.ltm.nc')
    # f = netCDF4.Dataset(DATA_PATH+'air.mon.anom.nc')

    print(f.variables.keys()) # get all variable names
    air = f.variables['air']
    print(type(air))
    time = f.variables['time']
    print(time)

    # which produces output
    # odict_keys(['lat', 'lon', 'time', 'climatology_bounds', 'air', 'valid_yr_count'])
    # <class 'netCDF4._netCDF4.Variable'>
    # float32 air(time, lat, lon)
    #     long_name: Long Term Mean Monthly Mean of Air Temperature
    #     valid_range: [150. 400.]
    #     units: degK
    #     add_offset: 0.0
    #     scale_factor: 1.0
    #     missing_value: -9.96921e+36
    #     precision: 2
    #     least_significant_digit: 1
    #     GRIB_id: 11
    #     GRIB_name: TMP
    #     var_desc: Air temperature
    #     level_desc: 2 m
    #     statistic: Long Term Mean
    #     parent_stat: Mean
    #     dataset: NCEP Reanalysis Derived Products
    #     actual_range: [199.70786 312.07498]
    # unlimited dimensions:
    # current shape = (12, 94, 192)
    # filling on, default _FillValue of 9.969209968386869e+36 used
    #
    # <class 'netCDF4._netCDF4.Variable'>
    # float64 time(time)
    #     long_name: Time
    #     delta_t: 0000-01-00 00:00:00
    #     avg_period: 0030-00-00 00:00:00
    #     prev_avg_period: 0017-00-00 00:00:00
    #     standard_name: time
    #     axis: T
    #     units: hours since 1800-01-01 00:00:0.0
    #     climatology: climatology_bounds
    #     climo_period: 1991/01/01 - 2020/12/31
    #     actual_range: [-15769752. -15761736.]
    #     ltm_range: [1674264. 1936512.]
    #     interpreted_actual_range: 0001/01/01 00:00:00 - 0001/12/01 00:00:00
    # unlimited dimensions:
    # current shape = (12,)
    # filling on, default _FillValue of 9.969209968386869e+36 used

  def walktree(self, top):
    yield top.groups.values()
    for value in top.groups.values():
      yield from self.walktree(value)

  def load_airtemp(self):
    # **Air Temp**
    # Sample code to read netCDF data
    # This is air temp above surface elevation, using long term mean averaging
    rootgrp = nc.Dataset(self.DATA_PATH+'air.mon.anom.nc')
    print(rootgrp.dimensions)
    # {'time': <class 'netCDF4._netCDF4.Dimension'> (unlimited): name = 'time', size = 2085,
    # 'lat' : <class 'netCDF4._netCDF4.Dimension'>: name = 'lat', size = 36,
    # 'lon' : <class 'netCDF4._netCDF4.Dimension'>: name = 'lon', size = 72,
    # 'nbnds' : <class 'netCDF4._netCDF4.Dimension'>: name = 'nbnds', size = 2
    # }

    for children in self.walktree(rootgrp):
      for child in children:
        print(child)
    # <class 'netCDF4._netCDF4.Variable'>
    # <class 'netCDF4._netCDF4.Variable'>

    print(rootgrp.variables.keys()) # get all variable names
    # dict_keys(['lat', 'lon', 'time', 'time_bnds', 'air'])

    df = pd.DataFrame()

    var_list = list(rootgrp.variables.keys())
    for name in var_list:
      data = rootgrp.variables[name]
      data = np.asarray(data[:])
      print(f'var: {name}: {data.shape}')
      # df[name] = data

    # extract dates
    # times = np.asarray(rootgrp.variables['time'][:])
    times = rootgrp.variables['time']
    # print(times[1])
    dates = nc.num2date(times, units=times.units, calendar=times.calendar,
                          only_use_cftime_datetimes=False,
                          only_use_python_datetimes=True)

    lats = rootgrp.variables['lat']
    lats = np.asarray(lats[:])
    print(lats)

    longs = rootgrp.variables['lon']
    longs = np.asarray(longs[:])
    print(longs)

    air = rootgrp.variables['air']
    air = np.asarray(air[:])
    print(air.shape)
    # <class 'netCDF4._netCDF4.Variable'>

    print(lats[11])
    print(longs[1])
    print(air[1111][11][1])

    # var: air: (2085, 36, 72)
    for i,d in enumerate(air):
      # first dimension is time/date

  def load_hadsst_sea(self):
    """
    **HadSST (??)**
    """
    f = nc.Dataset(self.DATA_PATH + "HadSST.4.0.1.0_median.nc")

    #f = netCDF4.Dataset(DATA_PATH + 'air.2m.mon.ltm.nc')

    # get all variable names
    fields = f.variables.keys()
    print(fields)

    #air = f.variables['air']
    #air

    lat = f.variables['latitude']
    long = f.variables['longitude']

    time = f.variables['time']
    print(time)

    print(lat.get_dims())
    # latitudes = f.variables[lat_dim.name][:]

    print(f)

    time_dim, lat_dim, long_dim = f.variables['tos'].get_dims()

    latitudes = f.variables[lat_dim.name][:]
    print(latitudes)

    longitude = f.variables[long_dim.name][:]
    print(longitude)

    # **Load into DataFrame**

    df = pd.DataFrame()
    # from https://stackoverflow.com/questions/14035148/import-netcdf-file-to-pandas-dataframe
    time_arr = nc.num2date(time[:], time.units)

    for f in fields:
      print(f)

    air = f.variables['air']
    ser = pd.Series(air[:,0],index=time_arr)

    lat = f.variables['lat']
    print(lat)
    ser = pd.Series(lat[:],index=time_arr)

DEBUG_NETCDF = True

if DEBUG_NETCDF:
  loader = NetCDFLoader()
  loader.load_airtemp()