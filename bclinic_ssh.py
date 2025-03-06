#%%
import numpy as np
import xarray as xr
from numba import njit

#%%
year = 2016
month = 8
days = range(1, 32)
# traverse all the days in the month
path_uv = "/data/lfs/Tide/Data2016/UV201608/"
path_dens = "/data/lfs/Tide/result2016/dens201608/"
ds_uv_mean = xr.open_dataset("/data/lfs/Tide/Data2016/UV201608/UV-08-mean.nc")
ds_dens_mean = xr.open_dataset("/data/lfs/Tide/result2016/dens201608/dens-08-mean.nc")
#%%
def get_file_name(type,year, month, day):
    if month < 10:
        month_str = "0" + str(month)
    else:
        month_str = str(month)
    if day < 10:
        day_str = "0" + str(day)
    else:
        day_str = str(day)
    file_name = type + "-" + str(year) + month_str + day_str + ".nc"
    return file_name
def 