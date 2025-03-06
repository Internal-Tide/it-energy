#%%
import numpy as np
import xarray as xr
# import numba as nb

#%%
year = 2016
month = 8
days = range(1, 32)
path_uv = "/data/lfs/Tide/Data2016/UV201608/"
path_dens = "/data/lfs/Tide/result2016/dens201608/"
ds_uv_mean = xr.open_dataset("/data/lfs/Tide/Data2016/UV201608/UV-08-mean.nc")