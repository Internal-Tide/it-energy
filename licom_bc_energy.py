#%%
import xarray as xr
import numpy as np
import pandas as pd
import gsw 
# %%
class Licom_dataset:
    """This class used to calculate the baroclinic energy
       The input data is a xr.dataset
    """
    def __init__(self, xr_dataset:xr.Dataset,resolution:float):
        self.dset = xr_dataset
        self.time = self.dset.time
        self.lon = self.dset.lon
        self.lat = self.dset.lat
        self.depth = self.dset.lev1
        self.rhoc = 1029
        self.grav = 9.81
        self.resolution = resolution
    def _get_ß(self):
        """This function is used to get the surface layer
        """
        return self.dset.isel(lev1=0)