#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
@Author: Pedro Herrera-Lormendez
"""

import numpy as np
import pandas as pd
import xarray as xr

from jcclass import JC_functions #Functions that help compute the CTs
from jcclass import JC_classification
from jcclass import CTs_functions
from jcclass import CTs_plots


class jc:
    """
    This computes the gridded Lamb Weather Types
    using the Jenkinson-Collison automated classification
    
    ···
    Attributes
    ----------
    filename  : str
        path and name of the mean sea level pressure dataset as netcdf
    cts       : A 2D ['lat','lon'] DataArray 
        Derived 27 gridded circulation types
    mslp      : A 2D ['lat', 'lon'] DataArray
        Mean Sea Level Pressure dataset in Pa

    *kwargs
    lat_south  : int
        Northernmost latitude value (-90 to 90)
    lat_north  : int
        Southernmost latitude value (-90 to 90)
    lon_west   : int
        Westernmost longitude value (-180 to 180)
    lon_east   : int
        Easternmost latitude value (-180 to 180)
    **kwargsglobe        
    lat_central: int
        Central latitude of plot (-90 to 90)
    lon_central: int  
        Central longitude of plot (-180 to 180)

    Methods
    -------
    classification(filename):
        Computes the 27 gridded synoptic circulation types.

    eleven_cts(cts):
        Computes the reduced eleven CTs.

    plot_cts(cts, lat_south, lat_north, lat_west, lat_east):
        Plots the reduced eleven CTs.

    plot_cts_mslp(cts, mslp, lat_south, lat_north, lat_west, lat_east)
        Plots the reduced eleven CTs and MSLP contour lines.

    plot_cts_globe(cts, mslp, lat_central, lon_central)
        Plots the reduced eleven CTs and MSLP contour lines 
        using a NearsidePerspective projection.

    """
    def __init__(self, filename):
        self.filename = filename
        
    def classification(self):
        ''' 
        Computes the 27 circulation types
        '''
        cts = JC_classification.JC_classification(self.filename)
        return (cts)
    @staticmethod    
    def eleven_cts(cts):
        '''
        Computes the reduced 11 circulation types
        '''
        cts_11 = CTs_functions.eleven_CTs(cts)
        return(cts_11)
    @staticmethod
    def plot_cts(cts, **kwargs):
        '''
        Plots the circulation types over a region. 
        Default area globe
        '''
        fig = CTs_plots.plot_CT(cts, **kwargs)
        return(fig)
    @staticmethod
    def plot_cts_mslp(cts, mslp, **kwargs):
        '''
        Plots the circulations types and the 
        contour lines of MSLP over a region.
        Default area globe
        '''
        fig = CTs_plots.plot_CT_MSLP(cts, mslp, **kwargs)
        return(fig)
    @staticmethod    
    def plot_cts_globe(cts, mslp, **kwargsglobe):
        '''
        Plots the circulation types and MSLP contour lines
        over the Globe using the NearsidePerspective projection
        '''
        fig = CTs_plots.plot_CT_MSLP_globe(cts,mslp, *args)
        return(fig)
