"""
This module contains utilities to create plots of wave conditions from wave data.
    
Author: Kelby Kramer, Massachusetts Institute of Technology and Woods Hole Oceanographic Institution, 2024

Many functions adapted from the CDIP Python toolbox:
"""

# load modules
import numpy as np
import numpy.ma as ma 
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import time
import calendar
import netCDF4 as nc4

from itertools import groupby


###################################################################################################
# TIME CONVERSION FUNCTIONS
###################################################################################################

def find_nearest(array,value):
    """
    Find the nearest value in a numpy array to a given value.
    CDIP 2024

    Arguments:
    ----------
    array : numpy array
        Array of values.
    value : float
        Value to find in the array.

    Returns:
    -----------
    array[idx] : float
        Nearest value in the array to the given value.
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# Convert to unix timestamp
def getUnixTimestamp(humanTime,dateFormat):
    """
    Convert a human readable timestamp to a unix timestamp.
    CDIP 2024

    Arguments:
    ----------
    humanTime : string
        Human readable timestamp.
    dateFormat : string
        Date format of the human readable timestamp.

    Returns:
    -----------
    unixTimestamp : int
        Unix timestamp.
    """
    unixTimestamp = int(calendar.timegm(datetime.datetime.strptime(humanTime, dateFormat).timetuple()))
    return unixTimestamp

def getHumanTimestamp(unixTimestamp, dateFormat):
    """
    Convert a unix timestamp to a human readable timestamp.
    CDIP 2024

    Arguments:
    ----------
    unixTimestamp : int
        Unix timestamp.
    dateFormat : string
        Date format of the human readable timestamp.
    
    Returns:
    -----------
    humanTimestamp : string
        Human readable timestamp.
    """
    humanTimestamp = datetime.datetime.utcfromtimestamp(int(unixTimestamp)).strftime(dateFormat)
    return humanTimestamp

###################################################################################################
# WAVE HISTORY PLOTTING FUNCTIONS
###################################################################################################

def wave_history_plt(ncTime, timeall, Hs,Tp, Dp, stn, startdate, enddate):
    """
    Creates a plot of wave conditions over a specified time period.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    ncTime : numpy array
        Time variable from netCDF file.
    timeall : numpy array
        Time variable converted to datetime stamps.
    Hs : numpy array
        Significant wave height.
    Tp : numpy array
        Peak wave period.
    Dp : numpy array
        Peak wave direction.
    stn : string
        Station name.
    startdate : string
        Start date of the time period to plot.
    enddate : string
        End date of the time period to plot.

    Returns:
    -----------
    plt : matplotlib.pyplot
        Plot of wave conditions over the specified time period.
    """

    # Create a variable of the Buoy Name and Month Name, to use in plot title
    buoytitle = "Buoy " + stn

    # Find the nearest unix timestamp to the start and end dates
    unixstart = getUnixTimestamp(startdate,"%Y-%m-%d") 
    neareststart = find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
    nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date

    unixend = getUnixTimestamp(enddate,"%Y-%m-%d")
    future = find_nearest(ncTime, unixend)  # Find the closest unix timestamp
    futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date

    # Crete figure and specify subplot orientation (3 rows, 1 column), shared x-axis, and figure size
    fig, (pHs, pTp, pDp) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))


    # Create 3 stacked subplots for three PARAMETERS (Hs, Tp, Dp)
    pHs.plot(timeall[nearIndex:futureIndex],Hs[nearIndex:futureIndex],'b')
    pTp.plot(timeall[nearIndex:futureIndex],Tp[nearIndex:futureIndex],'b')
    pDp.scatter(timeall[nearIndex:futureIndex],Dp[nearIndex:futureIndex],color='blue',s=5) # Plot Dp variable as a scatterplot, rather than line

    # Set Titles
    plt.suptitle(buoytitle, fontsize=28)
    pHs.set_title('Significant Wave Height (Hs)', fontsize=20)
    pTp.set_title('Peak Wave Period (Tp)', fontsize=20)
    pDp.set_title('Peak Wave Direction (Dp)', fontsize=20)

    # Label x-axis
    plt.xlabel('Date', fontsize=18)

    # Make a second y-axis for the Hs plot, to show values in both meters and feet
    pHs2 = pHs.twinx()

    # Set y-axis limits for each plot
    pHs.set_ylim(0,8)
    pHs2.set_ylim(0,25)
    pTp.set_ylim(0,28)
    pDp.set_ylim(0,360)

    # Label each y-axis
    pHs.set_ylabel('Hs, m', fontsize=18)
    pHs2.set_ylabel('Hs, ft', fontsize=18)
    pTp.set_ylabel('Tp, s', fontsize=18)
    pDp.set_ylabel('Dp, deg', fontsize=18)

    # Plot dashed gridlines
    pHs.grid(which='major', color='b', linestyle='--')
    pTp.grid(which='major', color='b', linestyle='--')
    pDp.grid(which='major', color='b', linestyle='--')

    return plt 

def WIS_wave_history(nc, stn, startdate, enddate):
    """
    Creates a plot of wave conditions over a specified time period.
    CDIP 2024 - KK Adapted for WIS data 2024

    Arguments:
    ----------
    nc : netCDF4 Dataset
        CDIP netCDF file.
    stn : string
        CDIP station name.
    startdate : string
        Start date of the time period to plot.
    enddate : string
        End date of the time period to plot.

    Returns:
    -----------
    plt : matplotlib.pyplot
        Plot of wave conditions over the specified time period.
    """

    # Read in variables
    ncTime = nc.variables['time'][:]
    timeall = [datetime.datetime.fromtimestamp(t) for t in ncTime] # Convert ncTime variable to datetime stamps
    Hs = nc.variables['waveHs'][:]
    Tp = nc.variables['waveTpPeak'][:]
    Dp = nc.variables['waveMeanDirection'][:]

    # Create a variable of the Buoy Name and Month Name, to use in plot title
    buoytitle = "Buoy " + stn

    # Find the nearest unix timestamp to the start and end dates
    unixstart = getUnixTimestamp(startdate,"%Y-%m-%d") 
    neareststart = find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
    nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date

    unixend = getUnixTimestamp(enddate,"%Y-%m-%d")
    future = find_nearest(ncTime, unixend)  # Find the closest unix timestamp
    futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date

    # Crete figure and specify subplot orientation (3 rows, 1 column), shared x-axis, and figure size
    fig, (pHs, pTp, pDp) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))


    # Create 3 stacked subplots for three PARAMETERS (Hs, Tp, Dp)
    pHs.plot(timeall[nearIndex:futureIndex],Hs[nearIndex:futureIndex],'b')
    pTp.plot(timeall[nearIndex:futureIndex],Tp[nearIndex:futureIndex],'b')
    pDp.scatter(timeall[nearIndex:futureIndex],Dp[nearIndex:futureIndex],color='blue',s=5) # Plot Dp variable as a scatterplot, rather than line

    # Set Titles
    plt.suptitle(buoytitle, fontsize=28)
    pHs.set_title('Significant Wave Height (Hs)', fontsize=20)
    pTp.set_title('Peak Wave Period (Tp)', fontsize=20)
    pDp.set_title('Peak Wave Direction (Dp)', fontsize=20)

    # Label x-axis
    plt.xlabel('Date', fontsize=18)

    # Make a second y-axis for the Hs plot, to show values in both meters and feet
    pHs2 = pHs.twinx()

    # Set y-axis limits for each plot
    pHs.set_ylim(0,8)
    pHs2.set_ylim(0,25)
    pTp.set_ylim(0,28)
    pDp.set_ylim(0,360)

    # Label each y-axis
    pHs.set_ylabel('Hs, m', fontsize=18)
    pHs2.set_ylabel('Hs, ft', fontsize=18)
    pTp.set_ylabel('Tp, s', fontsize=18)
    pDp.set_ylabel('Dp, deg', fontsize=18)

    # Plot dashed gridlines
    pHs.grid(which='major', color='b', linestyle='--')
    pTp.grid(which='major', color='b', linestyle='--')
    pDp.grid(which='major', color='b', linestyle='--')

    return plt 
