import sys

import warnings
warnings.filterwarnings('ignore')

import holoviews as hv
import xarray as xr
import pandas as pd
import numpy as np
import nptdms
from nptdms import TdmsFile
import colorcet as cc
from scipy.signal import savgol_filter
from colorcet.plotting import swatch, swatches
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh')
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import holoviews.operation.datashader as hd
from holoviews.plotting.util import process_cmap
from scipy.optimize import curve_fit
from scipy.misc import derivative
from scipy.stats import chisquare
from functools import partial
from multiprocessing import Pool
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.stats import poisson
from scipy.stats import chisquare
from scipy.stats import ks_2samp
from scipy.stats import expon
import random
import pickle

import TimeSeries_20230111
from TimeSeries_20230111 import Histogram3D,ForceExtHeatmap
from TimeSeries_20230111 import ForceRampHandler
from TimeSeries_20230111 import TimeSeriesLoader as tsl



import warnings
warnings.filterwarnings('ignore')

import importlib



FileList = [] # put hdf file name inside the list




for FN in FileList:
    # Import data
    data_df = pd.read_hdf(f'{FN}.hdf')

    ywave = data_df['ywavecut']
    forcewave = data_df['forcewavecut']
    ywavesmooth = data_df['ywavesmooth']
    forcewavesmooth = data_df['forcewavesmooth']

    maxima, _ = find_peaks(forcewave, height=3e-11, distance=2.5e5) # If intervals between ramps changed, distance subject to change
    n_cyc = len(maxima)
    
    ramps = ForceRampHandler(forcewave, ywave, n_cyc, smooth = True, auto = False, savgol_coeff = [101,3])
    
    min_force = 2

    def ZeroData(halfcycle, min_force):
        ext_zeroed = halfcycle.values - np.min(halfcycle.values)
        force_zeroed = halfcycle['force'].values - min_force
        return ext_zeroed, force_zeroed

    def UnzeroData(segments, halfcycle, min_force):
        for i in range(len(segments)):
            segments[i] = segments[i].assign_coords(force=(segments[i].force + min_force))
            segments[i].values = segments[i].values + np.min(halfcycle.values)
            #segments[i] = segments[i].assign_coords(extension=(segments[i].extension + np.min(halfcycle.extension.values)))
            #segments[i].values = segments[i].values + min_force
        return segments
    
    
    # Pull dev=4
    seg = []
    for i in range(n_cyc):
        pull,release = ramps.returnCycle(i,min_force)
        pull_ext_zeroed, pull_force_zeroed = ZeroData(pull, min_force)
        segments_pulls,ruptureForce_pulls = ramps.returnSegmentsOneCycle(pull_force_zeroed, pull_ext_zeroed, pull=True, numsdevs=4, force_threshold=3, window=1000)
        UnzeroData(segments_pulls, pull, min_force)
        seg.append(segments_pulls)

    # Save pickle file
    with open(f'PullSeg_{FN}_dev4.pkl', 'wb') as f:
        # dump the list to the file
        pickle.dump(seg, f)
    print(FN, 'PullSeg dev4')
    
    
    # Release dev=4
    seg_r = []
    for i in range(n_cyc):
        pull,release = ramps.returnCycle(i,min_force)
        release_ext_zeroed, release_force_zeroed = ZeroData(release, min_force)
        segments_releases,ruptureForce_releases = ramps.returnSegmentsOneCycle(release_force_zeroed,release_ext_zeroed, pull=False, numsdevs=4, force_threshold=3, window=1000)
        UnzeroData(segments_releases, release, min_force)
        seg_r.append(segments_releases)

    with open(f'ReleaseSeg_{FN}_dev4.pkl', 'wb') as f:
        # dump the list to the file
        pickle.dump(seg_r, f) 
    print(FN, 'ReleaseSeg dev4')



"""    # Pull dev=5
    seg = []
    for i in range(n_cyc):
        pull,release = ramps.returnCycle(i,min_force)
        pull_ext_zeroed, pull_force_zeroed = ZeroData(pull, min_force)
        segments_pulls,ruptureForce_pulls = ramps.returnSegmentsOneCycle(pull_force_zeroed, pull_ext_zeroed, pull=True,numsdevs=5, force_threshold=3, window=1000)
        UnzeroData(segments_pulls, pull, min_force)
        seg.append(segments_pulls)

    # Save pickle file
    with open(Path_XD+f'/Clustering_States/Processed_Seg/PullSeg_{FN}_dev5.pkl', 'wb') as f:
        # dump the list to the file
        pickle.dump(seg, f)
    print(FN, 'PullSeg dev5')


    # Release dev=5
    seg_r = []
    for i in range(n_cyc):
        pull,release = ramps.returnCycle(i,min_force)
        release_ext_zeroed, release_force_zeroed = ZeroData(release, min_force)
        segments_releases,ruptureForce_releases = ramps.returnSegmentsOneCycle(release_force_zeroed,release_ext_zeroed, pull=False, numsdevs=5, force_threshold=3, window=1000)
        UnzeroData(segments_releases, release, min_force)
        seg_r.append(segments_releases)

    with open(Path_XD+f'/Clustering_States/Processed_Seg/ReleaseSeg_{FN}_dev5.pkl', 'wb') as f:
        # dump the list to the file
        pickle.dump(seg_r, f)    
    print(FN, 'ReleaseSeg dev5')"""